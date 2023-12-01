# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from omegaconf import OmegaConf, open_dict

import torch
import tqdm
from megatron.core import parallel_state
from lm_eval import utils
from lm_eval.base import LM
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.modules.common.text_generation_utils import generate, get_computeprob_response
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.model_utils import inject_model_parallel_rank
from pytorch_lightning.trainer.trainer import Trainer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from .nemo_gpt3 import RequestDataset, setup_trainer_and_model, DDP_initialize

class NeMo_LLAMALM_TP_PP(LM):
    def __init__(self, args, truncate=False, batch_size=1):
        super().__init__()

        # get nemo megatron
        logging.info(f"**** Building LLaMA model ...")
        self.trainer, self.model = setup_trainer_and_model(args)
        self.tokenizer = self.model.tokenizer
        self.model.eval()

        self.max_length = self.model.cfg.get("max_position_embeddings")
        assert self.tokenizer.text_to_ids("hello\n\nhello") == [
            22172,
            13,
            13,
            12199,
        ], "Tokenizer text_to_ids is not working as expected."

        self.truncate = truncate
        self.batch_size = batch_size

        # initialize DDP and move model to GPU
        DDP_initialize(self.model)
        self.model = self.model.cuda()

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config={}):
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(args, **args2)

    def loglikelihood(self, requests):
        return self._loglikelihood(requests)

    """
    request: (context, continuation)
    how this all works:
             CTX      CONT
    inp    0 1 2 3|4 5 6 7 8 9 <- last token is deleted by inp[:, :-1]
    gpt2    \               \
    logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the [:, -len(continuation_enc):, :self.VOCAB_SIZE] slice
    cont_toks      4 5 6 7 8 9
    when too long to fit in context, truncate from the left
    """

    def _loglikelihood(self, requests):
        def pad_collate(batch, eos_id=2):
            tokens = [item[0] for item in batch]
            conti_lens = [item[1] for item in batch]
            lens = [len(token) - 1 for token in tokens]  # fake delete last token by reducing input len
            max_len = max(lens)
            extra_pad_len = 0
            if max_len % 8 != 0:
                extra_pad_len = 8 - (max_len % 8)
                max_len += extra_pad_len
            # extra_pad_len = 2048 - max_len
            # max_len += extra_pad_len

            tokens_pad = pad_sequence(tokens, batch_first=False, padding_value=eos_id)
            if extra_pad_len > 0:
                extra_pad = torch.ones(extra_pad_len, len(batch)) * eos_id
                extra_pad = extra_pad.type_as(tokens_pad)
                tokens_pad = torch.vstack((tokens_pad, extra_pad))
            # Add padding to all samples to adapt nemo generate api

            new_batch = []
            for token, lenn, conti_len in zip(tokens_pad.T, lens, conti_lens):
                # (token, lenn, tokens_to_generate, compute_logprobs)
                new_batch.append((token, max_len, lenn, conti_len))

            new_batch = default_collate(new_batch)
            return new_batch

        def _collate(x):  # used to reorder request and remove duplications
            """
              the negative sign on len(toks) sorts descending - this has a few advantages:
              - time estimates will always be over not underestimates, which is more useful for planning
              - to know the size of a batch when going through the list, you know the first one is always the batch padded context length.
                this is useful to simplify the batching logic and more importantly to make automatic adaptive batches much much easier to implement
              - any OOMs will happen right away rather than near the end
            """
            toks = x[0] + x[1]
            return -len(toks), tuple(toks)

        reord = utils.Reorderer(requests, _collate)
        request_ds = RequestDataset(reord.get_reordered(), self.model.tokenizer, self.max_length)
        request_dl = DataLoader(request_ds, collate_fn=pad_collate, batch_size=self.batch_size, shuffle=False)

        def logits_to_results(batch, response):
            input_token_ids_batch, _, lens, conti_lens = batch
            batch_size = len(lens)
            assert len(response['token_ids']) == batch_size, "Response's length not equal to batch size."

            batch_res = []
            for index in range(batch_size):
                inp_len = lens[index]
                conti_len = conti_lens[index]

                inp_token_ids = input_token_ids_batch[index].tolist()[: inp_len + 1]  # recover fake deleted token
                response_token_ids = response['token_ids'][index][:inp_len]

                assert response_token_ids == inp_token_ids[:-1], f"Mismatch in input tokens."

                log_probs = response['full_logprob'][index][:inp_len]  # torch.tensor
                log_probs = log_probs[-conti_len:]

                greedy_tokens = log_probs.argmax(dim=-1)
                greedy_tokens = self.tokenizer.ids_to_tokens(greedy_tokens.cpu().numpy().tolist())

                conti_token_ids = inp_token_ids[-conti_len:]
                conti_tokens = self.tokenizer.ids_to_tokens(conti_token_ids)

                max_equal = greedy_tokens == conti_tokens
                log_probs = log_probs.cpu().to(torch.float32)
                conti_enc = torch.tensor(self.tokenizer.tokens_to_ids(conti_tokens))
                conti_probs = torch.gather(log_probs, 1, conti_enc.unsqueeze(-1)).squeeze(-1)

                batch_res.append((float(conti_probs.sum()), bool(max_equal), greedy_tokens, conti_tokens))
            return batch_res

        res = []
        for batch in tqdm.tqdm(request_dl):
            # inputs = (token_ids, conti_lens)
            inputs = (batch[0].cuda(), batch[1].cuda())
            response = generate(
                model=self.model,
                inputs=inputs,
                tokens_to_generate=1,
                all_probs=True,
                temperature=1.0,
                add_BOS=False,
                top_k=0,
                top_p=0.9,
                greedy=True,
                repetition_penalty=1.0,
                min_tokens_to_generate=0,
                compute_logprob=True,
                end_strings=['</s>'],
            )
            response = get_computeprob_response(self.tokenizer, response, inputs)

            if is_global_rank_zero():
                res.extend(logits_to_results(batch, response))

            del inputs, response

        return reord.get_original(res) if self.can_access_output() else None

    def loglikelihood_rolling(self, requests):
        loglikelihoods = []
        len_rolling_token_windows = [0]
        all_rolling_token_windows = []

        for (string,) in requests:
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tokenizer.text_to_ids(string),
                        prefix_token=2,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            len_rolling_token_windows.append(len(rolling_token_windows) + len_rolling_token_windows[-1])
            all_rolling_token_windows.extend(rolling_token_windows)

        string_nll = self._loglikelihood(all_rolling_token_windows)
        if self.can_access_output():
            string_nll = [x[0] for x in string_nll]
            # discard is_greedy
            for i in range(len(len_rolling_token_windows) - 1):
                loglikelihoods.append(sum(string_nll[len_rolling_token_windows[i] : len_rolling_token_windows[i + 1]]))

        return loglikelihoods

    def greedy_until(self, requests):
        raise NotImplementedError

    def can_access_output(self):
        return is_global_rank_zero()
