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

import torch
import tqdm
from lm_eval import utils
from lm_eval.base import LM
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.modules.common.text_generation_utils import generate, get_computeprob_response
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.get_rank import is_global_rank_zero
from pytorch_lightning.trainer.trainer import Trainer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from .nemo_gpt3_prompt import PromptRequestDataset, setup_trainer_and_model, DDP_initialize

class NeMo_LLAMA_PROMPTLM(LM):
    def __init__(self, args, truncate=False, batch_size=1):
        super().__init__()

        # get nemo megatron
        logging.info(f"**** Building LLaMA Prompt model ...")
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
            tokens, conti_lens, task_ids, *_ = map(list, zip(*batch))
            lens = [len(token) - 1 for token in tokens]  # fake delete last token by reducing input len
            max_len = max(lens)

            tokens_pad = pad_sequence(tokens, batch_first=False, padding_value=eos_id)
            # Add padding to all samples to adapt nemo generate api
            # tokens_pad = torch.cat((tokens_pad, torch.ones((1, len(tokens)), dtype=torch.int) * eos_id), 0)

            new_batch = []
            for token, lenn, conti_len, task_id in zip(tokens_pad.T, lens, conti_lens, task_ids):
                new_batch.append((token, max_len, task_id, lenn, conti_len))

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
        request_ds = PromptRequestDataset(reord.get_reordered(), self.model.tokenizer)
        request_dl = DataLoader(request_ds, collate_fn=pad_collate, batch_size=self.batch_size, shuffle=False)

        def logits_to_results(batch, response):
            input_token_ids_batch, _, _, lens, conti_lens = batch
            batch_size = len(lens)
            assert len(response["token_ids"]) == batch_size, "Response's length not equal to batch size."

            batch_res = []
            for index in range(batch_size):
                inp_len = lens[index]
                conti_len = conti_lens[index]

                inp_token_ids = input_token_ids_batch[index].tolist()[: inp_len + 1]  # recover fake deleted token

                log_probs = response["full_logprob"][index][:inp_len]  # torch.tensor
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
            task_ids = torch.zeros((self.batch_size, 1), device='cuda')
            response = generate(
                model=self.model,
                inputs=inputs,
                task_ids=task_ids,
                tokens_to_generate=1,
                all_probs=True,
                temperature=1.0,
                add_BOS=False,
                top_k=0,
                top_p=0.9,
                greedy=True,
                repetition_penalty=1.0,
                min_tokens_to_generate=0,
            )
            response = get_computeprob_response(self.tokenizer, response, inputs)

            if is_global_rank_zero():
                res.extend(logits_to_results(batch, response))

        return reord.get_original(res) if self.can_access_output() else None

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def greedy_until(self, requests):
        raise NotImplementedError

    def can_access_output(self):
        return is_global_rank_zero()
