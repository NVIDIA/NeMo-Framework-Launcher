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


class RequestDataset(Dataset):
    def __init__(self, requests, tokenizer) -> None:
        super().__init__()
        self.requests = requests
        self.tokenizer = tokenizer
        self.max_length = 2048

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, index):
        context, continuation = self.requests[index]
        context_enc = self.tokenizer.text_to_ids(context) if isinstance(context, str) else context
        continuation_enc = self.tokenizer.text_to_ids(continuation) if isinstance(continuation, str) else continuation
        # sanity check
        assert len(context_enc) > 0
        assert len(continuation_enc) > 0
        assert len(continuation_enc) <= self.max_length

        conti_len = len(continuation_enc)
        inp_enc = torch.tensor((context_enc + continuation_enc)[-(self.max_length + 1) :])
        return inp_enc, conti_len


def setup_trainer_and_model(args):
    """Setup model and optimizer."""
    torch.set_grad_enabled(False)

    assert args.nemo_model is not None or (
        args.checkpoint_folder is not None and args.checkpoint_name is not None
    ), "Path to checkpoints is required."

    # cast precision to int if 32 or 16
    if args.precision in ["32", "16"]:
        args.precision = int(args.precision)

    model_parallel_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    num_nodes = max(model_parallel_size // 8, 1)
    num_gpus = min(model_parallel_size, 8)
    trainer = Trainer(
        strategy=NLPDDPStrategy(), devices=num_gpus, num_nodes=num_nodes, precision=args.precision, accelerator='gpu'
    )

    app_state = AppState()
    if args.tensor_model_parallel_size > 1 or args.pipeline_model_parallel_size > 1:
        app_state.model_parallel_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
        (
            app_state.tensor_model_parallel_rank,
            app_state.pipeline_model_parallel_rank,
            app_state.model_parallel_size,
            *_,
        ) = fake_initialize_model_parallel(
            world_size=app_state.model_parallel_size,
            rank=trainer.global_rank,
            tensor_model_parallel_size_=args.tensor_model_parallel_size,
            pipeline_model_parallel_size_=args.pipeline_model_parallel_size,
        )

    if args.nemo_model is not None and args.nemo_model != "None":
        logging.info(f"**** Loading checkpoint from nemo model: {args.nemo_model}")
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(args.nemo_model):
            save_restore_connector._model_extracted_dir = args.nemo_model
        pretrained_cfg = MegatronGPTModel.restore_from(
            restore_path=args.nemo_model,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        OmegaConf.set_struct(pretrained_cfg, True)
        with open_dict(pretrained_cfg):
            pretrained_cfg.sequence_parallel = False
            pretrained_cfg.activations_checkpoint_granularity = None
            pretrained_cfg.activations_checkpoint_method = None
            pretrained_cfg.precision = trainer.precision
            if trainer.precision == "16":
                pretrained_cfg.megatron_amp_O2 = False
        model = MegatronGPTModel.restore_from(
            restore_path=args.nemo_model,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=save_restore_connector,
            map_location=f'cuda:{trainer.local_rank}',
        )

    else:
        if args.tensor_model_parallel_size > 1 or args.pipeline_model_parallel_size > 1:
            app_state.pipeline_model_parallel_size = args.pipeline_model_parallel_size
            app_state.tensor_model_parallel_size = args.tensor_model_parallel_size

        logging.info(f"**** Loading checkpoint from {args.checkpoint_folder} - {args.checkpoint_name}")
        # inject model parallel rank
        checkpoint_path = inject_model_parallel_rank(os.path.join(args.checkpoint_folder, args.checkpoint_name))
        model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer)

    model.freeze()

    return trainer, model


def DDP_initialize(model):
    if parallel_state.is_unitialized():

        def dummy():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

        if model.cfg.get('transformer_engine', False) and model.cfg.get('tensor_model_parallel_size', 1) > 1:
            logging.info(f'Setting up transformer engine modules for tensor parallelism.')
            if model.cfg.get('megatron_amp_O2', 'False'):
                # when using O2 additional module key is added that casts the weights
                for layer in model.model.module.language_model.encoder.layers:
                    layer.set_tensor_parallel_group(parallel_state.get_tensor_model_parallel_group())

            else:
                for layer in model.model.language_model.encoder.layers:
                    layer.set_tensor_parallel_group(parallel_state.get_tensor_model_parallel_group())


class NeMo_GPT3LM_TP_PP(LM):
    def __init__(self, args, truncate=False, batch_size=1):
        super().__init__()

        # get nemo megatron
        logging.info(f"**** Building GPT model ...")
        self.trainer, self.model = setup_trainer_and_model(args)
        self.tokenizer = self.model.tokenizer
        self.model.eval()

        self.max_length = self.model.cfg.get("max_position_embeddings")
        assert self.tokenizer.text_to_ids("hello\n\nhello") == [
            31373,
            198,
            198,
            31373,
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
        def pad_collate(batch, eos_id=50256):
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
        request_ds = RequestDataset(reord.get_reordered(), self.model.tokenizer)
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
                compute_logprob=True,
                repetition_penalty=1.0,
                min_tokens_to_generate=0,
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
                        prefix_token=50256,
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
