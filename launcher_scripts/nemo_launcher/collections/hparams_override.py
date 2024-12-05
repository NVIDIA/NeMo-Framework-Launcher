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
import time

import hydra
from nemo.utils.env_var_parsing import get_envint
from nemo.utils.get_rank import is_global_rank_zero
from omegaconf import OmegaConf


@hydra.main(config_path="conf", config_name="hparams_override", version_base="1.2")
def hparams_override(cfg):
    """
    This script verrides hyper-parameters inside NeMo's `hparams.yaml` and will generate
    a new yaml file called `hparams_override.yaml`. The new yaml file will be
    fed into NeMo conversion scripts to convert training checkpoints to a .nemo
    checkpoint.
    """
    hparams_file = cfg.get("hparams_file")
    if hparams_file is not None:
        assert os.path.exists(hparams_file), hparams_file
        assert os.access(hparams_file, os.R_OK), hparams_file
        output_path = cfg.get("output_path")
        assert os.path.exists(output_path), output_path
        assert os.path.isdir(output_path), output_path
        assert os.access(output_path, os.W_OK), output_path
        hparams_override_file = os.path.join(output_path, "hparams_override.yaml")
        conf = OmegaConf.load(hparams_file)

        # (yaoyu) temporary WAR for stable diffusion legacy
        if "base_learning_rate" in conf.cfg:
            import yaml

            conf_dict = OmegaConf.to_container(conf, resolve=True)
            additional_conf_str = """
            precision: 32
            micro_batch_size: 2
            global_batch_size: 2
            seed: 1234
            resume_from_checkpoint: null
            apex_transformer_log_level: 30
            gradient_as_bucket_view: True
            optim:
              name: fused_adam
              lr: 1e-4
              weight_decay: 0.
              betas:
                - 0.9
                - 0.999
              sched:
                name: WarmupHoldPolicy
                warmup_steps: 10000
                hold_steps: 10000000000000
            """
            additional_conf_dict = yaml.safe_load(additional_conf_str)
            conf_dict["cfg"].update(additional_conf_dict)
            conf = OmegaConf.create(conf_dict)

        else:
            vocab_file = cfg.get("vocab_file")
            merge_file = cfg.get("merge_file")
            tokenizer_model = cfg.get("tokenizer_model")

            if vocab_file is not None:
                conf.cfg.tokenizer.vocab_file = vocab_file
            if merge_file is not None:
                conf.cfg.tokenizer.merge_file = merge_file
            if tokenizer_model is not None:
                conf.cfg.tokenizer.model = tokenizer_model
            if "activations_checkpoint_granularity" in conf.cfg:
                conf.cfg.activations_checkpoint_granularity = None
            if "activations_checkpoint_method" in conf.cfg:
                conf.cfg.activations_checkpoint_method = None
            # if "sequence_parallel" in conf.cfg:
            #     conf.cfg.sequence_parallel = False
            if "optim" in conf.cfg and conf.cfg.optim.name == "mcore_distributed_optim":
                conf.cfg.optim.name = "fused_adam"

        node_rank = get_envint("NODE_RANK", get_envint("GROUP_RANK", 0))
        if node_rank != 0:
            return

        if is_global_rank_zero():
            with open(hparams_override_file, "w") as f:
                OmegaConf.save(config=conf, f=f)
        else:
            wait_time = 0
            while not os.path.exists(hparams_override_file):
                time.sleep(1)
                wait_time += 1
                if wait_time > 60:
                    raise TimeoutError("Timeout waiting for config file to be created.")


if __name__ == "__main__":
    hparams_override()
