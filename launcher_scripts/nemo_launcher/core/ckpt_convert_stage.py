# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import functools
import glob, os
import logging
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import omegaconf
from nemo_launcher.core.launchers import AutoLauncher
from nemo_launcher.utils.job_utils import JobPaths
from omegaconf import OmegaConf
from nemo_launcher.core.stages import NeMoStage, clean_command_groups

class ConvertHF2NeMo(NeMoStage):
    """Convert existing Huggingface ckpt from hub to nemo"""    

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "convert_hf2nemo"
        self.stage_cfg = cfg.get("convert_hf2nemo")

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders and fine-tuning/prompt-learning dataset"""
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)

        data_cfg = self.stage_cfg
        self.hf_ckpt_dir = data_cfg.get("hf_ckpt_dir")
        self.nemo_save_dir = data_cfg.get("output_nemo_folder")
        self.tokenizer_file = data_cfg.get("tokenizer_file")

    def _get_nemo_code_path(self, model_type: str) -> Path:
        """
        Provide the essential nemo code path for running the stage, usually different model types use different nemo scripts.
        For example, `megatron_t5_pretraining.py` for t5 and `megatron_gpt_pretraining.py` for gpt3.

        :param str model_type: i.e. `gpt3`, `t5`, `mt5`, etc.
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """
        model_type_to_code_path = {
            "convert_hf_llama_to_nemo": self._convert_hf2nemo_code_path / "nlp_language_modeling/convert_hf_llama_to_nemo.py",
        }
        return model_type_to_code_path[model_type]

    def _make_private_cluster_parameters(self, cluster: str, sub_stage: str) -> Dict:
        """
        A simplifying function to make cluster parameters specific to each cluster type.
        Shared cluster parameters are handled in _make_cluster_parameters.
        This is function is introduced because for different dataset preparation the required slurm params are different,
            but the shared parameters are always the same. As a result, one only needs to override private parameters
            for different DataStage.

        :param str cluster: cluster type
        :param str sub_stage: current sub_stage name
        :return: a dictionary of private cluster parameters, e.g. `bcp_preproc_npernode`
        :rtype: Dict
        """
        cfg = self.cfg
        stage_cfg = self.stage_cfg
        run_cfg = stage_cfg.get("run")

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        node_array_size = run_cfg.get("node_array_size")
        array = run_cfg.get("array")
        bcp_preproc_npernode = run_cfg.get("bcp_preproc_npernode") if sub_stage == "preprocess" else 1
        if cluster == "bcm":
            return {
                "nodes": 1,
                "array": f"{array}%{node_array_size}",
                "container_image": container_image,
                "container_mounts": container_mounts,
            }
        if cluster == "bcp":
            return {
                "nodes": node_array_size,
                "ntasks_per_node": bcp_preproc_npernode,
                "bcp_launcher": "'mpirun --allow-run-as-root'",
            }
        return {}

