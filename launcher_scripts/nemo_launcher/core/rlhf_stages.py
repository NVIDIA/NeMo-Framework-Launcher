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
import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import omegaconf
from nemo_launcher.core.launchers import AutoLauncher
from nemo_launcher.core.stages import NeMoStage, clean_command_groups
from nemo_launcher.utils.job_utils import JobPaths
from omegaconf import OmegaConf


class RLHFRewardModel(NeMoStage):
    """Stage class of rlhf_rm with NeMo scripts"""

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "rlhf_rm"
        self.stage_cfg = cfg.get("rlhf_rm")

    def _get_nemo_code_path(self, model_type: str) -> Path:
        """
        Provide the essential nemo code path for running the stage, usually different model types use different nemo scripts.
        For example, `megatron_t5_pretraining.py` for t5 and `megatron_gpt_pretraining.py` for gpt3.

        :param str model_type: i.e. `gpt3`, `t5`, `mt5`, etc.
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """
        model_type_to_code_path = {
            "gpt3": self._rlhf_code_path / "examples/nlp/gpt/train_reward_model.py",
        }
        return model_type_to_code_path[model_type]


class RLHFPPO(NeMoStage):
    """Stage class of rlhf_rm with NeMo scripts"""

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "rlhf_ppo"
        self.stage_cfg = cfg.get("rlhf_ppo")

    def get_env_vars(self) -> Dict:
        """
        Set up dictionary for environment variables
        The environment variables from hydra config will be set inside the job scripts.
        For Example:
            Set `env_vars.NVTE_BIAS_DROPOUT_FUSION=1` while calling nemo_launcherlauncher-scripts,
            `NVTE_BIAS_DROPOUT_FUSION=1` will be set while running the job.

        :return: a dictionary of env vars while running the job.
        :rtype: Dict
        """
        env_vars = {k: v for k, v in self.cfg.get("env_vars").items() if v is not None}
        return env_vars

    def _make_cluster_parameters(self, cluster: str) -> Dict:
        """
        Make a cluster-specific parameters for jobs on different clusters.
        Current clusters include bcm(slurm).
        For example for bcm, it will return slurm parameters:
            {'job_name': 'some_name', 'nodes': 2, 'ntasks_per_node': 8, ...}

        :return: a dictionary of cluster parameters, e.g. `ntasks_per_node`
        :rtype: Dict
        """
        cfg = self.cfg
        stage_cfg = self.stage_cfg
        run_cfg = stage_cfg.get("run")
        time_limit = run_cfg.get("time_limit")
        dependency = run_cfg.get("dependency")
        subcfg_list = [
            "reward_model_server",
            "initial_policy_server",
            "critic_server",
            "actor",
        ]

        job_name = run_cfg.get("name")

        nodes = []
        for subcfg in subcfg_list:
            nodes.append(stage_cfg.get(subcfg).get("trainer").get("num_nodes"))

        ntasks_per_node = []
        for subcfg in subcfg_list:
            ntasks_per_node.append(stage_cfg.get(subcfg).get("trainer").get("devices"))

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        setup = None
        env_vars = self.get_env_vars()
        for i in range(3):
            env_vars[
                f"HETJOB{i}_HOST"
            ] = f"$(scontrol show hostnames=$SLURM_JOB_NODELIST_HET_GROUP_{i} | head -n1)"
        if env_vars:
            setup = [f"export {k}={v}" for k, v in env_vars.items()]

        cluster_parameters = {}
        shared_parameters = {
            "job_name": job_name,
            "nodes": nodes,
            "time": time_limit,
            "ntasks_per_node": ntasks_per_node,
            "setup": setup,
            "heterogeneous": True,
        }
        if cluster == "bcm":
            cluster_cfg = cfg.get("cluster")
            slurm_cfg = {**copy.deepcopy(cluster_cfg)}
            job_name_prefix = slurm_cfg.pop("job_name_prefix")
            cluster_parameters = {**slurm_cfg}
            cluster_parameters.update(
                {
                    **shared_parameters,
                    "dependency": dependency,
                    "container_image": container_image,
                    "container_mounts": container_mounts,
                }
            )
            cluster_parameters["job_name"] = (
                job_name_prefix + cluster_parameters["job_name"]
            )

        return cluster_parameters

    def _cuda_visible_devices(self, cfg_name) -> str:
        ntasks_per_node = self.stage_cfg.run.get("ntasks_per_node")
        if ntasks_per_node is None:
            ntasks_per_node = self.stage_cfg.get(cfg_name).trainer.get("devices", 1)
        return (
            "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
            if ntasks_per_node == 8
            else f"CUDA_VISIBLE_DEVICES={','.join(map(str, range(ntasks_per_node)))}"
        )

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """
        Make the command groups for current stage
        Command groups is a list of command group. A command group is defined as:
              0. Command group is a list of command strings
              1. Each command group occupies one bcprun, srun or bash
              2. Each command group eventually has multiple commands connected by ";"

        :param Path stage_cfg_path: path to interpolated and saved configuration
        :return: command groups for current stage
        :rtype: List[List[str]]
        """
        command_groups = []
        subcfg_list = [
            "reward_model_server",
            "initial_policy_server",
            "critic_server",
            "actor",
        ]
        code_path_list = [
            self._rlhf_code_path / "examples/nlp/gpt/serve_reward_model.py",
            self._rlhf_code_path / "examples/nlp/gpt/serve_initial_policy.py",
            self._rlhf_code_path / "examples/nlp/gpt/serve_ppo_critic.py",
            self._rlhf_code_path / "examples/nlp/gpt/train_gpt_ppo_actor.py",
        ]

        for i, code_path in enumerate(code_path_list):
            command = self._make_wandb_login_command()
            command += self._make_nemo_path_command()
            core_command = [
                self._cuda_device_max_connections,
                self._cuda_visible_devices(subcfg_list[i]),
                self._set_ln_sm_margin,
                self._skip_ag_overlap,
                self._nvte_bias_gelu_nvfusion,
            ]
            nemo_cammnd = [
                f"python3 -u {code_path} ",
                f"--config-path={stage_cfg_path.parents[0]}",
                f"--config-name={stage_cfg_path.name}",
            ]
            if i == 3:
                nemo_cammnd += [
                    "actor.model.rlhf.reward_model.ip=${HETJOB0_HOST}",
                    "actor.model.rlhf.initial_policy.ip=${HETJOB1_HOST}",
                    "actor.model.rlhf.critic.ip=${HETJOB2_HOST}",
                ]
            nemo_call_string = " \\\n  ".join(nemo_cammnd)
            core_command += [
                self._make_api_log_command_prefix(
                    results_dir=self.get_job_path().results_folder
                ),
                self._make_nsys_command_prefix(
                    results_dir=self.get_job_path().results_folder
                ),
                nemo_call_string,
            ]
            core_command_string = " ".join([c for c in core_command if c])
            command += [core_command_string]
            command_groups.append(command)

        command_groups = clean_command_groups(command_groups)

        return command_groups
