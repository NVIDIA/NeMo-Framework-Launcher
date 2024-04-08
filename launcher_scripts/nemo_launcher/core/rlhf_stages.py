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
import shutil
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
            "critic",
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
        for i in range(2):
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
        elif cluster == "k8s":
            cluster_cfg = cfg.get("cluster")
            k8s_cfg = {**copy.deepcopy(cluster_cfg)}

            cluster_parameters = {**k8s_cfg}
            cluster_parameters.update(
                {
                    **shared_parameters,
                    "container_image": container_image,
                    "env_vars": env_vars,
                }
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

    def save_stage_hydra_config(
        self, stage_cfg: OmegaConf, job_path: JobPaths, cfg: OmegaConf
    ) -> Path:
        """
        Interpolate and save hydra config file for current stage
        :param OmegaConf stage_cfg: current stage's hydra configuration
        :param JobPaths job_path: JobPaths object
        :param OmegaConf cfg: base config for job
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """
        # Since k8s uses a Helm chart that launches a job based on the Hydra config
        # file, the Hydra config file that is generated needs to contain all of the
        # required keys for each stage.
        temp_config = OmegaConf.to_object(stage_cfg)
        critic_conf = temp_config["critic"]
        actor_conf = temp_config["actor"]

        filename = "gpt_ppo_critic.yaml"
        cfg_save_path = os.path.join(job_path.folder, filename)
        omegaconf.OmegaConf.save(critic_conf, cfg_save_path)

        filename = "gpt_ppo_actor.yaml"
        cfg_save_path = os.path.join(job_path.folder, filename)
        omegaconf.OmegaConf.save(actor_conf, cfg_save_path)

        # This path is usless for the subsequence jobs, so it's fine to return last conf file.
        return Path(cfg_save_path)

    def _make_k8s_spec_file(
        self, template_root: str, cluster_parameters: Dict, job_path: JobPaths
    ):
        """
        Create a spec file for a Kubernetes RLHF PPO job.
        The spec file is generated based on the parameters in the cluster and conversion config files.
        :param str template_root: path to where the k8s template files are located
        :param Dict cluster_parameters: settings specific to the cluster that is being used
        :param JobPaths job_path: JobPaths object
        """
        with open(os.path.join(template_root, "values.yaml")) as value_file:
            values_template = OmegaConf.load(value_file)

        values_template.image.trainingImage = cluster_parameters["container_image"]
        values_template.image.pullSecret = cluster_parameters["pull_secret"]
        values_template.trainingConfig.shmSize = cluster_parameters["shm_size"]
        values_template.trainingConfig.NFSServer = cluster_parameters["nfs_server"]
        values_template.trainingConfig.NFSPath = cluster_parameters["nfs_path"]
        values_template.trainingConfig.ibResourceName = cluster_parameters[
            "ib_resource_name"
        ]
        values_template.trainingConfig.ibCount = cluster_parameters["ib_count"]
        values_template.trainingConfig.envVars = cluster_parameters["env_vars"]

        if cluster_parameters["dns_policy"] is not None:
            values_template.trainingConfig.dnsPolicy = cluster_parameters["dns_policy"]

        if self.cfg.wandb_api_key_file is not None:
            values_template.trainingConfig.wandbKey = self._add_wandb_key_to_chart()

        values_template.trainingConfig.namespace = cluster_parameters["namespace"]
        values_template.actor.numGPUs = self.stage_cfg.actor.trainer.devices
        values_template.actor.nodes = self.stage_cfg.actor.trainer.num_nodes
        values_template.critic.numGPUs = self.stage_cfg.critic.trainer.devices
        values_template.critic.nodes = self.stage_cfg.critic.trainer.num_nodes

        k8s_template_path = job_path.folder
        k8s_template_file = Path(k8s_template_path / "k8s_template" / "values.yaml")
        k8s_template_file.parent.mkdir(parents=True, exist_ok=True)

        conf = OmegaConf.create(values_template)
        OmegaConf.save(conf, k8s_template_file)

    def _copy_k8s_helm_chart(self, template_root: str, job_path: JobPaths):
        """
        Copy the k8s Helm charts to the results directory.
        :param str template_root: path to where the k8s template files are located
        :param JobPaths job_path: JobPaths object
        """
        chart_file = os.path.join(template_root, "Chart.yaml")
        template_file_critic = os.path.join(template_root, "rlhf-ppo-critic.yaml")
        rlhf_ppo_critic_path = Path(
            job_path.folder / "k8s_template" / "templates" / "rlhf-ppo-critic.yaml"
        )

        template_file_actor = os.path.join(template_root, "rlhf-ppo-actor.yaml")
        rlhf_ppo_actor_path = Path(
            job_path.folder / "k8s_template" / "templates" / "rlhf-ppo-actor.yaml"
        )
        rlhf_ppo_actor_path.parent.mkdir(parents=True, exist_ok=True)

        config_path = Path(job_path.folder / "k8s_template" / "config")
        config_path.mkdir(parents=True, exist_ok=True)
        chart_path = Path(job_path.folder / "k8s_template" / "Chart.yaml")
        rlhf_ppo_config_file = os.path.join(template_root, "rlhf-ppo-config.yaml")
        rlhf_ppo_config_path = Path(
            job_path.folder / "k8s_template" / "templates" / "rlhf-ppo-config.yaml"
        )
        hydra_config_path = Path(job_path.folder / "k8s_template" / "config")

        shutil.copy2(template_file_critic, rlhf_ppo_critic_path)
        shutil.copy2(template_file_actor, rlhf_ppo_actor_path)
        shutil.copy2(chart_file, chart_path)
        shutil.copy2(rlhf_ppo_config_file, rlhf_ppo_config_path)

        config_file_list = [
            "gpt_ppo_critic.yaml",
            "gpt_ppo_actor.yaml",
        ]
        for i, config_path in enumerate(config_file_list):
            shutil.copy2(os.path.join(job_path.folder, config_path), hydra_config_path)

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
            "critic",
            "actor",
        ]
        code_path_list = [
            self._rlhf_code_path / "examples/nlp/gpt/serve_ppo_critic.py",
            self._rlhf_code_path / "examples/nlp/gpt/train_gpt_ppo_actor.py",
        ]

        cfg_name_list = [
            "gpt_ppo_critic.yaml",
            "gpt_ppo_actor.yaml",
        ]

        for i, (code_path, cfg_name) in enumerate(zip(code_path_list, cfg_name_list)):
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
                f"--config-name={cfg_name}",
            ]
            if i == 1:
                nemo_cammnd += [
                    "remote_critic_rm.critic.ip=${HETJOB0_HOST}",
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
