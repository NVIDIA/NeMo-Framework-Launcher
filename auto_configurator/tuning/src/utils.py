# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import subprocess
from typing import Dict, List, Tuple, Set, Optional

import omegaconf


def add_container_mounts(container_mounts: Optional[List[str]]) -> str:
    """
    Converts the config container mounts to the right format for an srun command.
    :param Optional[List[str]] container_mounts: list of container mounts as in the config file.
    :return: the string that can be used in the srun command to add the container mounts.
    :rtype: str
    """
    mounts_str = ""
    if container_mounts[0] is None or container_mounts[0] == "None":
        return ""
    if container_mounts is not None:
        assert isinstance(
            container_mounts, omegaconf.listconfig.ListConfig
        ), "container_mounts must be a list."
        for mount in container_mounts:
            if mount is not None and isinstance(mount, str):
                mounts_str += f",{mount}" if ":" in mount else f",{mount}:{mount}"
    return mounts_str


def generate_finetuning_overrides_str(
    file_name: str, results_dir: str, cfg: omegaconf.dictconfig.DictConfig
) -> str:
    """
    Generates string with hydra-like parameter overrides for NeMo Framework Launcher fine-tuning job.
    :param str file_name: name of the .yaml configuration file for NeMo Framework Launcher fine-tuning job.
    :param str results_dir: path to the directory where the results will be stored.
    :param omegaconf.dictconfig.DictConfig cfg: main hydra config object.
    :return: string containing all hydra-like overrides required for the NeMo Framework Launcher fine-tuning job.
    :rtype: str
    """
    model_name = cfg.search_config.model_name
    file_name = file_name.replace(".yaml", "")
    training_model = f"{model_name}/{file_name}"
    cluster_type = cfg.get("cluster_type")
    container = cfg.get("training_container")
    ft_configurator_path = cfg.get("ft_configurator_path")
    ft_configurator_path = os.path.abspath(ft_configurator_path)
    launcher_scripts_path = cfg.get("launcher_scripts_path")
    launcher_scripts_path = os.path.abspath(launcher_scripts_path)
    container_mounts = cfg.get("container_mounts", "null")
    data_dir = cfg.get("data_dir")
    api_key_file = cfg.get("wandb").get("api_key_file")
    if api_key_file is None:
        api_key_file = "null"
    # Process container-mounts.
    model_path = cfg.search_config.base_model_path
    mounts_str = f"{ft_configurator_path}:{ft_configurator_path},{results_dir}:{results_dir},{model_path}:{model_path}"
    mounts_str += add_container_mounts(container_mounts)
    # Process Hydra overrides
    overrides_str = (
        f"peft={training_model} "
        f"stages=[peft] "
        f"peft.exp_manager.create_checkpoint_callback=True "
        f"cluster_type={cluster_type} "
        f"base_results_dir={results_dir} "
        f"\"container='{container}'\" "
        f"launcher_scripts_path={launcher_scripts_path} "
        f"container_mounts=\[{mounts_str}\] "
        f"data_dir={data_dir} "
        f"wandb_api_key_file={api_key_file} "
    )
    return overrides_str


def load_slurm_params_from_sh(sh_path) -> Tuple[Dict[str, str], Set[str]]:
    """
    Extracts sbatch arguments in key: value format and sbatch flags.
    :param str sh_path: path to Slurm job .sh file used with sbatch.
    :returns: tuple (slurm_args, slurm_params)
        WHERE
        Dict[str, str] slurm_args: dictionary of sbatch arguments in key: value format
        Set[str] slurm_flags: set of sbatch flags
    """
    slurm_args = {}
    slurm_params = set()
    with open(sh_path, "r") as f:
        for line in f:
            if line.startswith("#SBATCH"):
                _, key_val = line.strip().split(maxsplit=1)
                if "=" in key_val:
                    key, val = key_val.split("=", maxsplit=1)
                    slurm_args[key] = val
                else:
                    slurm_params.add(key_val)
    return slurm_args, slurm_params


def modify_cfg(
    base_cfg: omegaconf.dictconfig.DictConfig, candidate_id: str, **kwargs
) -> omegaconf.dictconfig.DictConfig:
    """
    Modifies baseline model configuration file, overrides parameters with those in kwargs.
    :param omegaconf.dictconfig.DictConfig: baseline model configuration file to be modified.
    :param str candidate_id: new model_train_name to be used.
    :return: modified model configuration file with values overwriten with those in kwargs.
    :rtype: omegaconf.dictconfig.DictConfig.
    """
    modified_cfg = base_cfg.copy()
    for key, value in kwargs.items():
        omegaconf.OmegaConf.update(modified_cfg, key, value)
    # Recording which values were modified for a specific grid search run
    modified_cfg["overrides"] = kwargs
    omegaconf.OmegaConf.update(modified_cfg, "run.model_train_name", candidate_id)
    return modified_cfg


def submit_slurm_job(cmd: str) -> int:
    """
    Submits Slurm command, returns the ID of created job.
    :param str cmd: command to be run.
    :return: id of created job.
    :rtype: int.
    """
    try:
        job_output = subprocess.check_output([cmd], shell=True).decode("utf-8")
        job_id = job_output.split(" ")[-1].strip()
    except Exception as err:
        job_id = None
        print(err)
    return int(job_id)
