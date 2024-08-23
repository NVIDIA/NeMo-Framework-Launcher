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

import datetime
import itertools
import os
import shutil
from typing import List, Tuple

import omegaconf

from src.utils import (
    modify_cfg,
    generate_finetuning_overrides_str,
    submit_slurm_job,
    load_slurm_params_from_sh,
)


def run_search(cfg: omegaconf.dictconfig.DictConfig) -> None:
    """
    Main function for fine-tuning hyperparameter autoconfiguration. Launches
    experiment and result analysis jobs.
    :param omegaconf.dictconfig.DictConfig cfg: main hydra config object.
    :return: None
    """
    # Creating search output directory with the model and current time in name
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    search_cfg = cfg.search_config
    cfg.base_results_dir = os.path.join(
        cfg.base_results_dir, cfg.search_config.model_name, start_time
    )
    # Modifying the base config file
    base_cfg = omegaconf.OmegaConf.load(search_cfg.base_config)
    base_cfg.run.name = "${.model_train_name}"
    base_cfg.run.time_limit = f"{search_cfg.max_minutes_per_run // 60}:{search_cfg.max_minutes_per_run % 60}:00"
    base_cfg.trainer.max_steps = search_cfg.max_steps_per_run
    base_cfg.trainer.val_check_interval = base_cfg.trainer.max_steps
    base_cfg.model.restore_from_path = search_cfg.base_model_path
    base_cfg.exp_manager.resume_if_exists = False
    # Tensorboard logging - always enabled, used to extract metrics
    base_cfg.exp_manager.create_tensorboard_logger = True
    # Weights & Biases logging (optional)
    base_cfg.exp_manager.create_wandb_logger = cfg.wandb.enable
    base_cfg.exp_manager.wandb_logger_kwargs.project = cfg.wandb.project

    cluster_cfg = cfg.get("cluster")
    dst = os.path.join(cfg.launcher_scripts_path, "conf/cluster/bcm.yaml")
    omegaconf.OmegaConf.save(cluster_cfg, dst)
    print(f"Copied cluster config to {dst}")

    base_dir, results_cfgs = generate_grid_search_configs(base_cfg, cfg)

    job_ids = launch_grid_search_configs(base_dir, results_cfgs, cfg)

    launch_result_analysis(cfg, job_ids, start_time)


def generate_grid_search_configs(
    base_cfg: omegaconf.dictconfig.DictConfig, cfg: omegaconf.dictconfig.DictConfig,
) -> Tuple[str, List[str]]:
    """
    Generates the grid of all possible hyperparameter combinations for the given model based on search configuration,
    and stores each distinct combination in a .yaml config file.
    :param dict base_cfg: base configuration of the model to be fine-tuned.
    :param omegaconf.dictconfig.DictConfig cfg: main hydra config object.
    :returns: tuple (base_dir, results_cfgs)
        WHERE
        str base_dir is the path to the directory where the results will be stored.
        List[str] results_cfgs is a list of all the config names that were generated.
    """
    base_dir = f"{cfg.base_results_dir}/candidate_configs"
    os.makedirs(base_dir, exist_ok=True)
    results_cfgs = []

    search_cfg = cfg.get("search_config")
    param_grid = search_cfg.get("param_grid")

    hp_keys, hp_values = param_grid.keys(), param_grid.values()
    # Exploring distinct hyperparameter combinations for every experiment, generating config files
    for exp_ind, exp_values in enumerate(itertools.product(*list(hp_values))):
        kwargs = dict(zip(hp_keys, exp_values))
        candidate_id = f"{search_cfg.model_name}_grid_search_{exp_ind}"
        candidate_cfg = modify_cfg(base_cfg, candidate_id, **kwargs)
        file_name = candidate_id + ".yaml"
        omegaconf.OmegaConf.save(candidate_cfg, os.path.join(base_dir, file_name))
        results_cfgs.append(file_name)
    return base_dir, results_cfgs


def launch_grid_search_configs(
    base_dir: str, results_cfgs: List[str], cfg: omegaconf.dictconfig.DictConfig,
) -> List[int]:
    """
    Launches fine-tuning jobs for hyperparameter autoconfiguration in parallel. The maximum number of
    jobs to launch is specified by limit_search_runs parameter in cfg.search_config.limit_search_runs.
    :param str base_dir: location where the experiment configs are stored.
    :param list results_cfgs: list of config names.
    :param omegaconf.dictconfig.DictConfig cfg: main hydra config object.
    :return: job_ids, list of job ids for all the training jobs.
    :rtype: list[int]
    """
    launcher_scripts_path = cfg.get("launcher_scripts_path")
    search_cfg = cfg.get("search_config")
    limit = search_cfg.get("limit_search_runs")
    results_dir = os.path.join(cfg.base_results_dir, "ft_logs")
    dst_dir = os.path.join(launcher_scripts_path, "conf/peft", search_cfg.model_name)

    job_ids = []
    for cfg_file_name in results_cfgs:
        src_file = os.path.join(base_dir, cfg_file_name)
        dst_file = os.path.join(dst_dir, cfg_file_name)
        shutil.copyfile(src_file, dst_file)
        job_id = run_finetuning_job(dst_file, results_dir, cfg)
        os.remove(dst_file)

        if job_id is not None:
            job_ids.append(job_id)
        if len(job_ids) == limit:
            return job_ids
    return job_ids


def run_finetuning_job(
    config_path: str, results_dir: str, cfg: omegaconf.dictconfig.DictConfig
) -> int:
    """
    Launch a fune-tuning job for a given model name and config file using NeMo Framework Launcher.
    :param str config_path: name of the configuration file to be used for fine-tuning run with NeMo Framework Launcher.
    :param str results_dir: path to the directory where the run results will be stored.
    :param omegaconf.dictconfig.DictConfig cfg: main hydra config object.
    :return: Slurm job_id of the fine-tuning job that was launched.
    :rtype: int
    """
    # Copy cluster config to nemo_framework_launcher.
    launcher_scripts_path = cfg.get("launcher_scripts_path")

    # Generate string of hydra overrides for nemo_framework_launcher.
    file_name = os.path.basename(config_path)
    overrides_str = generate_finetuning_overrides_str(file_name, results_dir, cfg)

    nemo_megatron_ci = (
        "NEMO_LAUNCHER_CI=1" if bool(os.getenv("NEMO_LAUNCHER_CI")) else ""
    )
    main_path = os.path.join(launcher_scripts_path, "main.py")
    cmd = f"HYDRA_FULL_ERROR=1 {nemo_megatron_ci} python3 {main_path} {overrides_str} "

    # Launch job with command cmd.
    job_id = submit_slurm_job(cmd)
    print(f"Submitted fine-tuning script with job id: {job_id}")
    return job_id


def launch_result_analysis(
    cfg: omegaconf.dictconfig.DictConfig, job_ids: List[int], start_time: str
) -> None:
    """
    Launch a result analysis job which creates results.csv file in final_result folder with validation loss for every
      hyperparameter combination.
    :param omegaconf.dictconfig.DictConfig cfg: main hydra config object.
    :param List[int] job_ids: list of Slurm job ids for fine-tuning experiment runs.
    :param str start_time: time in %Y-%m-%d_%H-%M-%S format when the autoconfiguration was run.
    :return: None
    """
    # Creating output directory
    result_dir = cfg.base_results_dir
    out_dir = os.path.join(result_dir, "final_result")
    os.makedirs(out_dir, exist_ok=True)
    # Extracting Slurm arguments .sh launch script, generated by NeMo Launcher
    for root, _, files in os.walk(result_dir):
        shell_files = [file for file in files if file.endswith(".sh")]
        if shell_files:
            shell_script = os.path.join(root, shell_files[0])
            break
    slurm_args, slurm_flags = load_slurm_params_from_sh(shell_script)
    # Slurm overrides
    slurm_args["--job-name"] = f"{cfg.cluster.job_name_prefix}result_analysis"
    slurm_args["--error"] = os.path.join(out_dir, "result_analysis_%j.err")
    slurm_args["--output"] = os.path.join(out_dir, "result_analysis_%j.out")
    slurm_args["--nodes"] = 1
    slurm_args["--ntasks-per-node"] = 1
    slurm_args["--time"] = "00:15:00"
    slurm_args["--dependency"] = ",".join(f"afterany:{job_id}" for job_id in job_ids)
    # Generating run command
    cmd = (
        f"srun "
        f"--container-image {cfg.training_container} "
        f"--container-mounts {cfg.ft_configurator_path}:{cfg.ft_configurator_path} "
        f"bash -c 'python3 {cfg.ft_configurator_path}/src/result_analysis.py "
        f"search_config.start_time={start_time}'"
    )
    # Saving Slurm script
    slurm_script = "\n".join(
        [
            "#!/bin/bash\n",
            "\n".join(
                f"#SBATCH {key}={str(value)}" for key, value in slurm_args.items()
            ),
            "\n".join(f"#SBATCH {flag}" for flag in slurm_flags),
            "\n",
            f"{cmd}",
        ]
    )
    script_path = os.path.abspath(os.path.join(out_dir, "run_analysis.sh"))
    with open(script_path, "w") as f:
        f.write(slurm_script)

    job_id = submit_slurm_job(f"sbatch {script_path}")
    print(f"Started result analysis job with id: {job_id}")
