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

import copy
import os
from pathlib import Path
from typing import Dict, List

from nemo_launcher.core.launchers import AutoLauncher
from nemo_launcher.core.stages import NemoMegatronStage, clean_command_groups

FT_PATH = Path("/opt/FasterTransformer")
FT_BACKEND_PATH = Path("/opt/fastertransformer_backend")

# for debugging
FT_PATH_WITH_BUILD = FT_PATH
FT_PATH = Path(os.environ.get("FT_PATH", FT_PATH))


class Export(NemoMegatronStage):
    """
    Stage is a FasterTransformer export stage.
    It includes two steps:
        - NeMo to FasterTransformer checkpoint export.
        - Triton model configuration.
    """

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "export"
        self.stage_cfg = cfg.get("export")

    def _make_checkpoint_search_command(self, **kwargs):
        checkpoint_override = [f"{k}={v}" for k, v in kwargs.items()]
        return (
            f"python3 {self._launcher_scripts_path / 'nemo_launcher/collections/checkpoint_search.py'} "
            f"{' '.join(checkpoint_override)}"
        )

    def make_stage_command_groups(
        self, stage_cfg_path, sub_stage=None,
    ) -> List[List[str]]:
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
        command_groups = [[]]

        command_groups[0] += self._make_sub_stage_command(sub_stage)
        command_groups = clean_command_groups(command_groups)
        return command_groups

    def _make_sub_stage_command(self, sub_stage):
        """
        Make the command group for current stage
        It occupies one bcprun, srun or bash.

        :return: command group for current stage
        :rtype: List[List[str]]
        """
        choice_model_type, choice_name = self.get_stage_config_choice()
        cmds_fn = {
            "convert": {
                "gpt3": self._get_gpt_conversion_cmds,
                "t5": self._get_t5_conversion_cmds,
                "mt5": self._get_t5_conversion_cmds,
            },
        }[sub_stage][choice_model_type]
        return cmds_fn(self.cfg)

    def _make_sub_stages(self):
        sub_stages = ["convert"]
        return sub_stages

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders and fine-tuning/prompt-learning dataset"""
        """Setup required folders and dataset"""
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)
        results_folder = job_path.results_folder
        results_folder.mkdir(parents=True, exist_ok=True)

    def run(self) -> str:
        """Execute export stage"""
        # Setup folders and datasets
        self.setup_folder_and_data()

        sub_stages = self._make_sub_stages()
        job_id = ""
        for sub_stage in sub_stages:
            # Save stage hydra config
            job_path = self.get_job_path(sub_stage)
            job_path.folder.mkdir(parents=True, exist_ok=True)

            stage_cfg_path = NemoMegatronStage.save_stage_hydra_config(
                self.stage_cfg, job_path, self.cfg
            )
            if job_id:
                dependency = f"aftercorr:{job_id}"
                self.stage_cfg["run"]["dependency"] = dependency

            # Make cluster parameters
            cluster_parameters = self._make_cluster_parameters(self.cluster, sub_stage)

            # Make command groups
            command_groups = self.make_stage_command_groups(stage_cfg_path, sub_stage)
            # Create launcher
            launcher = AutoLauncher(
                folder=job_path.folder, cluster=self.cluster, **cluster_parameters,
            )
            job_id = launcher.launch(command_groups=command_groups)

        return job_id

    def _make_cluster_parameters(self, cluster: str, sub_stage=None,) -> Dict:
        """Prepare cluster configuration"""
        cfg = self.cfg
        stage_cfg = self.stage_cfg

        ft_model_cfg = stage_cfg.get("model")
        triton_cfg = stage_cfg.get("triton_deployment")
        run_cfg = stage_cfg.get("run")

        job_name = run_cfg.get("name")
        time_limit = run_cfg.get("time_limit")
        dependency = run_cfg.get("dependency")

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        num_tasks = (
            ft_model_cfg.tensor_model_parallel_size
            * triton_cfg.pipeline_model_parallel_size
        )
        nodes = 1
        ntasks_per_node = 1

        setup = None
        env_vars = self.get_env_vars()
        if env_vars:
            setup = [f"export {k}={v}" for k, v in env_vars.items()]

        cluster_parameters = {}
        shared_parameters = {
            "job_name": job_name,
            "nodes": nodes,
            "time": time_limit,
            "ntasks_per_node": ntasks_per_node,
            "setup": setup,
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
        elif cluster == "bcp":
            cluster_parameters.update(
                {**shared_parameters, "env_vars": env_vars,}
            )
        elif cluster == "interactive":
            cluster_parameters.update(shared_parameters)

        return cluster_parameters

    def _get_gpt_conversion_cmds(self, cfg):
        """ Generate export commands for GPT-3 models"""
        run_cfg = cfg.export.run
        ft_model_cfg = cfg.export.model
        triton_cfg = cfg.export.triton_deployment
        tokenizer_cfg = cfg.training.model.tokenizer

        checkpoint_path = ft_model_cfg.checkpoint_path
        triton_model_dir = triton_cfg.triton_model_dir

        nemo_megatron_scripts_path = Path(cfg.launcher_scripts_path)
        converter_path = FT_PATH / "examples/pytorch/gpt/utils/nemo_ckpt_convert.py"
        prepare_model_config_script_path = (
            nemo_megatron_scripts_path
            / "nemo_launcher/collections/export_scripts/prepare_triton_model_config.py"
        )
        template_path = (
            FT_BACKEND_PATH / "all_models/gpt/fastertransformer/config.pbtxt"
        )

        triton_model_version_dir = f"{triton_model_dir}/1"

        convert_cmd = (
            f"python -u {converter_path} \\\n"
            f" --in-file {checkpoint_path} \\\n"
            f" --saved-dir {triton_model_version_dir} \\\n"
            f" --infer-gpu-num {ft_model_cfg.tensor_model_parallel_size} \\\n"
            f" --weight-data-type {ft_model_cfg.weight_data_type} \\\n"
            f" --vocab-path {tokenizer_cfg.vocab_file} \\\n"
            f" --merges-path {tokenizer_cfg.merge_file} \\\n"
            f" --processes {ft_model_cfg.processes} \\\n"
            f" --load-checkpoints-to-cpu {int(ft_model_cfg.load_checkpoints_to_cpu)}"
        )
        triton_prepare_model_config_cmd = (
            f"python -u {prepare_model_config_script_path} \\\n"
            f" --model-train-name {run_cfg.model_train_name} \\\n"
            f" --template-path {template_path} \\\n"
            f" --ft-checkpoint {triton_model_version_dir}/{ft_model_cfg.tensor_model_parallel_size}-gpu \\\n"
            f" --config-path {triton_model_dir}/config.pbtxt \\\n"
            f" --max-batch-size {triton_cfg.max_batch_size} \\\n"
            f" --pipeline-model-parallel-size {triton_cfg.pipeline_model_parallel_size} \\\n"
            f" --tensor-model-parallel-size {ft_model_cfg.tensor_model_parallel_size} \\\n"
            f" --data-type {triton_cfg.data_type}"
        )
        if triton_cfg.int8_mode:
            triton_prepare_model_config_cmd += " \\\n --int8-mode"
        if triton_cfg.enable_custom_all_reduce:
            triton_prepare_model_config_cmd += " \\\n --enable-custom-all-reduce"
        return [
            (
                f"export PYTHONPATH={FT_PATH}:${{PYTHONPATH}} && \\\n"
                + f"rm -rf {triton_model_dir} && \\\n"  # to not mix old and newly generated FT checkpoint files
                + f"{convert_cmd} && \\\n"
                + triton_prepare_model_config_cmd
            )
        ]

    def _get_t5_conversion_cmds(self, cfg):
        """ Generate export commands for T5/mT5 models"""
        run_cfg = cfg.export.run
        ft_model_cfg = cfg.export.model
        triton_cfg = cfg.export.triton_deployment

        checkpoint_path = ft_model_cfg.checkpoint_path
        triton_model_dir = triton_cfg.triton_model_dir

        nemo_megatron_scripts_path = Path(cfg.launcher_scripts_path)
        converter_path = FT_PATH / "examples/pytorch/t5/utils/nemo_t5_ckpt_convert.py"
        prepare_model_config_script_path = (
            nemo_megatron_scripts_path
            / "nemo_launcher/collections/export_scripts/prepare_triton_model_config.py"
        )
        template_path = FT_BACKEND_PATH / "all_models/t5/fastertransformer/config.pbtxt"

        triton_model_version_dir = f"{triton_model_dir}/1"

        convert_cmd = (
            f"python -u {converter_path} \\\n"
            f" --in-file {checkpoint_path} \\\n"
            f" --saved-dir {triton_model_version_dir} \\\n"
            f" --model-name {run_cfg.model_train_name} \\\n"
            f" --infer-gpu-num {ft_model_cfg.tensor_model_parallel_size} \\\n"
            f" --weight-data-type {ft_model_cfg.weight_data_type} \\\n"
            f" --processes {ft_model_cfg.processes}"
        )
        triton_prepare_model_config_cmd = (
            f"python -u {prepare_model_config_script_path} \\\n"
            f" --model-train-name {run_cfg.model_train_name} \\\n"
            f" --template-path {template_path} \\\n"
            f" --ft-checkpoint {triton_model_version_dir}/{ft_model_cfg.tensor_model_parallel_size}-gpu \\\n"
            f" --config-path {triton_model_dir}/config.pbtxt \\\n"
            f" --max-batch-size {triton_cfg.max_batch_size} \\\n"
            f" --pipeline-model-parallel-size {triton_cfg.pipeline_model_parallel_size} \\\n"
            f" --tensor-model-parallel-size {ft_model_cfg.tensor_model_parallel_size} \\\n"
            f" --data-type {triton_cfg.data_type}"
        )
        if triton_cfg.int8_mode:
            triton_prepare_model_config_cmd += " \\\n --int8-mode"
        if triton_cfg.enable_custom_all_reduce:
            triton_prepare_model_config_cmd += " \\\n --enable-custom-all-reduce"
        return [
            (
                f"export PYTHONPATH={FT_PATH}:${{PYTHONPATH}} && \\\n"
                + f"rm -rf {triton_model_dir} && \\\n"  # to not mix old and newly generated FT checkpoint files
                + f"{convert_cmd} && \\\n"
                + triton_prepare_model_config_cmd
            )
        ]
