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
import glob
import os
import random
import gzip
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import omegaconf
from nemo_launcher.core.launchers import AutoLauncher
from nemo_launcher.core.stages import (
    NemoMegatronStage,
    clean_command_groups,
    create_args_list,
)
from nemo_launcher.utils.file_utils import download_single_file
from nemo_launcher.utils.job_utils import JobPaths


class DataStage(NemoMegatronStage):
    """
    DataStage is base class for data preprocessing stages.
    It can hold multiple sub-stages. For example, preparing the Pile dataset includes data downloading,
        extraction and data preprocessing. They have dependencies on each other and will be launched one by one.
    """

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "data_preparation"
        self.stage_cfg = cfg.get("data_preparation")

    def _make_sub_stages(self):
        raise NotImplementedError

    def _make_sub_stage_command(self, sub_stage: str) -> List[str]:
        raise NotImplementedError

    def run(self) -> str:
        """
        Run current stage including all of the substages, returns job id on slurm based system otherwise empty string

        :return: job id on slurm based system otherwise empty string
        :rtype: str
        """
        # Setup folders and datasets
        self.setup_folder_and_data()

        sub_stages = self._make_sub_stages()
        job_id = ""
        for sub_stage in sub_stages:
            # Save stage hydra config
            job_path = self.get_job_path(sub_stage)
            job_path.folder.mkdir(parents=True, exist_ok=True)

            stage_cfg_path = self.save_stage_hydra_config(
                self.stage_cfg, job_path, self.cfg
            )
            if job_id:
                dependency = f"aftercorr:{job_id}"
                self.stage_cfg["run"]["dependency"] = dependency

            # Make cluster parameters
            cluster_parameters = self._make_cluster_parameters(self.cluster, sub_stage)

            # Make command groups
            command_groups = self.make_stage_command_groups(stage_cfg_path, sub_stage)

            # Prepare Helm chart for k8s
            if self.cluster == "k8s":
                template_root = os.path.join(
                    os.path.abspath(os.path.dirname(__file__)),
                    "k8s_templates/data_preparation",
                )
                self._make_k8s_helm_chart(
                    template_root, cluster_parameters, job_path, sub_stage
                )

            # Create launcher
            launcher = AutoLauncher(
                folder=job_path.folder, cluster=self.cluster, **cluster_parameters,
            )

            if self.cluster == "k8s":
                # For k8s clusters, only launch on the final stage (preprocess) as
                # the Helm chart contains all stages in a single chart.
                if sub_stage == sub_stages[-1]:
                    job_id = launcher.launch(command_groups=command_groups)
                else:
                    job_id = ""
            else:
                job_id = launcher.launch(command_groups=command_groups)

        return job_id

    def make_stage_command_groups(
        self, stage_cfg_path: Path, sub_stage: Optional = None,
    ) -> List[List[str]]:
        """
        Make the command groups for current stage
        Command groups is a list of command group. A command group is defined as:
              0. Command group is a list of command strings
              1. Each command group occupies one bcprun, srun or bash
              2. Each command group eventually has multiple commands connected by ";"

        :param Path stage_cfg_path: path to interpolated and saved configuration
        :param Optional sub_stage: current sub_stage name
        :return: command groups for current stage
        :rtype: List[List[str]]
        """

        command_groups = [[]]

        command_groups[0] += self._make_sub_stage_command(sub_stage)
        command_groups = clean_command_groups(command_groups)
        return command_groups

    def _make_private_cluster_parameters(self, cluster, sub_stage):
        raise NotImplementedError

    def _make_cluster_parameters(
        self, cluster: str, sub_stage: Optional = None,
    ) -> Dict:
        """
        Make a cluster-specific parameters for jobs on different clusters.
        Current clusters include bcm(slurm), bcp, k8s, and interactive.
        For example for bcm, it will return slurm parameters:
            {'job_name': 'some_name', 'nodes': 2, 'ntasks_per_node': 8, ...}

        :param str cluster: i.e. `bcm`, `bcp`, `interactive`, `k8s`, etc.
        :param Optional sub_stage: current sub_stage name
        :return: a dictionary of cluster parameters, e.g. `ntasks_per_node`
        :rtype: Dict
        """
        cfg = self.cfg
        stage_cfg = self.stage_cfg

        run_cfg = stage_cfg.get("run")
        job_name = run_cfg.get("name")
        time_limit = run_cfg.get("time_limit")
        dependency = run_cfg.get("dependency")

        env_vars = self.get_env_vars()
        env_vars[
            "PYTHONPATH"
        ] = f"{self._launcher_scripts_path}:${{PYTHONPATH}}"  # Required by pile download
        setup = [f"export {k}={v}" for k, v in env_vars.items()]

        cluster_parameters = {}
        shared_parameters = {
            "job_name": job_name,
            "time": time_limit,
            "setup": setup,
        }
        private_parameters = self._make_private_cluster_parameters(cluster, sub_stage,)
        if cluster == "bcm":
            cluster_cfg = cfg.get("cluster")
            slurm_cfg = {**copy.deepcopy(cluster_cfg)}
            job_name_prefix = slurm_cfg.pop("job_name_prefix")
            cluster_parameters = {
                **slurm_cfg,
                "dependency": dependency,
            }
            cluster_parameters.update(
                {**shared_parameters, **private_parameters,}
            )
            cluster_parameters["job_name"] = (
                job_name_prefix + cluster_parameters["job_name"]
            )
        elif cluster == "bcp":
            cluster_parameters.update(
                {
                    **shared_parameters,
                    **private_parameters,
                    "no_redirect": cfg.get("bcp_no_redirect"),
                }
            )
        elif cluster == "k8s":
            cluster_cfg = cfg.get("cluster")
            container_image = cfg.get("container")
            k8s_cfg = {**copy.deepcopy(cluster_cfg)}

            cluster_parameters = {**k8s_cfg}

            cluster_parameters.update(
                {
                    **shared_parameters,
                    **private_parameters,
                    "container_image": container_image,
                }
            )
        elif cluster == "interactive":
            # cluster_parameters.update(shared_parameters)
            raise ValueError("Data preparation is not supported in interactive mode.")

        return cluster_parameters

    def _make_k8s_helm_chart(
        self,
        template_root: str,
        cluster_parameters: Dict,
        job_path: JobPaths,
        sub_stage: str,
    ):
        """
        Create a Helm chart for data preparation.
        The Helm chart uses a base template which is extended with user-defined
        cluster settings as specified in the config files. The generated Hydra
        config file needs to be copied to the Helm chart as this will be used
        for launching the job.

        :param str template_root: the path to where the k8s template files are located.
        :param dict cluster_parameters: additional parameters specific to the cluster config.
        :param JobPaths job_path: the path to the job results directory.
        :param str sub_stage: the current stage.
        """
        with open(os.path.join(template_root, "values.yaml")) as value_file:
            values_template = omegaconf.OmegaConf.load(value_file)

        procs_per_node = (
            self.stage_cfg.run.bcp_preproc_npernode if sub_stage == "preprocess" else 1
        )
        total_processes = procs_per_node * self.stage_cfg.run.node_array_size

        # Update the Helm chart template with the user-specified settings
        values_template.image.trainingImage = cluster_parameters["container_image"]
        values_template.image.pullSecret = cluster_parameters["pull_secret"]
        values_template.image.nodes = self.stage_cfg.run.node_array_size
        values_template.dataPrepConfig.shmSize = cluster_parameters["shm_size"]
        values_template.dataPrepConfig.NFSServer = cluster_parameters["nfs_server"]
        values_template.dataPrepConfig.NFSPath = cluster_parameters["nfs_path"]
        values_template.dataPrepConfig.totalProcesses = total_processes
        values_template.dataPrepConfig.procsPerNode = procs_per_node
        values_template.dataPrepConfig.stage = sub_stage

        if cluster_parameters["dns_policy"] is not None:
            values_template.dataPrepConfig.dnsPolicy = cluster_parameters["dns_policy"]

        k8s_template_path = job_path.folder
        k8s_template_file = Path(k8s_template_path / "k8s_template" / "values.yaml")
        k8s_template_file.parent.mkdir(parents=True, exist_ok=True)

        conf = omegaconf.OmegaConf.create(values_template)
        omegaconf.OmegaConf.save(conf, k8s_template_file)

        # Copy the data prep spec files to the Helm chart
        template_file = os.path.join(template_root, "data-prep.yaml")
        chart_file = os.path.join(template_root, "Chart.yaml")
        data_prep_path = Path(
            job_path.folder / "k8s_template" / "templates" / "data-prep.yaml"
        )
        data_prep_path.parent.mkdir(parents=True, exist_ok=True)
        config_path = Path(job_path.folder / "k8s_template" / "config")
        config_path.mkdir(parents=True, exist_ok=True)
        chart_path = Path(job_path.folder / "k8s_template" / "Chart.yaml")
        data_prep_config_file = os.path.join(template_root, "data-prep-config.yaml")
        data_prep_config_path = Path(
            job_path.folder / "k8s_template" / "templates" / "data-prep-config.yaml"
        )
        hydra_config_path = Path(job_path.folder / "k8s_template" / "config")

        shutil.copy2(template_file, data_prep_path)
        shutil.copy2(chart_file, chart_path)
        shutil.copy2(data_prep_config_file, data_prep_config_path)
        shutil.copy2(job_path.config_file, hydra_config_path)


class PileDataPreparation(DataStage):
    """DataStage for preparing the Pile dataset for gpt3 and t5"""

    def _make_sub_stages(self) -> List[str]:
        """
        Create a list of sub-stage names which are required to run in current data stage.
        Based on the input config, some of sub stages may not need to run.

        :return: a list of sub-stage names which are required to run
        :rtype: List[str]
        """
        sub_stages = []
        if self.stage_cfg.get("download_the_pile", False):
            sub_stages += ["download", "extract"]
        if self.stage_cfg.get("preprocess_data", False):
            sub_stages += ["preprocess"]
        return sub_stages

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders and fine-tuning/prompt-learning dataset"""
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)

        data_cfg = self.stage_cfg
        download_vocab_url = data_cfg.get("download_vocab_url")
        download_merges_url = data_cfg.get("download_merges_url")
        vocab_save_dir = data_cfg.get("vocab_save_dir")
        merges_save_dir = data_cfg.get("merges_save_dir")
        download_tokenizer_url = data_cfg.get("download_tokenizer_url")
        tokenizer_save_dir = data_cfg.get("tokenizer_save_dir")

        if download_tokenizer_url is not None:
            assert (
                tokenizer_save_dir is not None
            ), "tokenizer_save_dir must be a valid path."
            download_single_file(
                url=download_tokenizer_url,
                save_dir=tokenizer_save_dir,
                file_name="llama_tokenizer.model",
            )

        # Download vocab
        if download_vocab_url is not None:
            assert vocab_save_dir is not None, "vocab_save_dir must be a valid path."
            download_single_file(
                url=download_vocab_url,
                save_dir=vocab_save_dir,
                file_name="vocab.json"
                if download_vocab_url.endswith("json")
                else "vocab.txt",
            )
        # Download merges
        if download_merges_url is not None:
            assert merges_save_dir is not None, "merges_save_dir must be a valid path."
            download_single_file(
                url=download_merges_url,
                save_dir=merges_save_dir,
                file_name="merges.txt",
            )

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
        bcp_preproc_npernode = (
            run_cfg.get("bcp_preproc_npernode") if sub_stage == "preprocess" else 1
        )
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
            }
        return {}

    def _make_sub_stage_command(self, sub_stage: str) -> List[str]:
        """Make a command of the specified sub-stage"""

        pile_prep_path = (
            self._launcher_scripts_path
            / "nemo_launcher/collections/dataprep_scripts/pile_dataprep"
        )
        stage_to_code_path = {
            "download": pile_prep_path / "download.py",
            "extract": pile_prep_path / "extract.py",
            "preprocess": pile_prep_path / "preprocess.py",
        }
        choice_model_type, choice_name = self.get_stage_config_choice()

        code_path = stage_to_code_path[sub_stage]
        args = create_args_list(
            hydra=True,
            data_config=choice_name,
            cluster_type=self.cluster,
            launcher_scripts_path=self._launcher_scripts_path,
            data_dir=self._data_dir,
            the_pile_url=self.stage_cfg.get("the_pile_url"),
            file_numbers=self.stage_cfg.get("file_numbers"),
            rm_downloaded=self.stage_cfg.get("rm_downloaded"),
            rm_extracted=self.stage_cfg.get("rm_extracted"),
            tokenizer_type=self.stage_cfg.get("tokenizer_type"),
            tokenizer_library=self.stage_cfg.get("tokenizer_library", "megatron"),
            tokenizer_model=self.stage_cfg.get("tokenizer_model", None),
            vocab_save_dir=self.stage_cfg.get("vocab_save_dir"),
            merges_save_dir=self.stage_cfg.get("merges_save_dir"),
        )
        sub_stage_command = [f"python3 -u {code_path}", *args]
        sub_stage_command = " \\\n  ".join(sub_stage_command)
        return [sub_stage_command]


class SlimPajamaDataPreparation(DataStage):
    """DataStage for preparing the Slim Pajama dataset for gpt"""

    def _make_sub_stages(self) -> List[str]:
        """
        Create a list of sub-stage names which are required to run in current data stage.
        Based on the input config, some of sub stages may not need to run.

        :return: a list of sub-stage names which are required to run
        """
        sub_stages = []
        if self.stage_cfg.get("download_slim_pajama", False):
            sub_stages += ["download"]
        if self.stage_cfg.get("extract_slim_pajama", False):
            sub_stages += ["extract"]
        if self.stage_cfg.get("concat_slim_pajama", False):
            sub_stages += ["concat"]
        if self.stage_cfg.get("preprocess_data", False):
            sub_stages += ["preprocess"]
        return sub_stages

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders and fine-tuning/prompt-learning dataset"""
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)

        data_cfg = self.stage_cfg
        download_vocab_url = data_cfg.get("download_vocab_url")
        download_merges_url = data_cfg.get("download_merges_url")
        vocab_save_dir = data_cfg.get("vocab_save_dir")
        merges_save_dir = data_cfg.get("merges_save_dir")
        download_tokenizer_url = data_cfg.get("download_tokenizer_url")
        tokenizer_save_dir = data_cfg.get("tokenizer_save_dir")

        # Set up preprocessing
        if self.stage_cfg.get("preprocess_data", False):
            # Get preprocess related variables
            run_cfg = data_cfg.get("run")
            nodes = run_cfg.get("node_array_size", 1)
            workers_per_node = run_cfg.get("workers_per_node", 1)
            preprocess_worker_mapping = data_cfg.get("preprocess_worker_mapping")

            files = glob.glob("train_chunk_[0-9]*.jsonl", root_dir=self._data_dir)
            dataset_files = [os.path.join(self._data_dir, file) for file in files]
            # Sort list of files in directory by size
            sorted_files = sorted(dataset_files, key=lambda x: os.stat(x).st_size)
            file_sizes = [os.stat(x).st_size for x in sorted_files]

            avail_workers = nodes * workers_per_node
            distributed_files = [[] for _ in range(avail_workers)]
            distributed_size = [0] * avail_workers
            for f, file_size in zip(sorted_files, file_sizes):
                min_ind = distributed_size.index(min(distributed_size))
                distributed_files[min_ind].append(f)
                distributed_size[min_ind] += file_size

            output = [",".join(distributed_files[i]) for i in range(avail_workers)]
            output = "\n".join(output)
            with open(preprocess_worker_mapping, "w") as file:
                file.write(output)
            print(f" ****** Workers mapping saved to {preprocess_worker_mapping} ...")

        if download_tokenizer_url is not None:
            assert (
                tokenizer_save_dir is not None
            ), "tokenizer_save_dir must be a valid path."
            download_single_file(
                url=download_tokenizer_url,
                save_dir=tokenizer_save_dir,
                file_name="llama_tokenizer.model",
            )

        # Download vocab
        if download_vocab_url is not None:
            assert vocab_save_dir is not None, "vocab_save_dir must be a valid path."
            download_single_file(
                url=download_vocab_url,
                save_dir=vocab_save_dir,
                file_name="vocab.json"
                if download_vocab_url.endswith("json")
                else "vocab.txt",
            )
        # Download merges
        if download_merges_url is not None:
            assert merges_save_dir is not None, "merges_save_dir must be a valid path."
            download_single_file(
                url=download_merges_url,
                save_dir=merges_save_dir,
                file_name="merges.txt",
            )

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
        """
        cfg = self.cfg
        run_cfg = self.stage_cfg.get("run")

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        ntasks_per_node = run_cfg.get("ntasks_per_node")
        array_size = run_cfg.get("node_array_size")
        array = f"0-{array_size-1}"
        parameters = {
            "nodes": 1,
            "ntasks_per_node": ntasks_per_node,
            "array": f"{array}",
            "container_image": container_image,
            "container_mounts": container_mounts,
        }
        # Change parameters for the preprocess job
        if sub_stage == "preprocess":
            parameters["ntasks_per_node"] = run_cfg.get("workers_per_node")
            parameters["cpus_per_task"] = (
                run_cfg.get("cpus_per_node") // parameters["ntasks_per_node"]
            )
        return parameters

    def _make_sub_stage_command(self, sub_stage: str) -> List[str]:
        """Make a command of the specified sub-stage"""

        prep_path = (
            self._launcher_scripts_path
            / "nemo_launcher/collections/dataprep_scripts/slim_pajama_dataprep"
        )
        stage_to_code_path = {
            "download": prep_path / "download.py",
            "extract": prep_path / "extract.py",
            "concat": prep_path / "concat.sh",
            "preprocess": prep_path / "preprocess.py",
        }
        choice_model_type, choice_name = self.get_stage_config_choice()

        code_path = stage_to_code_path[sub_stage]
        # Return special command for the concat stage
        args = []
        if sub_stage == "concat":
            return [
                f'bash {code_path} "{self._data_dir}" {self.stage_cfg.get("rm_extracted")}'
            ]
        elif sub_stage == "preprocess":
            run_cfg = self.stage_cfg.get("run")
            args = create_args_list(
                output_path=self.stage_cfg.get("preprocessed_dir"),
                workers_per_node=run_cfg.get("workers_per_node"),
                worker_mapping_file=self.stage_cfg.get("preprocess_worker_mapping"),
                tokenizer_library=self.stage_cfg.get("tokenizer_library"),
                tokenizer_model=self.stage_cfg.get("tokenizer_model"),
                tokenizer_type=self.stage_cfg.get("tokenizer_type"),
                dataset_impl="mmap",
                log_interval="2000",
                apply_ftfy="store_true",
                workers=run_cfg.get("cpus_per_node") // run_cfg.get("workers_per_node"),
                vocab_file=f'{self.stage_cfg.get("vocab_save_dir")}/vocab.json',
                merges_file=f'{self.stage_cfg.get("merges_save_dir")}/merges.txt',
            )
        elif sub_stage in ["download", "extract"]:
            args = create_args_list(
                hydra=True,
                data_config=choice_name,
                cluster_type=self.cluster,
                launcher_scripts_path=self._launcher_scripts_path,
                data_dir=self._data_dir,
                slim_pajama_url=self.stage_cfg.get("slim_pajama_url"),
                file_numbers=self.stage_cfg.get("file_numbers"),
                rm_downloaded=self.stage_cfg.get("rm_downloaded"),
                rm_extracted=self.stage_cfg.get("rm_extracted"),
                tokenizer_type=self.stage_cfg.get("tokenizer_type"),
                tokenizer_library=self.stage_cfg.get("tokenizer_library", "megatron"),
                tokenizer_model=self.stage_cfg.get("tokenizer_model", None),
                vocab_save_dir=self.stage_cfg.get("vocab_save_dir"),
                merges_save_dir=self.stage_cfg.get("merges_save_dir"),
                approved_sources=self.stage_cfg.get("approved_sources"),
            )
        else:
            print("Unknown Stage, quitting")
            return
        sub_stage_command = [f"python3 -u {code_path}", *args]
        sub_stage_command = " \\\n  ".join(sub_stage_command)
        return [sub_stage_command]


class MC4DataPreparation(DataStage):
    """DataStage for preparing the mC4 dataset for mt5"""

    def _make_sub_stages(self) -> List[str]:
        """
        Create a list of sub-stage names which are required to run in current data stage.
        Based on the input config, some of sub stages may not need to run.

        :return: a list of sub-stage names which are required to run
        :rtype: List[str]
        """
        sub_stages = []
        if self.stage_cfg.get("download_mc4", False):
            sub_stages += ["prepare", "download"]
        if self.stage_cfg.get("preprocess_data", False):
            sub_stages += ["setup_preprocess", "preprocess"]
        return sub_stages

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders and fine-tuning/prompt-learning dataset"""
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)

        data_cfg = self.stage_cfg
        download_vocab_url = data_cfg.get("download_vocab_url")
        download_tokenizer_url = data_cfg.get("download_tokenizer_url")
        vocab_save_dir = data_cfg.get("vocab_save_dir")
        tokenizer_save_dir = data_cfg.get("tokenizer_save_dir")

        if download_vocab_url is not None:
            assert vocab_save_dir is not None, "vocab_save_dir must be a valid path."
            download_single_file(
                url=download_vocab_url, save_dir=vocab_save_dir, file_name="vocab.txt",
            )
        if download_tokenizer_url is not None:
            assert (
                tokenizer_save_dir is not None
            ), "vocab_save_dir must be a valid path."
            download_single_file(
                url=download_tokenizer_url,
                save_dir=tokenizer_save_dir,
                file_name="mt5_tokenizer.model",
            )

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

        node_array_size = (
            run_cfg.get("node_array_size")
            if sub_stage in ["download", "preprocess"]
            else 1
        )
        array = f"0-{node_array_size - 1}"
        if sub_stage == "preprocess":
            ntasks_per_node = run_cfg.get("workers_per_node")
            cpus_per_task = run_cfg.get("cpus_per_node") // ntasks_per_node
        else:
            ntasks_per_node = 1
            cpus_per_task = None

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        if cluster == "bcm":
            return {
                "nodes": 1,
                "array": f"{array}%{node_array_size}",
                "container_image": container_image,
                "container_mounts": container_mounts,
                "ntasks_per_node": ntasks_per_node,
                "cpus_per_task": cpus_per_task,
            }
        if cluster == "bcp":
            return {
                "nodes": node_array_size,
                "ntasks_per_node": ntasks_per_node,
            }
        return {}

    def _make_sub_stage_command(self, sub_stage: str) -> List[str]:
        """Make a command of the specified sub-stage"""
        mc4_prep_path = (
            self._launcher_scripts_path
            / "nemo_launcher/collections/dataprep_scripts/mc4_dataprep"
        )
        stage_to_code_path = {
            "prepare": mc4_prep_path / "prepare.py",
            "download": mc4_prep_path / "download.py",
            "setup_preprocess": mc4_prep_path / "setup_preprocess.py",
            "preprocess": mc4_prep_path / "preprocess.py",
        }

        data_cfg = self.stage_cfg
        run_cfg = data_cfg.get("run")

        code_path = stage_to_code_path[sub_stage]
        if sub_stage == "prepare":
            args = create_args_list(
                data_path=data_cfg.get("mc4_dir"),
                git_lfs_path=data_cfg.get("git_lfs_dir"),
                languages=data_cfg.get("languages"),
                node_array_size=run_cfg.get("node_array_size"),
                worker_mapping_file=data_cfg.get("download_worker_mapping"),
            )
            if data_cfg.get("use_cleaned_english"):
                args += ["--cleaned-en"]
        elif sub_stage == "download":
            args = create_args_list(
                c4_path=Path(data_cfg.get("mc4_dir")) / "c4",
                git_lfs_path=data_cfg.get("git_lfs_dir"),
                worker_mapping_file=data_cfg.get("download_worker_mapping"),
            )
        elif sub_stage == "setup_preprocess":
            args = create_args_list(
                c4_path=Path(data_cfg.get("mc4_dir")) / "c4",
                soft_link_path=data_cfg.get("softlinks_dir"),
                languages=data_cfg.get("languages"),
                node_array_size=run_cfg.get("node_array_size"),
                workers_per_node=run_cfg.get("workers_per_node"),
                max_split_size=200,
                worker_mapping_file=data_cfg.get("preprocess_worker_mapping"),
            )
            if data_cfg.get("use_cleaned_english"):
                args += ["--cleaned-en"]
        else:
            assert sub_stage == "preprocess", f"Unknown substage {sub_stage}"
            args = create_args_list(
                output_path=data_cfg.get("preprocessed_dir"),
                workers_per_node=run_cfg.get("workers_per_node"),
                worker_mapping_file=data_cfg.get("preprocess_worker_mapping"),
                tokenizer_library="sentencepiece",
                tokenizer_model=data_cfg.get("tokenizer_model"),
                dataset_impl="mmap",
                log_interval="2000",
                preproc_folder="store_true",
                apply_ftfy="store_true",
                workers=run_cfg.get("cpus_per_node") // run_cfg.get("workers_per_node"),
            )
            if data_cfg.get("rm_downloaded"):
                args += ["--rm-downloaded"]

        sub_stage_command = [f"python3 -u {code_path}", *args]
        sub_stage_command = " \\\n  ".join(sub_stage_command)
        return [sub_stage_command]


class CustomDataPreparation(DataStage):
    """DataStage for preparing a customized dataset"""

    def _make_sub_stages(self) -> List[str]:
        """
        Create a list of sub-stage names which are required to run in current data stage.
        Based on the input config, some of sub stages may not need to run.

        :return: a list of sub-stage names which are required to run
        :rtype: List[str]
        """
        sub_stages = []
        if self.stage_cfg.get("train_tokenizer", False):
            sub_stages += ["train_tokenizer"]
        if self.stage_cfg.get("preprocess_data", False):
            sub_stages += ["preprocess"]
        return sub_stages

    def _filter_raw_json_files(self, raw_dataset_files: list) -> List:
        """
        Filter the input dataset files to only include json files and derivatives.

        :param list raw_dataset_files: List of the raw dataset files specified in the config
        :return: a list of only the json files in the dataset.
        :rtype: list
        """
        if isinstance(raw_dataset_files, omegaconf.listconfig.ListConfig):
            return raw_dataset_files

        filtered_files = []

        for raw_file in os.listdir(raw_dataset_files):
            # Only select files that end in .jsonl
            if not Path(raw_file).suffix.lower() in [".json", ".jsonl", "json.gz"]:
                continue
            filtered_files.append(os.path.join(raw_dataset_files, raw_file))
        return filtered_files

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders and fine-tuning/prompt-learning dataset"""
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)

        # Setup preprocess data
        data_cfg = self.stage_cfg
        run_cfg = data_cfg.get("run")
        nodes = run_cfg.get("node_array_size", 1)
        workers_per_node = run_cfg.get("workers_per_node", 1)
        raw_dataset_files = data_cfg.get("raw_dataset_files")
        preprocess_worker_mapping = data_cfg.get("preprocess_worker_mapping")

        if data_cfg.get("preprocess_data", False):
            raw_dataset_files = self._filter_raw_json_files(raw_dataset_files)

            # Sort list of files in directory by size
            sorted_files = sorted(raw_dataset_files, key=lambda x: os.stat(x).st_size)
            file_sizes = [os.stat(x).st_size for x in sorted_files]

            avail_workers = nodes * workers_per_node
            distributed_files = [[] for _ in range(avail_workers)]
            distributed_size = [0] * avail_workers
            for f, file_size in zip(sorted_files, file_sizes):
                min_ind = distributed_size.index(min(distributed_size))
                distributed_files[min_ind].append(f)
                distributed_size[min_ind] += file_size

            output = [",".join(distributed_files[i]) for i in range(avail_workers)]
            output = "\n".join(output)
            with open(preprocess_worker_mapping, "w") as file:
                file.write(output)
            print(f" ****** Workers mapping saved to {preprocess_worker_mapping} ...")
            for i in range(avail_workers):
                print(
                    f"{i + 1:>4d} "
                    f"{distributed_size[i]:>7.2f}GB  "
                    f"{','.join([os.path.basename(file) for file in distributed_files[i]]):s}"
                )

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

        if sub_stage == "preprocess":
            node_array_size = run_cfg.get("node_array_size")
            ntasks_per_node = run_cfg.get("workers_per_node")
            cpus_per_task = run_cfg.get("cpus_per_node") // ntasks_per_node
        else:
            node_array_size = 1
            ntasks_per_node = 1
            cpus_per_task = None
        array = f"0-{node_array_size - 1}"

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        if cluster == "bcm":
            return {
                "nodes": 1,
                "array": f"{array}%{node_array_size}",
                "container_image": container_image,
                "container_mounts": container_mounts,
                "ntasks_per_node": ntasks_per_node,
                "cpus_per_task": cpus_per_task,
            }
        if cluster == "bcp":
            return {
                "nodes": node_array_size,
                "ntasks_per_node": ntasks_per_node,
            }
        return {}

    def _make_sub_stage_command(self, sub_stage: str) -> List[str]:
        """Make a command of the specified sub-stage"""
        data_cfg = self.stage_cfg
        run_cfg = data_cfg.get("run")
        cluster_type = self.cfg.cluster_type

        if sub_stage == "train_tokenizer":
            bpe_save_dir = Path(data_cfg.get("bpe_save_dir"))
            bpe_save_dir.mkdir(parents=True, exist_ok=True)
            train_tokenizer_args = data_cfg.get("train_tokenizer_args")
            code_path = f"cd {bpe_save_dir} && spm_train"
            args = create_args_list(**train_tokenizer_args)
        else:
            assert sub_stage == "preprocess", f"Unknown substage {sub_stage}"
            code_path = (
                self._launcher_scripts_path
                / "nemo_launcher/collections/dataprep_scripts/custom_dataprep/preprocess.py"
            )
            args = create_args_list(
                output_path=data_cfg.get("preprocessed_dir"),
                workers_per_node=run_cfg.get("workers_per_node"),
                worker_mapping_file=data_cfg.get("preprocess_worker_mapping"),
                tokenizer_library=data_cfg.get("tokenizer_library"),
                tokenizer_model=data_cfg.get("tokenizer_model"),
                tokenizer_type=data_cfg.get("tokenizer_type"),
                dataset_impl="mmap",
                log_interval="2000",
                apply_ftfy="store_true",
                workers=run_cfg.get("cpus_per_node") // run_cfg.get("workers_per_node"),
            )

            if cluster_type == "bcp":
                args += create_args_list(bcp="store_true")

            if data_cfg.vocab_file and data_cfg.merges_file:
                args += create_args_list(
                    vocab_file=data_cfg.vocab_file, merges_file=data_cfg.merges_file
                )

        sub_stage_command = [f"python3 -u {code_path}", *args]
        sub_stage_command = " \\\n  ".join(sub_stage_command)
        return [sub_stage_command]


class SteerLMDataPreparation(DataStage):
    """DataStage for preparing the Pile dataset for gpt3 and t5"""

    def _make_sub_stages(self) -> List[str]:
        """
        Create a list of sub-stage names which are required to run in current data stage.
        Based on the input config, some of sub stages may not need to run.

        :return: a list of sub-stage names which are required to run
        :rtype: List[str]
        """
        sub_stages = []
        if self.stage_cfg.get("prep_stage") == "1":
            if self.stage_cfg.get("dataset") == "openassistant":
                task_name = "preprocess_openassistant"
            elif self.stage_cfg.get("dataset") == "helpsteer":
                task_name = "preprocess_helpsteer"
            else:
                print(
                    "Currently SteerLM support only openassistand / helpsteer dataset"
                )
        elif self.stage_cfg.get("prep_stage") == "2":
            task_name = "process_to_regression_format"
        else:
            print(
                "Refer to NeMo-Aligner for data preparation support: https://github.com/NVIDIA/NeMo-Aligner/blob/main/docs/user-guide/SteerLM.rst#step-2-download-and-preprocess-data-for-attribute-prediction-modelling"
            )
        if self.stage_cfg.get("preprocess_data", False):
            sub_stages += [task_name]
        return sub_stages

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders and steerlm/openassistant dataset"""
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)

        data_cfg = self.stage_cfg

        output_dir = data_cfg.get("output_dir")

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
        array = f"0-{node_array_size-1}"
        bcp_preproc_npernode = (
            run_cfg.get("bcp_preproc_npernode") if sub_stage == "preprocess" else 1
        )
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
            }
        return {}

    def _make_sub_stage_command(self, sub_stage: str) -> List[str]:
        """Make a command of the specified sub-stage"""

        data_prep_script = self._aligner_code_path / "examples/nlp/data/steerlm/"
        stage_to_code_path = {
            "preprocess_openassistant": data_prep_script
            / "preprocess_openassistant_data.py",
            "preprocess_helpsteer": data_prep_script / "preprocess_helpsteer_data.py",
            "process_to_regression_format": data_prep_script
            / "process_to_regression_format.py",
        }
        choice_model_type, choice_name = self.get_stage_config_choice()
        if choice_name == "steerlm_data_prep2_reg":
            input_file = self.stage_cfg.get("input_dataset")
            output_file = self.stage_cfg.get("output_dir")
            args = create_args_list(
                input_file=input_file, output_file=output_file, replace_underscore=True
            )
        else:
            output_dir = self.stage_cfg.get("output_dir")
            args = create_args_list(
                output_directory=output_dir, replace_underscore=False
            )
        code_path = stage_to_code_path[sub_stage]
        sub_stage_command = [f"python3 -u {code_path}", *args]
        sub_stage_command = " ".join(sub_stage_command)
        return [sub_stage_command]


class MultimodalDataPreparation(DataStage):
    """
    DataStage for preparing a multimodal dataset from huggingface
    Examples include kakaobrain/coyo-700m, laion/laion2B-en-aesthetic and ChristophSchuhmann/improved_aesthetics_5plus
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.sub_stage_names = [
            "download_parquet",
            "download_images",
            "reorganize_tar",
            "precache_encodings",
            "generate_wdinfo",
            "merge_source_tar",
        ]

    def _make_sub_stages(self) -> List[str]:
        """
        Create a list of sub-stage names which are required to run in current data stage.
        Based on the input config, some of sub stages may not need to run.

        :return: a list of sub-stage names which are required to run
        :rtype: List[str]
        """
        sub_stages = []

        for name in self.sub_stage_names:
            if self.stage_cfg.get(name) and self.stage_cfg.get(name).get(
                "enable", False
            ):
                sub_stages.append(name)
        return sub_stages

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders for each substage"""
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)

        data_cfg = self.stage_cfg
        for sub_stage in self.sub_stage_names:
            if (
                data_cfg.get(sub_stage)
                and data_cfg.get(sub_stage).get("enable", False)
                and data_cfg.get(sub_stage).get("output_dir", False)
            ):
                os.makedirs(data_cfg.get(sub_stage).output_dir, exist_ok=True)

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

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        if cluster == "bcm":
            ntasks_per_node = 1
            if sub_stage == "download_images":
                node_array_size = len(
                    glob.glob(
                        os.path.join(
                            stage_cfg.download_parquet.output_dir,
                            "**",
                            stage_cfg.download_parquet.parquet_pattern,
                        ),
                        recursive=True,
                    )
                )
                if node_array_size == 0:
                    node_array_size = stage_cfg.download_images.get(
                        "num_parquets_downloaded"
                    ) * stage_cfg.download_parquet.get("parquet_subpartitions")
            elif sub_stage in ["reorganize_tar", "merge_source_tar"]:
                node_array_size = stage_cfg.get(sub_stage).get("node_array_size", 1)
            elif sub_stage == "precache_encodings":
                node_array_size = stage_cfg.precache_encodings.get("node_array_size", 1)
                ntasks_per_node = 8
            else:  # download_parquet, generate_wdinfo
                node_array_size = 1

            max_simultaneous_jobs = 50
            if isinstance(node_array_size, str) and "-" in node_array_size:
                # for advance usage: resuming an interrupted node array job
                node_array_start, node_array_end = node_array_size.split("-")
                node_array_size = int(node_array_end) - int(node_array_start) + 1
                array = f"{node_array_start}-{node_array_end}%{min(max_simultaneous_jobs, node_array_size)}"
            elif node_array_size > 1:
                array = f"0-{node_array_size - 1}%{min(max_simultaneous_jobs, node_array_size)}"
            else:
                array = None

            return {
                "nodes": 1,
                "array": array,
                "container_image": container_image,
                "container_mounts": container_mounts,
                "ntasks_per_node": ntasks_per_node,
            }
        else:  # will support bcp later
            raise NotImplementedError

    def _make_sub_stage_command(self, sub_stage: str) -> List[str]:
        """Make a command of the specified sub-stage"""

        dataprep_path = (
            self._launcher_scripts_path
            / "nemo_launcher/collections/dataprep_scripts/multimodal_dataprep"
        )
        stage_to_code_path = {
            sub_stage: dataprep_path / f"{sub_stage}.py"
            for sub_stage in self.sub_stage_names
        }
        code_path = stage_to_code_path[sub_stage]
        cfg = self.stage_cfg
        if sub_stage == "download_parquet":
            args = create_args_list(
                hydra=True,
                dataset_repo_id=cfg.get("dataset_repo_id"),
                output_dir=cfg.download_parquet.get("output_dir"),
                parquet_subpartitions=cfg.download_parquet.get(
                    "parquet_subpartitions", 1
                ),
                parquet_pattern=cfg.download_parquet.get("parquet_pattern"),
            )
        elif sub_stage == "download_images":
            args = create_args_list(
                hydra=True,
                input_dir=cfg.download_images.get("input_dir"),
                output_dir=cfg.download_images.get("output_dir"),
                parquet_pattern=cfg.download_images.get("parquet_pattern"),
                download_num_processes=cfg.download_images.get(
                    "download_num_processes"
                ),
                download_num_threads=cfg.download_images.get("download_num_threads"),
                img2dataset_additional_arguments=cfg.download_images.get(
                    "img2dataset_additional_arguments"
                ),
            )
        elif sub_stage == "reorganize_tar":
            args = create_args_list(
                hydra=True,
                input_dir=cfg.reorganize_tar.get("input_dir"),
                output_dir=cfg.reorganize_tar.get("output_dir"),
                file_ext_in_tar=cfg.reorganize_tar.get("file_ext_in_tar"),
                tar_chunk_size=cfg.reorganize_tar.get("tar_chunk_size"),
            )
        elif sub_stage == "precache_encodings":
            args = create_args_list(
                hydra=True,
                input_dir=cfg.precache_encodings.get("input_dir"),
                output_dir=cfg.precache_encodings.get("output_dir"),
                tar_chunk_size=cfg.precache_encodings.get("tar_chunk_size"),
                precache_config_path=cfg.precache_encodings.get("precache_config_path"),
            )
        elif sub_stage == "generate_wdinfo":
            args = create_args_list(
                hydra=True,
                input_dir=cfg.generate_wdinfo.get("input_dir"),
                output_wdinfo_path=cfg.generate_wdinfo.get("output_wdinfo_path"),
                tar_chunk_size=cfg.generate_wdinfo.get("tar_chunk_size"),
                file_ext_in_tar=cfg.generate_wdinfo.get("file_ext_in_tar"),
            )
        elif sub_stage == "merge_source_tar":
            args = create_args_list(
                hydra=True,
                append_tar_dir=cfg.merge_source_tar.get("append_tar_dir"),
                source_dir=cfg.merge_source_tar.get("source_dir"),
                source_extensions=cfg.merge_source_tar.get("source_extensions"),
            )
        else:
            raise ValueError("Invalid sub_stage:", sub_stage)

        sub_stage_command = [f"python3 -u {code_path}", *args]
        sub_stage_command = " \\\n  ".join(sub_stage_command)
        return [sub_stage_command]


class FIDEvaluationDataPreparation(DataStage):
    """
    DataStage for preparing COCO2014 validation set which is used for FID evaluation of multimodal models
    """

    def _make_sub_stages(self) -> List[str]:
        """
        Create a list of sub-stage names which are required to run in current data stage.
        Based on the input config, some of sub stages may not need to run.

        :return: a list of sub-stage names which are required to run
        :rtype: List[str]
        """
        return ["preprocess"]

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders for each substage"""
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)

        dataset_output_root = self.stage_cfg.dataset_output_root
        if os.path.exists(dataset_output_root):
            print(f"WARNING: dataset_output_root already exists")
            response = ""
            while response.lower() not in ["y", "n"]:
                response = input(
                    f"Do you want to wipe everything at {dataset_output_root}? [y/n] \n>>> "
                )
            if response.lower() == "y":
                shutil.rmtree(dataset_output_root)
            else:
                print(
                    "Not removing existing folder. Subsequent processing may produce inaccurate results."
                )
        os.makedirs(dataset_output_root, exist_ok=True)

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
        container_image = self.cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        if cluster == "bcm":
            return {
                "nodes": 1,
                "array": None,
                "container_image": container_image,
                "container_mounts": container_mounts,
                "ntasks_per_node": 1,
            }
        else:  # will support bcp later
            raise NotImplementedError

    def _make_sub_stage_command(self, sub_stage: str) -> List[str]:
        """Make a command of the specified sub-stage"""

        dataprep_path = (
            self._launcher_scripts_path
            / "nemo_launcher/collections/dataprep_scripts/fid_evaluation_dataprep"
        )
        cfg = self.stage_cfg

        if sub_stage == "preprocess":
            code_path = dataprep_path / f"preprocess.py"
            args = create_args_list(
                hydra=True,
                root_dir=cfg.dataset_output_root,
                num_processes=cfg.num_processes,
                preprocess_images=cfg.preprocess_images,
                preprocess_captions=cfg.preprocess_captions,
            )
            sub_stage_command = [f"python3 -u {code_path}", *args]
            sub_stage_command = " \\\n  ".join(sub_stage_command)
        else:
            raise NotImplementedError
        return [sub_stage_command]


class HumanEvalDataPreparation(DataStage):
    """DataStage for preparing a customized dataset"""

    def _make_sub_stages(self) -> List[str]:
        """There is no sub-stages needed for HumanEval dataset"""
        return []

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders for HumanEval dataset"""
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)

        data_cfg = self.stage_cfg
        human_eval_url = data_cfg.get("human_eval_url")
        output_dir = data_cfg.get("output_dir")
        split_string = data_cfg.get("split_string")

        os.makedirs(output_dir, exist_ok=True)

        assert output_dir is not None, "output_dir must be a valid path."
        filename = download_single_file(url=human_eval_url, save_dir=output_dir)

        output_data = self.read_data_into_list(filename)

        random.shuffle(output_data)
        data_splits = [float(split) for split in split_string.split(",")]
        assert (
            abs(sum(data_splits) - 1) < 1e-9
        ), "The values in the split string should sum to one."
        assert (
            len(data_splits) == 3
        ), "Need 3 values, (train,test,validation) in split string"
        num_samples_in_train = int(len(output_data) * data_splits[0])
        num_samples_in_test = int(len(output_data) * data_splits[1])
        num_samples_in_validation = int(len(output_data) * data_splits[2])
        train = output_data[:num_samples_in_train]
        test = output_data[
            num_samples_in_train : num_samples_in_train + num_samples_in_test
        ]
        validation = output_data[-num_samples_in_validation:]

        with open(f"{output_dir}/train.jsonl", "w") as f:
            for entry in train:
                json.dump(entry, f)
                f.write("\n")

        with open(f"{output_dir}/test.jsonl", "w") as f:
            for entry in test:
                json.dump(entry, f)
                f.write("\n")

        with open(f"{output_dir}/validation.jsonl", "w") as f:
            for entry in validation:
                json.dump(entry, f)
                f.write("\n")

    def read_data_into_list(self, filename):
        output_data = []
        with gzip.open(filename, "r") as fin:
            for line in fin:
                input_json_obj = json.loads(line)
                processed_json_obj = {
                    "input": input_json_obj["prompt"],
                    "output": input_json_obj["canonical_solution"],
                }
                output_data.append(processed_json_obj)
        return output_data
