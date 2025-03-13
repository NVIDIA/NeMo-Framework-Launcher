import os
import copy
import os
import shlex
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

import omegaconf
from nemo_launcher.core.launchers import AutoLauncher
from nemo_launcher.core.stages import (
    NemoMegatronStage,
    clean_command_groups,
    create_args_list,
)


@dataclass
class PipelineMemory:
    """
    PipelineMemory keeps track of all the directories for inputs and
    outputs of the data curator. This allows flexible reordering of
    the data curator substages without needing to manually adjust all
    the input and output directories.
    """

    data_dir: str = None
    nested_dir: str = None
    # Should only be used for when the path needs to be passed between substages
    filter_config_path: str = None
    ngrams_path: str = None


class DataCurationSubStage(NemoMegatronStage):
    """
    DataCurationSubStage is a base class for data curation sub stages.
    It can hold multiple sub-stages. For example, preparing data from
    Common Crawl requires download, extraction, deduplication and filtering.
    They have dependencies on each other and will be launched one by one.
    """

    def __init__(self, cfg, memory):
        super().__init__(cfg)
        self.log_folder = Path()
        self.conf_folder = Path()
        self.memory = memory

    def setup_folder_and_data(self):
        """
        Each job in the data curation pipeline creates a directory
        for writing logs (log_folder), writing and reading intermediate
        results (results_folder) and for reading configs (conf_folder)
        """
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)
        # make the results dir
        results_folder = job_path.results_folder
        results_folder.mkdir(parents=True, exist_ok=True)
        # make the log dir
        self.log_folder = Path(results_folder, "log")
        self.log_folder.mkdir(parents=True, exist_ok=True)
        # Make the conf dir
        self.conf_folder = Path(results_folder, "config")
        self.conf_folder.mkdir(parents=True, exist_ok=True)

    def _make_cluster_parameters(self, cluster: str) -> Dict:
        """
        Make a cluster-specific parameters for jobs on different clusters.
        Current clusters include bcm(slurm), bcp and interactive.
        For example for bcm, it will return slurm parameters:
            {'job_name': 'some_name', 'nodes': 2, 'ntasks_per_node': 8, ...}

        :param str cluster: i.e. `bcm`, `bcp`, `interactive`, etc.
        :param Optional sub_stage: current sub_stage name
        :return: a dictionary of cluster parameters, e.g. `ntasks_per_node`
        :rtype: Dict
        """
        cfg = self.cfg
        stage_cfg = self.stage_cfg

        run_cfg = stage_cfg.get("run")
        job_name = run_cfg.get("name")
        time_limit = run_cfg.get("time_limit")
        nodes = run_cfg.get("nodes")
        dependency = run_cfg.get("dependency")
        # Allow for updating the partition as we might run
        # on CPU only nodes
        node_type = run_cfg.get("node_type")
        node_config = cfg.get("data_curation").get(f"{node_type}_config")

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        setup = None
        env_vars = self.get_env_vars()
        if env_vars:
            setup = [f"export {k}={v}" for k, v in env_vars.items()]

        shared_parameters = {
            "job_name": job_name,
            "time": time_limit,
            "setup": setup,
        }
        if cluster == "bcm":
            cluster_cfg = cfg.get("cluster")
            slurm_cfg = {**copy.deepcopy(cluster_cfg)}
            job_name_prefix = slurm_cfg.pop("job_name_prefix")
            cluster_params = {
                **slurm_cfg,
            }
            cluster_params.update(
                {
                    **shared_parameters,
                    "container_image": container_image,
                    "container_mounts": container_mounts,
                }
            )
            cluster_params["job_name"] = job_name_prefix + cluster_params["job_name"]
            cluster_params["nodes"] = nodes
            cluster_params["dependency"] = dependency
            cluster_params.update(node_config)

        return cluster_params

    def run(self) -> str:
        """
        Run current stage including all of the substages, returns job id on slurm based system otherwise empty string

        :return: job id on slurm based system otherwise empty string
        :rtype: str
        """
        # Create the job folders
        self.setup_folder_and_data()
        job_path = self.get_job_path()

        # Make cluster configuration parameters
        cluster_parameters = self._make_cluster_parameters(self.cluster)
        stage_cfg_path = self.save_stage_hydra_config(
            self.stage_cfg, job_path, self.cfg
        )

        # Build commands to launch on cluster
        command_groups = self.make_stage_command_groups(stage_cfg_path)

        # Create the launcher for the cluster
        launcher = AutoLauncher(
            folder=self.get_job_path().folder,
            cluster=self.cluster,
            **cluster_parameters,
        )

        # Launch the job on the cluster
        job_id = launcher.launch(command_groups)

        return job_id

    def _get_sub_stage_confg(self, sub_stage_name):
        dataset_name = self.cfg.get("data_curation").get("dataset_name")
        sub_stage_config = (
            self.cfg.get("data_curation").get(dataset_name).get(sub_stage_name)
        )
        return sub_stage_config

    def make_dask_command_string(self, runscript_path):
        run_config = self.stage_cfg.get("run")
        dask_config = self.stage_cfg.get("dask", {})

        command_string = []
        # Logging
        command_string.append(f"LOGDIR={self.log_folder}")
        scheduler_file = self.log_folder / "scheduler.json"
        command_string.append(f"SCHEDULER_FILE={scheduler_file}")
        scheduler_log = self.log_folder / "scheduler.log"
        command_string.append(f"SCHEDULER_LOG={scheduler_log}")
        done_marker = self.log_folder / "done.txt"
        command_string.append(f"DONE_MARKER={done_marker}")

        command_string.append(f"RUNSCRIPT={runscript_path}")

        device = run_config.get("node_type")
        command_string.append(f"DEVICE={device}")

        # CPU config
        cpu_worker_memory_limit = dask_config.get("cpu_worker_memory_limit", "0")
        command_string.append(f"CPU_WORKER_MEMORY_LIMIT={cpu_worker_memory_limit}")
        nworkers = dask_config.get("nworkers", "-1")
        command_string.append(f"NUM_WORKERS={nworkers}")

        # GPU config
        scheduler_pool_size = dask_config.get("scheduler_pool_size", "1GB")
        command_string.append(f"RMM_SCHEDULER_POOL_SIZE={scheduler_pool_size}")
        worker_pool_size = dask_config.get("pool_size", "72GiB")
        command_string.append(f"RMM_WORKER_POOL_SIZE={worker_pool_size}")

        # Common
        protocol = dask_config.get("protocol", "tcp")
        command_string.append(f"PROTOCOL={protocol}")
        interface = dask_config.get("interface", "ibp12s0")
        command_string.append(f"INTERFACE={interface}")

        dask_script_path = (
            self._launcher_scripts_path / "nemo_launcher/collections/run_dask_stage.sh"
        )

        return " ".join(command_string) + f" bash {dask_script_path}"


class PipelineException(Exception):
    pass


class InitializeMemory:
    """Dummy stage for initializing the PipelineMemory"""

    def __init__(self, cfg, memory):
        self.cfg = cfg
        self.memory = memory

    def run(self):
        self.memory.data_dir = self.cfg.get("data_dir")


class ChooseLanguage:
    """Dummy stage for choosing a language"""

    def __init__(self, cfg, memory):
        self.cfg = cfg
        self.memory = memory
        self.stage_cfg = (
            self.cfg.get("data_curation").get("special").get("choose_language")
        )
        with omegaconf.open_dict(self.stage_cfg):
            self.stage_cfg.run = {"dependency": "singleton"}

    def run(self):
        lang = self.stage_cfg.get("language")
        base_path = self.memory.nested_dir
        self.memory.data_dir = os.path.join(base_path, lang)
        self.memory.nested_dir = None

        if self.stage_cfg.run["dependency"] != "singleton":
            job_id = self.stage_cfg.run["dependency"].split(":")[1]
            return int(job_id)
        else:
            raise PipelineException(
                "choose_language is only used after separate_by_language"
            )


class QualityFiltering(DataCurationSubStage):
    """DataCurationSubStage for performing quality filtering on documents"""

    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "quality_filtering"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        filter_cfg = Path(self.conf_folder, "heuristic_filter.yaml")
        omegaconf.OmegaConf.save(stage_cfg.get("filter"), filter_cfg)

        command_groups = [[]]

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "output_removed_document_dir": stage_cfg.get("output_removed_document_dir"),
            "output_document_score_dir": stage_cfg.get("output_document_score_dir"),
            "input_json_field": stage_cfg.get("input_json_field"),
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg] for arg in optional_args if optional_args[arg]
        }

        output_dir = stage_cfg.get("output_retained_document_dir")

        # Create the list of arguments for the filter_documents command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            input_data_dir=self.memory.data_dir,
            filter_config_file=f"{filter_cfg}",
            output_retained_document_dir=output_dir,
            scheduler_file=self.log_folder / "scheduler.json",
            **optional_args,
        )

        self.memory.data_dir = output_dir

        runscript = " \\\n  ".join(["filter_documents", *args])
        runscript_path = os.path.join(self.log_folder, f"{self.stage_name}.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class FastTextDownload(NemoMegatronStage):
    """Stage class of downloading the fastText model for language identification"""

    def __init__(self, cfg, memory):
        super().__init__(cfg)
        self.log_folder = Path()
        self.conf_folder = Path()
        self.memory = memory

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "fasttext_download"
        dataset_name = self.cfg.get("data_curation").get("dataset_name")
        self.stage_cfg = (
            self.cfg.get("data_curation").get(dataset_name).get(self.stage_name)
        )

    def _make_cluster_parameters(self, cluster: str,) -> Dict:
        """
        Make a cluster-specific parameters for jobs on different clusters.
        Current clusters include bcm(slurm), bcp and interactive.
        For example for bcm, it will return slurm parameters:
            {'job_name': 'some_name', 'nodes': 2, 'ntasks_per_node': 8, ...}

        :param str cluster: i.e. `bcm`, `bcp`, `interactive`, etc.
        :param Optional sub_stage: current sub_stage name
        :return: a dictionary of cluster parameters, e.g. `ntasks_per_node`
        :rtype: Dict
        """
        cfg = self.cfg
        stage_cfg = self.stage_cfg

        run_cfg = stage_cfg.get("run")
        job_name = run_cfg.get("name")
        time_limit = run_cfg.get("time_limit")
        nodes = run_cfg.get("nodes")
        # Allow for updating the partition as we might run
        # on CPU only nodes
        node_type = run_cfg.get("node_type")
        node_config = cfg.get("data_curation").get(f"{node_type}_config")

        shared_parameters = {
            "job_name": job_name,
            "time": time_limit,
        }
        if cluster == "bcm":
            cluster_cfg = cfg.get("cluster")
            slurm_cfg = {**copy.deepcopy(cluster_cfg)}
            job_name_prefix = slurm_cfg.pop("job_name_prefix")
            cluster_params = {
                **slurm_cfg,
            }
            cluster_params.update(
                {**shared_parameters,}
            )
            cluster_params["job_name"] = job_name_prefix + cluster_params["job_name"]
            cluster_params["nodes"] = nodes
            cluster_params.update(node_config)

        return cluster_params

    def run(self) -> str:
        # Create the log and res dir
        self.setup_folder_and_data()

        cluster_parameters = self._make_cluster_parameters(self.cluster)

        # Write out the filter configuration as a separate config file
        results_path = self.get_job_path().results_folder
        filter_cfg = self.stage_cfg["filter_config"]
        bin_path = Path(results_path, filter_cfg["filters"][0]["params"]["model_path"])
        filter_cfg_file = Path(results_path, "fasttext_langid.yaml")
        filter_cfg["filters"][0]["params"]["model_path"] = str(bin_path)

        omegaconf.OmegaConf.save(filter_cfg, filter_cfg_file)
        self.memory.filter_config_path = filter_cfg_file

        # Get the path to download script
        # preface it with bash
        start_download_script = str(
            Path().joinpath(
                self.cfg["launcher_scripts_path"],
                "nemo_launcher",
                "collections",
                "datacuration_scripts",
                "download_fasttext.sh",
            )
        )
        cmd = [
            "bash",
            f"{start_download_script}",
            str(bin_path),
        ]
        cluster_parameters["setup"] = [shlex.join(cmd)]

        # Create launcher
        launcher = AutoLauncher(
            folder=self.get_job_path().folder,
            cluster=self.cluster,
            **cluster_parameters,
        )
        job_id = launcher.launch(command_groups=[])

        return job_id


class LanguageIdentification(DataCurationSubStage):
    """DataCurationSubStage for performing language identification on documents"""

    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "language_identification"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        command_groups = [[]]

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "output_removed_document_dir": stage_cfg.get("output_removed_document_dir"),
            "output_document_score_dir": stage_cfg.get("output_document_score_dir"),
            "output_retained_document_dir": stage_cfg.get(
                "output_retained_document_dir"
            ),
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg] for arg in optional_args if optional_args[arg]
        }

        # Create the list of arguments for the filter_documents command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            log_scores=stage_cfg.get("log_scores"),
            input_data_dir=self.memory.data_dir,
            filter_config_file=self.memory.filter_config_path,
            scheduler_file=self.log_folder / "scheduler.json",
            **optional_args,
        )

        if optional_args["output_retained_document_dir"] is not None:
            self.memory.data_dir = optional_args["output_retained_document_dir"]

        runscript = " \\\n  ".join(["filter_documents", *args])
        runscript_path = os.path.join(self.log_folder, f"{self.stage_name}.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class SeparateByLanguage(DataCurationSubStage):
    """DataCurationSubStage for separating documents by language"""

    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "separate_by_language"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "remove_language_field": stage_cfg.get("remove_language_field")
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg] for arg in optional_args if optional_args[arg]
        }
        output_dir = stage_cfg.get("output_data_dir")

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            input_data_dir=self.memory.data_dir,
            output_data_dir=output_dir,
            output_metadata_distribution=stage_cfg.get("output_language_distribution"),
            scheduler_file=self.log_folder / "scheduler.json",
            **optional_args,
        )
        self.memory.data_dir = None
        self.memory.nested_dir = output_dir

        runscript = " \\\n  ".join(["separate_by_metadata", *args])
        runscript_path = os.path.join(self.log_folder, f"{self.stage_name}.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class TextCleaning(DataCurationSubStage):
    """DataCurationSubStage for cleaning documents"""

    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "text_cleaning"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]

        output_dir = stage_cfg.get("output_clean_dir")

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            input_data_dir=self.memory.data_dir,
            output_clean_dir=output_dir,
            scheduler_file=self.log_folder / "scheduler.json",
        )

        self.memory.data_dir = output_dir

        runscript = " \\\n  ".join(["text_cleaning", *args])
        runscript_path = os.path.join(self.log_folder, f"{self.stage_name}.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class PrepareTaskData(DataCurationSubStage):
    """DataCurationSubStage for preparing the task specific ngrams"""

    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "prepare_task_data"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        command_groups = [[]]
        output_task_ngrams = stage_cfg.get("output_task_ngrams")
        self.memory.ngrams_path = output_task_ngrams

        # Use the cache if configured to and it exists
        if stage_cfg.get("use_ngram_cache") and Path(output_task_ngrams).is_file():
            return command_groups

        # Write out the task configuration as a separate config file
        task_cfg = Path(self.conf_folder, "lm_tasks.yaml")
        omegaconf.OmegaConf.save(stage_cfg.get("lm_tasks_config"), task_cfg)

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            output_task_ngrams=output_task_ngrams,
            task_config_file=f"{task_cfg}",
            scheduler_file=self.log_folder / "scheduler.json",
        )

        runscript = " \\\n  ".join(["prepare_task_data", *args])
        runscript_path = os.path.join(self.log_folder, f"{self.stage_name}.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class FindMatchingNgrams(DataCurationSubStage):
    """DataCurationSubStage for finding task ngrams in the dataset"""

    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "find_matching_ngrams"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "min_ngram_size": stage_cfg.get("min_ngram_size"),
            "max_ngram_size": stage_cfg.get("max_ngram_size"),
            "input_json_text_field": stage_cfg.get("input_json_text_field"),
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg] for arg in optional_args if optional_args[arg]
        }

        output_dir = stage_cfg.get("output_matched_ngram_data")

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            input_data_dir=self.memory.data_dir,
            input_task_ngrams=self.memory.ngrams_path,
            output_matched_ngram_data=output_dir,
            scheduler_file=self.log_folder / "scheduler.json",
            **optional_args,
        )

        self.memory.ngrams_path = output_dir

        runscript = " \\\n  ".join(["find_matching_ngrams", *args])
        runscript_path = os.path.join(self.log_folder, f"{self.stage_name}.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class RemoveMatchingNgrams(DataCurationSubStage):
    """DataCurationSubStage for removing dataset text matching task ngrams"""

    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "remove_matching_ngrams"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "input_json_text_field": stage_cfg.get("input_json_text_field"),
            "match_threshold": stage_cfg.get("match_threshold"),
            "max_document_splits": stage_cfg.get("max_document_splits"),
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg] for arg in optional_args if arg in stage_cfg
        }

        output_dir = stage_cfg.get("output_task_deduped_dir")

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            input_data_dir=self.memory.data_dir,
            input_matched_ngrams=self.memory.ngrams_path,
            output_task_deduped_dir=output_dir,
            scheduler_file=self.log_folder / "scheduler.json",
            **optional_args,
        )

        self.memory.data_dir = output_dir

        runscript = " \\\n  ".join(["remove_matching_ngrams", *args])
        runscript_path = os.path.join(self.log_folder, f"{self.stage_name}.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class DataCurationStage(NemoMegatronStage):
    """Stage class for running all steps of the data curator"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.log_folder = Path()
        self.conf_folder = Path()
        self.STR2SUBSTAGECLASS = {
            "fasttext_download": FastTextDownload,
            "language_identification": LanguageIdentification,
            "separate_by_language": SeparateByLanguage,
            "text_cleaning": TextCleaning,
            "prepare_task_data": PrepareTaskData,
            "find_matching_ngrams": FindMatchingNgrams,
            "remove_matching_ngrams": RemoveMatchingNgrams,
            "choose_language": ChooseLanguage,
            "quality_filtering": QualityFiltering,
            "compute_minhashes": ComputeMinhashes,
            "minhash_buckets": MinhashBuckets,
            "jaccard_map_buckets": JaccardMapBuckets,
            "jaccard_shuffle": JaccardShuffle,
            "jaccard_compute": JaccardCompute,
            "connected_component": ConnectedComponent,
            "write_deduped_result_with_text": WriteDedupedResultWithText,
            "verify_all_pairs_jaccard": VerifyAllPairsJaccard,
            "add_id": AddId,
        }

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "data_curation"
        self.stage_cfg = cfg.get("data_curation")

    def setup_sub_stages(self) -> List:
        """Process and validate the substages in order"""
        dataset_name = self.stage_cfg.get("dataset_name")
        stages = self.stage_cfg.get("stages")
        sub_stage_classes = [InitializeMemory]
        for stage_name in stages:
            stage = self.stage_cfg.get(stage_name)
            assert isinstance(
                stage, omegaconf.listconfig.ListConfig
            ), f"{stage_name} not defined in data curator config."
            for sub_stage_name in stage:
                is_valid_substage = sub_stage_name in self.STR2SUBSTAGECLASS
                assert is_valid_substage, f"{sub_stage_name} not recognized"
                has_stage_config = sub_stage_name in self.stage_cfg.get(
                    dataset_name
                ) or sub_stage_name in self.stage_cfg.get("special")
                assert has_stage_config, f"Config for {sub_stage_name} not found"

                sub_stage = self.STR2SUBSTAGECLASS[sub_stage_name]
                sub_stage_classes.append(sub_stage)

        return sub_stage_classes

    def run(self) -> str:
        """
        Run current stage including all of the substages,
        returns job id on slurm based system otherwise empty string

        :return: job id on slurm based system otherwise empty string
        :rtype: str
        """
        # Create the job folders
        self.setup_folder_and_data()

        substages = self.setup_sub_stages()
        memory = PipelineMemory()

        job_id = ""
        for sub_stage_class in substages:
            # Create the sub-stage
            sub_stage = sub_stage_class(self.cfg, memory)
            if job_id:
                dependency = f"aftercorr:{job_id}"
                sub_stage.stage_cfg["run"]["dependency"] = dependency
            # Launch the sub-stage
            job_id = sub_stage.run()

        assert memory.data_dir, "Data dir cannot be None"
        self.cfg["data_dir"] = memory.data_dir

        return job_id


class ComputeMinhashes(DataCurationSubStage):
    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "compute_minhashes"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        command_groups = [[]]

        # Create the list of arguments for the filter_documents command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            input_data_dirs=self.cfg.get("data_dir"),
            minhash_length=stage_cfg.get("minhash_length"),
            char_ngram=stage_cfg.get("char_ngram"),
            hash_bytes=stage_cfg.get("hash_bytes"),
            seed=stage_cfg.get("seed"),
            output_minhash_dir=stage_cfg.get("output_fuzzy_deduped_dir"),
            num_files=stage_cfg.get("num_files"),
            files_per_partition=stage_cfg.get("files_per_partition"),
            scheduler_file=self.log_folder / "scheduler.json",
        )

        runscript = " \\\n  ".join(["gpu_compute_minhashes", *args])
        runscript_path = os.path.join(self.log_folder, "compute_minhashes.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class MinhashBuckets(DataCurationSubStage):
    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "minhash_buckets"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        command_groups = [[]]

        # Create the list of arguments for the filter_documents command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            input_data_dirs=stage_cfg.get("input_minhash_dir"),
            minhash_length=stage_cfg.get("minhash_length"),
            output_bucket_dir=stage_cfg.get("output_fuzzy_deduped_dir"),
            num_bands=stage_cfg.get("num_bands"),
            buckets_per_shuffle=stage_cfg.get("buckets_per_shuffle"),
            protocol=stage_cfg.get("dask").get("protocol"),
            scheduler_file=self.log_folder / "scheduler.json",
        )

        runscript = " \\\n  ".join(["minhash_buckets", *args])
        runscript_path = os.path.join(self.log_folder, "minhash_buckets.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class JaccardMapBuckets(DataCurationSubStage):
    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "jaccard_map_buckets"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        command_groups = [[]]

        # Create the list of arguments for the filter_documents command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            input_data_dirs=self.cfg.get("data_dir"),
            input_bucket_dir=stage_cfg.get("input_bucket_dir"),
            text_ddf_blocksize=stage_cfg.get("text_ddf_blocksize"),
            output_dir=stage_cfg.get("output_fuzzy_deduped_dir"),
            scheduler_file=self.log_folder / "scheduler.json",
        )

        runscript = " \\\n  ".join(["jaccard_map_buckets", *args])
        runscript_path = os.path.join(self.log_folder, "jaccard_map_buckets.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class JaccardShuffle(DataCurationSubStage):
    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "jaccard_shuffle"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        command_groups = [[]]

        # Create the list of arguments for the filter_documents command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            input_data_dirs=self.cfg.get("data_dir"),
            input_bucket_mapping_dir=stage_cfg.get("input_bucket_mapping_dir"),
            text_ddf_blocksize=stage_cfg.get("text_ddf_blocksize"),
            output_dir=stage_cfg.get("output_fuzzy_deduped_dir"),
            parts_per_worker=stage_cfg.get("parts_per_worker"),
            scheduler_file=self.log_folder / "scheduler.json",
        )

        runscript = " \\\n  ".join(["jaccard_shuffle", *args])
        runscript_path = os.path.join(self.log_folder, "jaccard_shuffle.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class JaccardCompute(DataCurationSubStage):
    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "jaccard_compute"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        command_groups = [[]]

        # Create the list of arguments for the filter_documents command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            shuffled_docs_path=stage_cfg.get("shuffled_docs_path"),
            output_dir=stage_cfg.get("output_fuzzy_deduped_dir"),
            num_files=stage_cfg.get("num_files"),
            files_per_partition=stage_cfg.get("files_per_partition"),
            scheduler_file=self.log_folder / "scheduler.json",
        )

        runscript = " \\\n  ".join(["jaccard_compute", *args])
        runscript_path = os.path.join(self.log_folder, "jaccard_compute.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class ConnectedComponent(DataCurationSubStage):
    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "connected_component"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        command_groups = [[]]

        # Create the list of arguments for the filter_documents command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            jaccard_pairs_path=stage_cfg.get("jaccard_pairs_path"),
            output_dir=stage_cfg.get("output_dir"),
            cache_dir=stage_cfg.get("cache_dir"),
            scheduler_file=self.log_folder / "scheduler.json",
        )

        runscript = " \\\n  ".join(["gpu_connected_component", *args])
        runscript_path = os.path.join(self.log_folder, "connected_component.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class WriteDedupedResultWithText(DataCurationSubStage):
    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "write_deduped_result_with_text"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        command_groups = [[]]

        # Create the list of arguments for the filter_documents command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            original_path=self.cfg.get("data_dir"),
            output_dir=stage_cfg.get("output_dir"),
        )

        runscript = " \\\n  ".join(["write_deduped_result_with_text", *args])
        runscript_path = os.path.join(
            self.log_folder, "write_deduped_result_with_text.sh"
        )

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class VerifyAllPairsJaccard(DataCurationSubStage):
    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "verify_all_pairs_jaccard"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        command_groups = [[]]

        # Create the list of arguments for the filter_documents command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            output_dir=stage_cfg.get("output_dir"),
            cache_dir=stage_cfg.get("cache_dir"),
            scheduler_file=self.log_folder / "scheduler.json",
        )

        runscript = " \\\n  ".join(["verify_all_pairs_jaccard", *args])
        runscript_path = os.path.join(self.log_folder, "verify_all_pairs_jaccard.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class AddId(DataCurationSubStage):
    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "add_id"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """Builds the command groups for the current stage"""
        stage_cfg = self.stage_cfg

        command_groups = [[]]

        # Create the list of arguments for the filter_documents command
        args = create_args_list(
            replace_underscore=True,
            input_data_dir=self.memory.data_dir,
            output_data_dir=stage_cfg.get("output_data_dir"),
            id_field_name=stage_cfg.get("id_field_name"),
            id_prefix=stage_cfg.get("id_prefix"),
            input_file_type=stage_cfg.get("input_file_type"),
            output_file_type=stage_cfg.get("output_file_type"),
            scheduler_file=self.log_folder / "scheduler.json",
        )

        self.memory.data_dir = stage_cfg.get("output_data_dir")

        runscript = " \\\n  ".join(["add_id", *args])
        runscript_path = os.path.join(self.log_folder, "add_id.sh")

        with open(runscript_path, "w") as f:
            f.write(runscript)

        core_command = [self.make_dask_command_string(runscript_path)]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups
