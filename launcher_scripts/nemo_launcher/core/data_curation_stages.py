import os
import copy
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
        stage_cfg_path = NemoMegatronStage.save_stage_hydra_config(
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


class QualityFiltering(DataCurationSubStage):
    """ DataCurationSubStage for performing quality filtering on documents """

    def __init__(self, cfg, memory):
        super().__init__(cfg, memory)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "quality_filtering"
        self.stage_cfg = self._get_sub_stage_confg(self.stage_name)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """ Builds the command groups for the current stage """
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        filter_cfg = Path(self.conf_folder, "heuristic_filter.yaml")
        omegaconf.OmegaConf.save(stage_cfg.get("filter"), filter_cfg)

        command_groups = [[]]

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "output_removed_document_dir": stage_cfg.get("output_removed_document_dir"),
            "output_document_score_dir": stage_cfg.get("output_document_score_dir"),
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
            **optional_args,
        )

        self.memory.data_dir = output_dir

        core_command = ["filter_documents", *args]

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
        filter_cfg = Path(self.get_job_path().results_folder, "fasttext_langid.yaml")
        omegaconf.OmegaConf.save(self.stage_cfg.get("filter_config"), filter_cfg)
        self.memory.filter_config_path = filter_cfg

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
            str(self.get_job_path().results_folder),
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
        """ Builds the command groups for the current stage """
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
            input_data_dir=self.memory.data_dir,
            filter_config_file=self.memory.filter_config_path,
            **optional_args,
        )

        core_command = ["filter_documents", *args]

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
        """ Builds the command groups for the current stage """
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
            log_dir=self.log_folder,
            input_data_dir=self.memory.data_dir,
            output_data_dir=output_dir,
            output_language_distribution=stage_cfg.get("output_language_distribution"),
            **optional_args,
        )
        self.memory.data_dir = None
        self.memory.nested_dir = output_dir

        core_command = ["separate_by_language", *args]

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
        """ Builds the command groups for the current stage """
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]

        output_dir = stage_cfg.get("output_clean_dir")

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            input_data_dir=self.memory.data_dir,
            output_clean_dir=output_dir,
        )

        self.memory.data_dir = output_dir

        core_command = ["text_cleaning", *args]

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
        """ Builds the command groups for the current stage """
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
            log_dir=self.log_folder,
            output_task_ngrams=output_task_ngrams,
            task_config_file=f"{task_cfg}",
        )

        core_command = ["prepare_task_data", *args]

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
        """ Builds the command groups for the current stage """
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "min_ngram_size": stage_cfg.get("min_ngram_size"),
            "max_ngram_size": stage_cfg.get("max_ngram_size"),
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg] for arg in optional_args if optional_args[arg]
        }

        output_dir = stage_cfg.get("output_matched_ngram_data")

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            input_data_dir=self.memory.data_dir,
            input_task_ngrams=self.memory.ngrams_path,
            output_matched_ngram_data=output_dir,
            **optional_args,
        )

        self.memory.ngrams_path = output_dir

        core_command = ["find_matching_ngrams", *args]

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
        """ Builds the command groups for the current stage """
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]

        output_dir = stage_cfg.get("output_task_deduped_dir")

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            input_data_dir=self.memory.data_dir,
            input_matched_ngrams=self.memory.ngrams_path,
            output_task_deduped_dir=output_dir,
        )

        self.memory.data_dir = output_dir

        core_command = ["remove_matching_ngrams", *args]

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
