import copy
import shlex
import subprocess
import omegaconf
from typing import Dict, List
from pathlib import Path

from nemo_launcher.core.stages import (
    NemoMegatronStage,
    create_args_list,
    clean_command_groups,
)
from nemo_launcher.core.launchers import AutoLauncher


class DataCurationStage(NemoMegatronStage):
    """
    DataCurationStage is a base class for data curation stages.
    It can hold multiple sub-stages. For example, preparing data from
    Common Crawl requires download, extraction, deduplication and filtering.
    They have dependencies on each other and will be launched one by one.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.log_folder = Path()
        self.conf_folder = Path()

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
        self.log_folder = Path(job_path.folder, 'log')
        self.log_folder.mkdir(parents=True, exist_ok=True)
        # Make the conf dir
        self.conf_folder = Path(job_path.folder, 'config')
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
        nodes = run_cfg.get('nodes')
        # Allow for updating the partition as we might run
        # on CPU only nodes
        partition = run_cfg.get('partition')
        dependency = run_cfg.get('dependency')

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
            cluster_params.update({
                **shared_parameters,
                "container_image": container_image,
                "container_mounts": container_mounts,
            })
            cluster_params[
                "job_name"] = job_name_prefix + cluster_params["job_name"]
            cluster_params['nodes'] = nodes
            cluster_params['partition'] = partition
            cluster_params['dependency'] = dependency

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
            self.stage_cfg,
            job_path,
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


class QualityFiltering(DataCurationStage):
    """ DataCurationStage for performing quality filtering on documents """

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "quality_filtering"
        self.stage_cfg = cfg.get("quality_filtering")

    def make_stage_command_groups(self,
                                  stage_cfg_path: Path) -> List[List[str]]:
        """ Builds the command groups for the current stage """
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        filter_cfg = Path(self.conf_folder, "heuristic_filter.yaml")
        omegaconf.OmegaConf.save(stage_cfg.get('filter'), filter_cfg)

        command_groups = [[]]

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "output_removed_document_dir":
                stage_cfg.get('output_removed_document_dir'),
            "output_document_score_dir":
                stage_cfg.get('output_document_score_dir'),
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg]
            for arg in optional_args
            if optional_args[arg]
        }

        # Create the list of arguments for the filter_documents command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            input_data_dir=stage_cfg.get("input_dir"),
            filter_config_file=f"{filter_cfg}",
            output_retained_document_dir=stage_cfg.get(
                "output_retained_document_dir"),
            **optional_args,
        )

        core_command = ["filter_documents", *args]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class GetWikipediaUrls(DataCurationStage):

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "get_wikipedia_urls"
        self.stage_cfg = cfg['wikipedia'].get("get_wikipedia_urls")

    def make_stage_command_groups(self,
                                  staage_cfg_path: Path) -> List[List[str]]:
        """ Builds the command groups for the current stage """
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "language": stage_cfg.get('language'),
            "wikidumps_index_base_url": stage_cfg.get('wikidump_index_baseurl'),
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg]
            for arg in optional_args
            if optional_args[arg]
        }

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            output_url_file=stage_cfg.get("output_url_file"),
            **optional_args,
        )

        core_command = ["get_wikipedia_urls", *args]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class DownloadAndExtractWikipedia(DataCurationStage):

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_stage_vars(
        self,
        cfg,
    ):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "download_and_extract"
        self.stage_cfg = cfg['wikipedia'].get("download_and_extract")

    def make_stage_command_groups(self,
                                  stage_cfg_path: Path) -> List[List[str]]:
        """ Builds the command groups for the current stage """
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]
        # Write out the filter configuration as a separate config file

        builder_cfg = Path(self.conf_folder, "dataset_builder.yaml")
        omegaconf.OmegaConf.save(stage_cfg.get('builder_config'), builder_cfg)

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "input_data_dir": stage_cfg.get('input_data_dir'),
            "download_only": stage_cfg.get('download_only'),
            "extract_only": stage_cfg.get('extract_only'),
            "keep_downloaded_files": stage_cfg.get('keep_downloaded_files'),
            "output_download_dir": stage_cfg.get('output_download_dir'),
            "overwrite_existing_json": stage_cfg.get('overwrite_existing_json')
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg]
            for arg in optional_args
            if optional_args[arg]
        }

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            input_url_file=stage_cfg.get("input_url_file"),
            output_json_dir=stage_cfg.get("output_json_dir"),
            builder_config_file=f"{builder_cfg}",
            max_queue_size=stage_cfg.get("max_queue_size"),
            download_processes_per_node=stage_cfg.get(
                "download_processes_per_node"),
            extract_processes_per_node=stage_cfg.get(
                "extract_processes_per_node"),
            **optional_args,
        )

        core_command = ["download_and_extract", *args]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class Wikipedia(NemoMegatronStage):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.log_folder = Path()
        self.conf_folder = Path()
        self.STR2SUBSTAGECLASS = {
            'get_wikipedia_urls': GetWikipediaUrls,
            'download_and_extract': DownloadAndExtractWikipedia,
        }

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "wikipedia"
        self.stage_cfg = cfg.get("wikipedia")

    def run(self) -> str:
        """
        Run current stage including all of the substages,
        returns job id on slurm based system otherwise empty string

        :return: job id on slurm based system otherwise empty string
        :rtype: str
        """
        # Create the job folders
        self.setup_folder_and_data()

        job_id = ""
        for sub_stage_name in self.stage_cfg.keys():
            if sub_stage_name != 'run':
                sub_stage_class = self.STR2SUBSTAGECLASS[sub_stage_name]
                # Create the sub-stage
                sub_stage = sub_stage_class(self.cfg)
                if job_id:
                    dependency = f"aftercorr:{job_id}"
                    sub_stage.stage_cfg["run"]["dependency"] = dependency
                # Launch the sub-stage
                job_id = sub_stage.run()

        return job_id


class GetCommonCrawlUrls(DataCurationStage):

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "get_common_crawl_urls"
        self.stage_cfg = cfg['common_crawl'].get("get_common_crawl_urls")

    def make_stage_command_groups(self,
                                  stage_cfg_path: Path) -> List[List[str]]:
        """ Builds the command groups for the current stage """
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "cc_news":
                stage_cfg.get('cc_news'),
            "cc_snapshot_index_file":
                Path().joinpath(
                    self.get_job_path().results_folder,
                    "collinfo.json",
                ),
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg]
            for arg in optional_args
            if optional_args[arg]
        }

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            log_file=Path().joinpath(self.log_folder, "get_cc_urls.log"),
            cc_index_prefix=stage_cfg.get("cc_index_prefix"),
            cc_data_domain_prefix=stage_cfg.get("cc_data_domain_prefix"),
            output_warc_url_file=stage_cfg.get("output_warc_url_file"),
            starting_snapshot=stage_cfg.get("starting_snapshot"),
            ending_snapshot=stage_cfg.get("ending_snapshot"),
            **optional_args,
        )

        core_command = ["get_common_crawl_urls", *args]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class DownloadAndExtractCommonCrawl(DataCurationStage):

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_stage_vars(
        self,
        cfg,
    ):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "download_and_extract"
        self.stage_cfg = cfg['common_crawl'].get("download_and_extract")

    def make_stage_command_groups(self,
                                  stage_cfg_path: Path) -> List[List[str]]:
        """ Builds the command groups for the current stage """
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]
        # Write out the filter configuration as a separate config file

        builder_cfg = Path(self.conf_folder, "dataset_builder.yaml")
        omegaconf.OmegaConf.save(stage_cfg.get('builder_config'), builder_cfg)

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "input_data_dir": stage_cfg.get('input_data_dir'),
            "download_only": stage_cfg.get('download_only'),
            "extract_only": stage_cfg.get('extract_only'),
            "keep_downloaded_files": stage_cfg.get('keep_downloaded_files'),
            "output_download_dir": stage_cfg.get('output_download_dir'),
            "overwrite_existing_json": stage_cfg.get('overwrite_existing_json')
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg]
            for arg in optional_args
            if optional_args[arg]
        }

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            input_url_file=stage_cfg.get("input_url_file"),
            output_json_dir=stage_cfg.get("output_json_dir"),
            builder_config_file=f"{builder_cfg}",
            max_queue_size=stage_cfg.get("max_queue_size"),
            download_processes_per_node=stage_cfg.get(
                "download_processes_per_node"),
            extract_processes_per_node=stage_cfg.get(
                "extract_processes_per_node"),
            **optional_args,
        )

        core_command = ["download_and_extract", *args]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class CommonCrawl(NemoMegatronStage):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.log_folder = Path()
        self.conf_folder = Path()
        self.STR2SUBSTAGECLASS = {
            'get_common_crawl_urls': GetCommonCrawlUrls,
            'download_and_extract': DownloadAndExtractCommonCrawl,
        }

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "common_crawl"
        self.stage_cfg = cfg.get("common_crawl")

    def run(self) -> str:
        """
        Run current stage including all of the substages,
        returns job id on slurm based system otherwise empty string

        :return: job id on slurm based system otherwise empty string
        :rtype: str
        """
        # Create the job folders
        self.setup_folder_and_data()

        job_id = ""
        for sub_stage_name in self.stage_cfg.keys():
            if sub_stage_name != 'run':
                sub_stage_class = self.STR2SUBSTAGECLASS[sub_stage_name]
                # Create the sub-stage
                sub_stage = sub_stage_class(self.cfg)
                if job_id:
                    dependency = f"aftercorr:{job_id}"
                    sub_stage.stage_cfg["run"]["dependency"] = dependency
                # Launch the sub-stage
                job_id = sub_stage.run()

        return job_id


class AddId(DataCurationStage):

    def __init__(self, cfg, super_stage_name=None):
        self.super_stage_name = super_stage_name
        super().__init__(cfg)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "add_id"
        if self.super_stage_name is not None:
            self.stage_cfg = cfg[self.super_stage_name].get("add_document_ids")
        else:
            self.stage_cfg = cfg.get("add_document_ids")

    def make_stage_command_groups(self,
                                  stage_cfg_path: Path) -> List[List[str]]:
        """ Builds the command groups for the current stage """
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            log_dir=self.log_folder,
            input_data_dir=stage_cfg.get('input_data_dir'),
        )

        core_command = ["add_id", *args]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class StartRedisCluster(NemoMegatronStage):

    def __init__(self, cfg, super_stage_name=None):
        self.super_stage_name = super_stage_name
        super().__init__(cfg)
        self.log_folder = Path()
        self.conf_folder = Path()

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "start_redis_cluster"
        if self.super_stage_name is not None:
            self.stage_cfg = cfg[self.super_stage_name].get(
                "start_redis_cluster")
        else:
            self.stage_cfg = cfg.get("start_redis_cluster")

    def _make_cluster_parameters(
        self,
        cluster: str,
    ) -> Dict:
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
        nodes = run_cfg.get('nodes')
        # Allow for updating the partition as we might run
        # on CPU only nodes
        partition = run_cfg.get('partition')

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
            cluster_params.update({
                **shared_parameters,
            })
            cluster_params[
                "job_name"] = job_name_prefix + cluster_params["job_name"]
            cluster_params['nodes'] = nodes
            cluster_params['partition'] = partition

        return cluster_params

    def run(self) -> str:
        # Create the log and res dir
        self.setup_folder_and_data()

        cluster_parameters = self._make_cluster_parameters(self.cluster)

        # Get the path to redis cluster script
        # preface it with bash
        start_redis_cluster_script = str(Path().joinpath(
            self.cfg['launcher_scripts_path'],
            'nemo_launcher',
            'collections'
            'datacuration_scripts/start_redis_cluster.sh',
        ))
        cmd = [
            'bash',
            f'{start_redis_cluster_script}',
            '$SLURM_JOB_NUM_NODES',
            str(self.get_job_path().results_folder),
        ]
        cluster_parameters['setup'] = [shlex.join(cmd)]

        # Create launcher
        launcher = AutoLauncher(
            folder=self.get_job_path().folder,
            cluster=self.cluster,
            **cluster_parameters,
        )
        job_id = launcher.launch(command_groups=[])

        return job_id


class HashDocuments(DataCurationStage):

    def __init__(self, cfg, super_stage_name=None):
        self.super_stage_name = super_stage_name
        super().__init__(cfg)

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "hash_documents"
        if self.super_stage_name is not None:
            self.stage_cfg = cfg[self.super_stage_name].get(
                "compute_document_hashes")
        else:
            self.stage_cfg = cfg.get("compute_document_hashes")

    def make_stage_command_groups(self,
                                  stage_cfg_path: Path) -> List[List[str]]:
        """ Builds the command groups for the current stage """
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "cc_news":
                stage_cfg.get('cc_news'),
            "cc_snapshot_index_file":
                Path().joinpath(
                    self.get_job_path().results_folder,
                    "collinfo.json",
                ),
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg]
            for arg in optional_args
            if optional_args[arg]
        }

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            log_file=Path().joinpath(self.log_folder, "get_cc_urls.log"),
            cc_index_prefix=stage_cfg.get("cc_index_prefix"),
            cc_data_domain_prefix=stage_cfg.get("cc_data_domain_prefix"),
            output_warc_url_file=stage_cfg.get("output_warc_url_file"),
            starting_snapshot=stage_cfg.get("starting_snapshot"),
            ending_snapshot=stage_cfg.get("ending_snapshot"),
            **optional_args,
        )

        core_command = ["get_common_crawl_urls", *args]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class ShutdownRedisCluster(NemoMegatronStage):

    def __init__(self, cfg, super_stage_name=None):
        self.super_stage_name = super_stage_name
        super().__init__(cfg)
        self.log_folder = Path()
        self.conf_folder = Path()

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "shutdown_redis_cluster"
        if self.super_stage_name is not None:
            self.stage_cfg = cfg[self.super_stage_name].get(
                "shutdown_redis_cluster")
        else:
            self.stage_cfg = cfg.get("shutdown_redis_cluster")

    def _make_cluster_parameters(
        self,
        cluster: str,
    ) -> Dict:
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
        nodes = run_cfg.get('nodes')
        # Allow for updating the partition as we might run
        # on CPU only nodes
        partition = run_cfg.get('partition')

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
            cluster_params.update({
                **shared_parameters,
            })
            cluster_params[
                "job_name"] = job_name_prefix + cluster_params["job_name"]
            cluster_params['nodes'] = nodes
            cluster_params['partition'] = partition

        return cluster_params

    # Call scancel from the command line. No need to go to SLURM/BCP
    def run(self) -> str:
        # Create the log and res dir
        self.setup_folder_and_data()

        shutdown_redis_cluster_script = str(
            Path().joinpath(self.cfg['launcher_scripts_path'], 'nemo_launcher',
                            'collections',
                            'datacuration_scripts/shutdown_redis_cluster.sh'),)
        if self.cfg['debug']:
            print(f'scancel {shutdown_redis_cluster_script}')
        else:
            subprocess.run(['scancel', shutdown_redis_cluster_script])


class GroupDuplicates(DataCurationStage):

    def __init__(self, cfg, super_stage_name=None):
        self.super_stage_name = super_stage_name
        super().__init__(cfg)

    def setup_stage_vars(self, cfg, super_stage_name=None):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "form_duplicate_groups"
        if self.super_stage_name is not None:
            self.stage_cfg = cfg[self.super_stage_name].get(
                "form_duplicate_groups")
        else:
            self.stage_cfg = cfg.get("form_duplicate_groups")

    def make_stage_command_groups(self,
                                  stage_cfg_path: Path) -> List[List[str]]:
        """ Builds the command groups for the current stage """
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "cc_news":
                stage_cfg.get('cc_news'),
            "cc_snapshot_index_file":
                Path().joinpath(
                    self.get_job_path().results_folder,
                    "collinfo.json",
                ),
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg]
            for arg in optional_args
            if optional_args[arg]
        }

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            log_file=Path().joinpath(self.log_folder, "get_cc_urls.log"),
            cc_index_prefix=stage_cfg.get("cc_index_prefix"),
            cc_data_domain_prefix=stage_cfg.get("cc_data_domain_prefix"),
            output_warc_url_file=stage_cfg.get("output_warc_url_file"),
            starting_snapshot=stage_cfg.get("starting_snapshot"),
            ending_snapshot=stage_cfg.get("ending_snapshot"),
            **optional_args,
        )

        core_command = ["get_common_crawl_urls", *args]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class RemoveDuplicates(DataCurationStage):

    def __init__(self, cfg, super_stage_name=None):
        self.super_stage_name = super_stage_name
        super().__init__(cfg)

    def setup_stage_vars(self, cfg, super_stage_name=None):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "remove_duplicates"
        if self.super_stage_name is not None:
            self.stage_cfg = cfg[self.super_stage_name].get(
                "remove_duplicate_documents",)
        else:
            self.stage_cfg = cfg.get("remove_duplicate_documents")

    def make_stage_command_groups(self,
                                  stage_cfg_path: Path) -> List[List[str]]:
        """ Builds the command groups for the current stage """
        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        command_groups = [[]]

        # If certain arguments are not specified, we remove them from the list
        optional_args = {
            "cc_news":
                stage_cfg.get('cc_news'),
            "cc_snapshot_index_file":
                Path().joinpath(
                    self.get_job_path().results_folder,
                    "collinfo.json",
                ),
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg]
            for arg in optional_args
            if optional_args[arg]
        }

        # Create the list of arguments for the command
        args = create_args_list(
            replace_underscore=True,
            log_file=Path().joinpath(self.log_folder, "get_cc_urls.log"),
            cc_index_prefix=stage_cfg.get("cc_index_prefix"),
            cc_data_domain_prefix=stage_cfg.get("cc_data_domain_prefix"),
            output_warc_url_file=stage_cfg.get("output_warc_url_file"),
            starting_snapshot=stage_cfg.get("starting_snapshot"),
            ending_snapshot=stage_cfg.get("ending_snapshot"),
            **optional_args,
        )

        core_command = ["get_common_crawl_urls", *args]

        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class ExactDeduplication(NemoMegatronStage):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.log_folder = Path()
        self.conf_folder = Path()
        self.STR2SUBSTAGECLASS = {
            'add_document_ids': AddId,
            'start_redis_cluster': StartRedisCluster,
            'compute_document_hashes': HashDocuments,
            'form_duplicate_groups': GroupDuplicates,
            'shutdown_redis_cluster': ShutdownRedisCluster,
            'remove_duplicate_documents': RemoveDuplicates,
        }

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "exact_deduplication"
        self.stage_cfg = cfg.get("exact_deduplication")

    def run(self) -> str:
        """
        Run current stage including all of the substages,
        returns job id on slurm based system otherwise empty string

        :return: job id on slurm based system otherwise empty string
        :rtype: str
        """
        # Create the job folders
        self.setup_folder_and_data()

        job_id = ""
        for sub_stage_name in self.stage_cfg.keys():
            if sub_stage_name != 'run':
                sub_stage_class = self.STR2SUBSTAGECLASS[sub_stage_name]
                # Create the sub-stage
                print(sub_stage_class)
                sub_stage = sub_stage_class(
                    self.cfg,
                    super_stage_name=self.stage_name,
                )
                if job_id:
                    dependency = f"aftercorr:{job_id}"
                    sub_stage.stage_cfg["run"]["dependency"] = dependency
                # Launch the sub-stage
                job_id = sub_stage.run()

        return job_id
