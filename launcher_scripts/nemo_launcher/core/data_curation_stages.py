import copy
import omegaconf
from typing import Dict
from pathlib import Path

from nemo_launcher.core.stages import (
    NemoMegatronStage,
    create_args_list,
)
from nemo_launcher.core.launchers import AutoLauncher


class DataCurationStage(NemoMegatronStage):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.log_folder = Path()
        self.conf_folder = Path()

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "data_curation"
        self.stage_cfg = cfg.get("data_curation")

    def setup_folder_and_data(self) -> None:
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)
        # make the results dir
        results_folder = job_path.results_folder
        results_folder.mkdir(parents=True, exist_ok=True)
        # make the log dir
        self.log_folder = Path(job_path.folder, 'log')
        self.log_folder.mkdir(parents=True, exist_ok=True)
        # Make the conf dir
        self.conf_folder = Path(job_path.folder, 'conf')
        self.conf_folder.mkdir(parents=True, exist_ok=True)


class CommonCrawl(DataCurationStage):
    # Does not have sub stages, just multiple command groups

    def run(self) -> str:
        # Create the log and res dir
        self.setup_folder_and_data()

        cluster_parameters = self._make_cluster_parameters(self.cluster)

        # Add the command to setup
        # This is because all of the srun calls are already described
        # within the example scripts in the NeMo Data Curator
        command = self._create_download_command()
        cluster_parameters['setup'] = [command]

        # Create launcher
        launcher = AutoLauncher(
            folder=self.get_job_path().folder,
            cluster=self.cluster,
            **cluster_parameters,
        )
        job_id = launcher.launch(command_groups=[])

        return job_id

    def _create_download_command(self, ) -> str:
        cfg = self.cfg
        stage_cfg = self.stage_cfg
        run_cfg = stage_cfg.get("run")
        job_path = self.get_job_path()

        # Write out the dataset builder as a separate config for the downloader
        builder_cfg = Path(self.conf_folder, f"{self.job_name}_builder.yaml")
        omegaconf.OmegaConf.save(stage_cfg.get('dataset_builder'), builder_cfg)

        download_args = create_args_list(
            log_dir=self.log_folder,
            res_dir=job_path.results_folder,
            conf_dir=self.conf_folder,
            cpus_per_node=run_cfg.get('cpus_per_node'),
            docker_image=cfg.get('container'),
            mounts=self._make_container_mounts_string(),
            starting_snapshot=stage_cfg.get(
                'url_retrieval')['starting_snapshot'],
            ending_snapshot=stage_cfg.get('url_retrieval')['ending_snapshot'],
            output_warc_url_file_name=stage_cfg.get('url_retrieval')
            ['output_warc_url_file_name'],
            builder_config_file=builder_cfg,
            output_json_dir=stage_cfg.get('output_json_dir'),
            replace_underscore=False,
        )
        download_args = [arg.replace('--', '') for arg in download_args]

        # Add the command
        cmd = download_args + [
            'bash', '/opt/nemo-data-curator/examples/download_common_crawl.sh'
        ]
        return " ".join(cmd)

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

        return cluster_params
