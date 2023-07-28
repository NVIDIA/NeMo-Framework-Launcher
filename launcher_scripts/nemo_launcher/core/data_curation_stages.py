import copy
import shlex
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
    def __init__(self, cfg):
        super().__init__(cfg)
        self.log_folder = Path()
        self.conf_folder = Path()

    def setup_folder_and_data(self):
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

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

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
                "container_image": container_image,
                "container_mounts": container_mounts,
            })
            cluster_params[
                "job_name"] = job_name_prefix + cluster_params["job_name"]
            cluster_params['nodes'] = nodes
            cluster_params['partition'] = partition

        return cluster_params

    def run(self):
        self.setup_folder_and_data()
        job_path = self.get_job_path()

        cluster_parameters = self._make_cluster_parameters(self.cluster)
        stage_cfg_path = NemoMegatronStage.save_stage_hydra_config(
            self.stage_cfg,
            job_path,
        )

        command_groups = self.make_stage_command_groups(stage_cfg_path)

        launcher = AutoLauncher(
            folder=self.get_job_path().folder,
            cluster=self.cluster,
            **cluster_parameters,
        )

        job_id = launcher.launch(command_groups)

        return job_id


class QualityFiltering(DataCurationStage):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_stage_vars(self, cfg):
        self.stage_name = "quality_filtering"
        self.stage_cfg = cfg.get("quality_filtering")

    def make_stage_command_groups(
        self,
        stage_cfg_path: Path,
    ) -> List[List[str]]:

        stage_cfg = self.stage_cfg

        # Write out the filter configuration as a separate config file
        filter_cfg = Path(self.conf_folder, "heuristic_filter.yaml")
        omegaconf.OmegaConf.save(stage_cfg.get('filter'), filter_cfg)

        command_groups = [[]]

        optional_args = {
            "output_removed_document_dir":
            stage_cfg.get('output_removed_document_dir'),
            "output_document_score_dir":
            stage_cfg.get('output_document_score_dir'),
        }

        # Remove any arguments that are not specified
        optional_args = {
            arg: optional_args[arg]
            for arg in optional_args if optional_args[arg]
        }

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
