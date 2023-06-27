import copy
import shlex
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
        self._ndc_path = "/opt/nemo-data-curator"
        self._ndc_script_name = None

    def setup_stage_vars(self, cfg):
      raise NotImplementedError

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

    def create_copy_command(self) -> None:
        cfg = self.cfg
        job_path = self.get_job_path()
        container = cfg.get('container')

        mounts = self._make_container_mounts_string()
        script_path = f"{self._ndc_path}/examples/{self._ndc_script_name}"

        copy_command = shlex.join([
            'srun',
            '--nodes=1',
            f'--container-image={container}',
            f'--container-mounts={mounts}',
            'bash',
            '-c',
            f'cp {script_path} {job_path.folder}',
        ])

        return copy_command

    def create_ndc_example_command(self, ) -> str:
      raise NotImplementedError

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

    def run(self) -> str:
        # Create the log and res dir
        self.setup_folder_and_data()

        cluster_parameters = self._make_cluster_parameters(self.cluster)

        # Add the command to setup
        # This is because all of the srun calls are already described
        # within the example scripts in the NeMo Data Curator
        # TODO: handle the case of changing arguments by removing those
        #       lines from the script
        #       so python will take a "template" script and modify it
        #       based on boolean arguments
        #       need a command that first copies the template
        #       out of the container
        copy_command = self.create_copy_command()
        command = self.create_ndc_example_command()
        cluster_parameters['setup'] = [copy_command, command]

        # Create launcher
        launcher = AutoLauncher(
            folder=self.get_job_path().folder,
            cluster=self.cluster,
            **cluster_parameters,
        )
        job_id = launcher.launch(command_groups=[])

        return job_id


class CommonCrawl(DataCurationStage):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._ndc_script_name = 'download_common_crawl.sh'

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "common_crawl"
        self.stage_cfg = cfg.get("common_crawl")

    def create_ndc_example_command(self, ) -> str:
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
            'bash', f'{job_path.folder}/{self._ndc_script_name}'
        ]
        return " ".join(cmd)


class Filter(DataCurationStage):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._ndc_script_name = 'filter_documents.sh'

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "filter"
        self.stage_cfg = cfg.get("filter")

    def create_ndc_example_command(self) -> str:
        cfg = self.cfg
        stage_cfg = self.stage_cfg
        run_cfg = stage_cfg.get("run")
        job_path = self.get_job_path()

        # Write out the filter configuration as a separate config file
        filter_cfg = Path(self.conf_folder, f"{self.job_name}_filter.yaml")
        omegaconf.OmegaConf.save(stage_cfg.get('filter'), filter_cfg)

        filter_args = create_args_list(
            log_dir=self.log_folder,
            res_dir=job_path.results_folder,
            conf_dir=self.conf_folder,
            cpus_per_node=run_cfg.get('cpus_per_node'),
            docker_image=cfg.get('container'),
            mounts=self._make_container_mounts_string(),
            input_dir=stage_cfg.get('input_dir'),
            output_dir=stage_cfg.get('output_dir'),
            filter_config_file=filter_cfg,
            replace_underscore=False,
        )
        filter_args = [arg.replace('--', '') for arg in filter_args]

        # Add the command
        cmd = filter_args + [
            'bash', f'{job_path.folder}/{self._ndc_script_name}'
        ]
        return " ".join(cmd)


class TextCleaning(DataCurationStage):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._ndc_script_name = 'text_cleaning.sh'

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "text_cleaning"
        self.stage_cfg = cfg.get("text_cleaning")

    def create_ndc_example_command(self) -> str:
        cfg = self.cfg
        stage_cfg = self.stage_cfg
        run_cfg = stage_cfg.get("run")
        job_path = self.get_job_path()

        filter_args = create_args_list(
            log_dir=self.log_folder,
            res_dir=job_path.results_folder,
            conf_dir=self.conf_folder,
            cpus_per_node=run_cfg.get('cpus_per_node'),
            docker_image=cfg.get('container'),
            mounts=self._make_container_mounts_string(),
            input_dir=stage_cfg.get('input_dir'),
            output_dir=stage_cfg.get('output_dir'),
            replace_underscore=False,
        )
        filter_args = [arg.replace('--', '') for arg in filter_args]

        # Add the command
        cmd = filter_args + [
            'bash', f'{job_path.folder}/{self._ndc_script_name}'
        ]
        return " ".join(cmd)


class LangIDAndSeparation(DataCurationStage):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._ndc_script_name = 'lang_id_and_separation.sh'

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "lang_id"
        self.stage_cfg = cfg.get("lang_id")

    def create_ndc_example_command(self) -> str:
        cfg = self.cfg
        stage_cfg = self.stage_cfg
        run_cfg = stage_cfg.get("run")
        job_path = self.get_job_path()

        # Write out the lang id config as a separate config
        lang_id_cfg = stage_cfg.get('lang_id')
        # Rename the parameters so filter_documents can read it
        filter_cfg = {}
        filter_cfg['filter_module'] = lang_id_cfg['lang_id_module']
        filter_cfg['params'] = lang_id_cfg['params']

        output_cfg = Path(self.conf_folder, f"{self.job_name}_lang_id.yaml")
        omegaconf.OmegaConf.save(filter_cfg, output_cfg)

        filter_args = create_args_list(
            log_dir=self.log_folder,
            res_dir=job_path.results_folder,
            conf_dir=self.conf_folder,
            cpus_per_node=run_cfg.get('cpus_per_node'),
            docker_image=cfg.get('container'),
            mounts=self._make_container_mounts_string(),
            input_dir=stage_cfg.get('input_dir'),
            output_dir=stage_cfg.get('output_dir'),
            filter_config_file=output_cfg,
            replace_underscore=False,
        )
        filter_args = [arg.replace('--', '') for arg in filter_args]

        # Add the command
        cmd = filter_args + [
            'bash', f'{job_path.folder}/{self._ndc_script_name}'
        ]
        return " ".join(cmd)
