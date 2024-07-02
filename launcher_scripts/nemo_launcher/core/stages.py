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
import functools
import glob
import json
import logging
import omegaconf
import os
import re
import shutil
from nemo_launcher.core.launchers import AutoLauncher
from nemo_launcher.utils.data_utils.prepare_squad import (
    prepare_squad_for_fine_tuning,
    prepare_squad_for_prompt_learning,
)
from nemo_launcher.utils.job_utils import JobPaths
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Any, Dict, List, Optional

__LANGUAGE_MODELS_LIST__ = [
    "gpt3",
    "t5",
    "mt5",
    "bert",
    "bert_embedding",
    "llama",
    "gemma",
    "falcon",
    "baichuan2",
    "mistral",
    "mistral_embedding",
    "mixtral",
    "starcoder2",
    "chatglm",
    "griffin",
    "qwen2",
]
__VISION_MODELS_LIST__ = ["vit"]
__MULTIMODAL_MODELS_LIST__ = [
    "clip",
    "stable_diffusion",
    "instruct_pix2pix",
    "dreambooth",
    "imagen",
    "controlnet",
    "nsfw",
    "neva",
    "video_neva",
]


class NemoMegatronStage:
    """
    Base class for NeMo Megatron stages. All stages should build on top of this class.
    Call `run` function to run current stage.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.cluster = cfg.get("cluster_type")

        self.stage_name = None
        self.stage_cfg = None
        self.setup_stage_vars(cfg)
        self.job_name = self.stage_cfg.run.get("name")

        self.nodes_scheduler = {}

    def setup_stage_vars(self, cfg: OmegaConf):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        raise NotImplementedError

    def run(self) -> str:
        """
        Run current stage returns job id on slurm based system otherwise empty string

        :return: job id on slurm based system otherwise empty string
        :rtype: str
        """
        # Setup folders and datasets
        self.setup_folder_and_data()
        # Save stage hydra config
        job_path = self.get_job_path()

        if (
            self.cfg.get("training").get("model").get("rampup_batch_size")
            and self.stage_name == "training"
        ):
            gpus = self.stage_cfg.get("trainer").get("devices")
            self._find_optimal_nodes(self.cfg, gpus)
            current_gbs = self._get_current_gbs(self.cfg)
            nodes = self.nodes_scheduler[str(current_gbs)]
            self.stage_cfg["trainer"]["num_nodes"] = nodes
            self.cfg["training"]["trainer"]["num_nodes"] = nodes
            logging.info(
                f"global batch size and number of nodes will change following this schedule:\n {self.nodes_scheduler}"
            )

        stage_cfg_path = self.save_stage_hydra_config(
            self.stage_cfg, job_path, self.cfg
        )
        # Make cluster parameters
        cluster_parameters = self._make_cluster_parameters(self.cluster)
        # Make k8s config file if necessary
        if self.cluster == "k8s":
            template_root = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                f"k8s_templates/{self.stage_name}",
            )
            self._make_k8s_spec_file(template_root, cluster_parameters, job_path)
            self._copy_k8s_helm_chart(template_root, job_path)
        # Make command groups
        command_groups = self.make_stage_command_groups(stage_cfg_path)
        # Create launcher
        print("job_path.folder: ", job_path.folder)
        print("self.cluster: ", self.cluster)
        launcher = AutoLauncher(
            folder=job_path.folder, cluster=self.cluster, **cluster_parameters,
        )
        job_id = launcher.launch(command_groups=command_groups)

        return job_id

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders and fine-tuning/prompt-learning dataset"""
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)
        results_folder = job_path.results_folder
        results_folder.mkdir(parents=True, exist_ok=True)

    def save_stage_hydra_config(
        self, stage_cfg: OmegaConf, job_path: JobPaths, cfg: OmegaConf
    ) -> Path:
        """
        Interpolate and save hydra config file for current stage

        :param OmegaConf stage_cfg: current stage's hydra configuration
        :param JobPaths job_path: JobPaths object
        :param OmegaConf cfg: base config for job
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """
        # Since k8s uses a Helm chart that launches a job based on the Hydra config
        # file, the Hydra config file that is generated needs to contain all of the
        # required keys for each stage.
        if cfg.cluster_type == "k8s":
            # OmegaConf doesn't allow adding new keys. Temporarily create a dictionary
            # representation and add the new keys before converting back to an
            # OmegaConf object.
            temp_config = OmegaConf.to_object(stage_cfg)
            temp_config["data_dir"] = cfg.data_dir
            temp_config["cluster_type"] = cfg.cluster_type
            temp_config["launcher_scripts_path"] = cfg.launcher_scripts_path
            temp_config["data_config"] = stage_cfg.run.name
            stage_cfg = OmegaConf.create(temp_config)

        _hydra_interpolation(stage_cfg)

        cfg_save_path = job_path.config_file
        omegaconf.OmegaConf.save(stage_cfg, cfg_save_path)
        return cfg_save_path

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
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
        raise NotImplementedError

    def _make_wandb_login_command(self) -> List[str]:
        """Make a command of login with w&b api key"""
        cfg = self.cfg
        wandb_cmd = ""

        if cfg.cluster_type == "bcp" and cfg.wandb_api_bcp_secret_key is not None:
            wandb_cmd = f"wandb login ${cfg.wandb_api_bcp_secret_key}"
        elif cfg.wandb_api_key_file is not None:
            with open(cfg.wandb_api_key_file, "r") as f:
                wandb_api_key = f.readline().rstrip()
            wandb_cmd = f"wandb login {wandb_api_key}"
        return [wandb_cmd]

    def _make_nemo_path_command(self) -> List[str]:
        """Extend nemo path to python path"""
        return [
            f"cd {self._nemo_code_path}",
            "git rev-parse HEAD",
            f"export PYTHONPATH={self._nemo_code_path}:\${{PYTHONPATH}}",
        ]

    def _make_git_log_command(self, stage_cfg_path: Path):
        """log last 5 commits for repos- NeMo, megatron-lm, NeMo-Framework-Launcher or NeMo-Framework-Launcher
        'NeMo-Framework-Launcher' was renamed to 'NeMo-Framework-Launcher'. We run git log for both for
        backwards compatibility.
        """
        append_to_file = f"{stage_cfg_path.parent}/git_log.txt"
        return [
            f"(echo PYT$\"NVIDIA_PYTORCH_VERSION\" && \
                git --git-dir=/opt/NeMo/.git log -n 5 --format='NeMo;%h;%aD;%s' && \
                git --git-dir=/opt/megatron-lm/.git log -n 5 --format='megatron-lm;%h;%aD;%s' && \
                git --git-dir=/opt/NeMo-Framework-Launcher/.git log -n 5 --format='NeMo-Framework-Launcher;%h;%aD;%s' && \
                git --git-dir=/opt/NeMo-Framework-Launcher/.git log -n 5 --format='NeMo-Framework-Launcher;%h;%aD;%s') > {append_to_file}"
        ]

    def _make_k8s_spec_file(
        self, template_root: str, cluster_parameters: Dict, job_path: JobPaths
    ):
        """Create a yaml spec file for kubernetes jobs"""
        raise NotImplementedError

    # def _make_numa_mapping_command(self) -> List[str]:
    #     """Make a command of numa mapping call"""
    #     cfg = self.cfg
    #     numa_cfg = cfg.get("numa_mapping")
    #     if not numa_cfg.get("enable"):
    #         return []

    #     numa_override = [f"{k}={v}" for k, v in numa_cfg.items()]
    #     numa_command = [
    #         f"python3 -u {self._launcher_scripts_path / 'nemo_launcher/collections/numa_mapping.py'}",
    #         *numa_override,
    #     ]
    #     numa_command = " \\\n  ".join(numa_command)
    #     return [numa_command]

    def _make_api_log_command_prefix(self, results_dir: str) -> str:
        """Make a command prefix of api logging"""
        choice_model_type, choice_name = self.get_stage_config_choice()
        api_log = self.cfg.get("api_log", False)
        api_log_prefix = ""
        if api_log:
            api_log_path = os.path.join(results_dir, "api_logs")
            api_log_prefix = (
                "[[ \${SLURM_LOCALID} -eq 0 ]] && "
                f"API_LOG_CMD='apiLog.sh -p {choice_model_type}/{choice_name} -v nemo_launcher' || API_LOG_CMD=''; "
                f"LOGPATH={api_log_path} \${{API_LOG_CMD}}"
            )
        return api_log_prefix

    def _make_nsys_command_prefix(self, results_dir: str) -> str:
        """Make a command prefix of nsys profiling"""
        model_cfg = self.stage_cfg.get("model")
        if not model_cfg:
            return ""

        nsys_cfg = model_cfg.get("nsys_profile", None)
        nsys_prefix = ""
        if nsys_cfg is not None and nsys_cfg.get("enabled", False):
            profile_out_path = os.path.join(results_dir, "profile_logs")
            os.makedirs(profile_out_path, exist_ok=True)
            slurm_node = "\${SLURM_NODEID}"
            slurm_rank = "\${SLURM_PROCID}"
            slurm_jobid = "\${SLURM_JOB_ID}"
            nsys_prefix = (
                f"nsys profile -s none "
                f"-t {','.join(nsys_cfg.trace)} "
                f"-o {profile_out_path}/profile_{slurm_jobid}_node{slurm_node}_rank{slurm_rank} "
                f"--force-overwrite true "
                f"--capture-range=cudaProfilerApi "
                f"--capture-range-end=stop"
            )
        return nsys_prefix

    def _make_container_mounts_string(self) -> str:
        """
        Make container mounting string based on hydra configurations

        :return: container mounting string, e.g. "/path/to/A:/path/to/A,/path/to/B:/path/to/B,..."
        :rtype: str
        """

        def add_container_mounts(container_mounts):
            mounts_str = ""
            if container_mounts is not None:
                assert isinstance(
                    container_mounts, omegaconf.listconfig.ListConfig
                ), "container_mounts must be a list."
                for mount in container_mounts:
                    if mount is not None and isinstance(mount, str):
                        mounts_str += (
                            f",{mount}" if ":" in mount else f",{mount}:{mount}"
                        )
            return mounts_str

        cfg = self.cfg
        data_dir = cfg.get("data_dir")
        base_results_dir = cfg.get("base_results_dir")
        mounts_string = f"{self._launcher_scripts_path}:{self._launcher_scripts_path},{data_dir}:{data_dir},{base_results_dir}:{base_results_dir}"

        container_mounts = cfg.get("container_mounts")
        mounts_string += add_container_mounts(container_mounts)
        return mounts_string

    def _make_cluster_parameters(self, cluster: str) -> Dict:
        """
        Make a cluster-specific parameters for jobs on different clusters.
        Current clusters include bcm(slurm), bcp and interactive.
        For example for bcm, it will return slurm parameters:
            {'job_name': 'some_name', 'nodes': 2, 'ntasks_per_node': 8, ...}

        :param str cluster: i.e. `bcm`, `bcp`, `interactive`, etc.
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
        if nodes is None:
            nodes = stage_cfg.get("trainer").get("num_nodes")

        ntasks_per_node = run_cfg.get("ntasks_per_node")
        if ntasks_per_node is None:
            ntasks_per_node = stage_cfg.get("trainer").get("devices")

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

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
            if cfg.get("training").get("model").get("ub_tp_comm_overlap", False) or (
                cfg.get("peft") is not None
                and cfg.get("peft").get("model").get("ub_tp_comm_overlap", False)
            ):
                if "srun_args" not in cluster_cfg:
                    cluster_cfg["srun_args"] = []
                cluster_cfg["srun_args"] += ["--mpi=pmix"]
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
                {
                    **shared_parameters,
                    "no_redirect": cfg.get("bcp_no_redirect"),
                    "env_vars": env_vars,
                }
            )
        elif cluster == "interactive":
            cluster_parameters.update(shared_parameters)
        elif cluster == "k8s":
            # Resolving since there is a dependency between soon-to-be deprecated
            # cluster.nfs_path which is referenced in cluster.volumes.nfs.path
            OmegaConf.resolve(cfg.get("cluster"))
            cluster_cfg = cfg.get("cluster")
            k8s_cfg = {**copy.deepcopy(cluster_cfg)}

            cluster_parameters = {**k8s_cfg}
            cluster_parameters.update(
                {
                    **shared_parameters,
                    "container_image": container_image,
                    "env_vars": env_vars,
                }
            )

        return cluster_parameters

    def _find_optimal_nodes(self, cfg, gpus) -> None:
        nodes_scheduler_path = (
            f"{cfg.get('training').get('run').get('results_dir')}/nodes_scheduler.json"
        )

        try:
            with open(nodes_scheduler_path, "r") as nodes_scheduler:
                self.nodes_scheduler = json.load(nodes_scheduler)
        except FileNotFoundError:
            mbs = cfg.get("training").get("model").get("micro_batch_size")
            gbs = cfg.get("training").get("model").get("global_batch_size")
            rampup_bs = cfg.get("training").get("model").get("rampup_batch_size")
            tp = cfg.get("training").get("model").get("tensor_model_parallel_size")
            pp = cfg.get("training").get("model").get("pipeline_model_parallel_size")
            num_nodes = cfg.get("training").get("trainer").get("num_nodes")
            start_bs = rampup_bs[0]
            increment = rampup_bs[1]

            cbs = start_bs
            rbs = [start_bs]
            while cbs <= (gbs - increment):
                rbs.append(rbs[-1] + increment)
                cbs += increment

            self.nodes_scheduler[str(gbs)] = num_nodes
            for b in rbs[::-1][1:]:
                optimal_lst = []
                prev = int(min(list(self.nodes_scheduler.values())))
                for nodes in range(1, prev + 1):
                    dp = (gpus * nodes) // (tp * pp)
                    if (
                        b % (mbs * dp) == 0
                        and b % (mbs * gpus * nodes) == 0
                        and nodes <= prev
                    ):
                        optimal_lst.append(nodes)

                self.nodes_scheduler[str(b)] = max(optimal_lst)

            sched_rbs = [int(i) for i in self.nodes_scheduler.keys()]
            assert rbs[::-1] == sched_rbs, (
                "please, make sure you enter the correct combination of"
                " ramp up batch size and number of nodes"
            )

            with open(nodes_scheduler_path, "w") as nodes_scheduler:
                nodes_scheduler.write(json.dumps(self.nodes_scheduler))

    def _get_current_gbs(self, cfg):
        start_bs = cfg.get("training").get("model").get("rampup_batch_size")[0]
        results_dir = cfg.get("training").get("run").get("results_dir")
        os.chdir(results_dir)
        job_numbers = []

        try:
            for file in glob.glob("*.out"):
                file = file.split("_")[-1].split(".")[0]
                job_numbers.append(int(file))

            job_number = max(job_numbers)
            last_job = glob.glob(f"*{job_number}.out")[0]
            with open(last_job, "r") as logs:
                logs = logs.read()

            current_gbs = re.findall(r"global_batch_size=(\d+)", logs)[-1]
        except:
            current_gbs = start_bs

        return current_gbs

    def get_env_vars(self) -> Dict:
        """
        Set up dictionary for environment variables
        The environment variables from hydra config will be set inside the job scripts.
        For Example:
            Set `env_vars.NVTE_BIAS_DROPOUT_FUSION=1` while calling nemo_launcherlauncher-scripts,
            `NVTE_BIAS_DROPOUT_FUSION=1` will be set while running the job.

        :return: a dictionary of env vars while running the job.
        :rtype: Dict
        """
        env_vars = {k: v for k, v in self.cfg.get("env_vars").items() if v is not None}
        return env_vars

    def get_stage_config_choice(self):
        """
        Return current stages config's corresponding `choice_model_type` and `choice_name`
        For example, if `training=gpt3/5b`, then `choice_model_type=gpt3` and `choice_name=5b`
        """
        stage_config_choice = self.cfg.get(f"{self.stage_name}_config")
        choice_model_type, choice_name = stage_config_choice.rsplit("/", 1)
        return choice_model_type, choice_name

    @property
    def _launcher_scripts_path(self) -> Path:
        return Path(self.cfg.get("launcher_scripts_path"))

    @property
    def _nemo_code_path(self) -> Path:
        return Path("/opt/NeMo")

    @property
    def _data_dir(self) -> Path:
        return Path(self.cfg.get("data_dir"))

    @property
    def _rlhf_code_path(self) -> Path:
        return Path("/opt/NeMo-Aligner")

    @property
    def _aligner_code_path(self) -> Path:
        return Path("/opt/NeMo-Aligner")

    @property
    def _cuda_visible_devices(self) -> str:
        ntasks_per_node = self.stage_cfg.run.get("ntasks_per_node")
        if ntasks_per_node is None:
            ntasks_per_node = self.stage_cfg.trainer.get("devices", 1)
        return (
            "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
            if ntasks_per_node == 8
            else f"CUDA_VISIBLE_DEVICES={','.join(map(str, range(ntasks_per_node)))}"
        )

    @property
    def _cuda_device_max_connections(self) -> str:
        model_cfg = self.stage_cfg.get("model")
        if not model_cfg:
            return ""
        tensor_model_parallel_size = model_cfg.get("tensor_model_parallel_size", 1)
        context_parallel_size = model_cfg.get("context_parallel_size", 1)
        fsdp = model_cfg.get("fsdp", False)
        return (
            "CUDA_DEVICE_MAX_CONNECTIONS=1"
            if (
                (tensor_model_parallel_size > 1 or context_parallel_size > 1)
                and not fsdp
            )
            else ""
        )

    @property
    def _nvte_bias_gelu_nvfusion(self) -> str:
        """Only used in pretraining; override in training class"""
        return ""

    @functools.lru_cache()
    def get_job_path(self, sub_stage: Optional[str] = None) -> JobPaths:
        """Fetch a JobPaths object for current stage"""
        run_cfg = self.stage_cfg.get("run")
        results_dir = Path(
            run_cfg.get("results_dir")
        )  # TODO: rename this to job dir in config
        if sub_stage is not None:
            results_dir = results_dir / sub_stage
        return JobPaths(results_dir, self.job_name)

    @property
    def _set_ln_sm_margin(self) -> str:
        """Set LayerNorm SM margin when using P2P communication overlap to support the overlap with LayerNorm kernel"""
        vpp = self.cfg.training.model.get("virtual_pipeline_model_parallel_size")
        if (
            self.cfg.training.model.get("pipeline_model_parallel_size", 1) > 1
            and vpp is not None
            and vpp > 1
        ):
            get_ln_sm_margin_command = (
                f"python3 {self._launcher_scripts_path / 'nemo_launcher/collections/conditional_cfgs.py'} "
                f"name=get_ln_sm_margin"
            )
            return f"NVTE_FWD_LAYERNORM_SM_MARGIN=\$({get_ln_sm_margin_command}) NVTE_BWD_LAYERNORM_SM_MARGIN=\$({get_ln_sm_margin_command})"
        return ""

    @property
    def _skip_ag_overlap(self) -> str:
        """Skip TP-AllGather overlap with ring-exchange at (1) bf16 and (2) PP > 1"""
        if (
            self.cfg.training.model.get("ub_tp_comm_overlap", False)
            and self.cfg.training.model.get("pipeline_model_parallel_size") > 1
        ):
            use_fp8 = self.cfg.training.model.get("fp8", False)
            get_ag_overlap_command = (
                f"python3 {self._launcher_scripts_path / 'nemo_launcher/collections/conditional_cfgs.py'} "
                f"name=get_ag_overlap "
                f"fp8={use_fp8} "
            )
            return f"NVTE_UB_SPLIT_AG=\$({get_ag_overlap_command})"
        return ""


class NeMoStage(NemoMegatronStage):
    """
    Stage is a nemo stage if it uses a nemo scripts
    Current nemo stage includes:
        - pretraining
        - fine-tuning
        - prompt-learning
        - t5/mt5 eval
    GPT3 eval is not a NeMo stage because it uses eval-harness inside nemo_launcher collections.
    """

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
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
        # Training has one command group
        # Shared with fine-tuning and prompt learning
        command_groups = [[]]
        command_groups[0] += self._make_wandb_login_command()
        command_groups[0] += self._make_nemo_path_command()
        command_groups[0] += self._make_git_log_command(stage_cfg_path)
        # command_groups[0] += self._make_numa_mapping_command()

        # _cuda_device_max_connections and _cuda_visible_devices cannot be used as command prefix on BCP
        if self.cluster == "bcp":
            core_command = []
        else:
            core_command = [
                self._cuda_device_max_connections,
                self._cuda_visible_devices,
                self._set_ln_sm_margin,
                self._skip_ag_overlap,
                self._nvte_bias_gelu_nvfusion,
            ]

        core_command += [
            self._make_api_log_command_prefix(
                results_dir=self.get_job_path().results_folder
            ),
            self._make_nsys_command_prefix(
                results_dir=self.get_job_path().results_folder
            ),
            self._make_nemo_call_string(stage_cfg_path),
        ]
        core_command_string = " ".join([c for c in core_command if c])
        command_groups[0] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups

    def _make_nemo_call_string(self, stage_cfg_path: Path) -> str:
        """
        Make nemo scripts calling command string
        This is for current nemo stage's essential nemo script calling.

        :param Path stage_cfg_path: path to interpolated and saved configuration
        :return: command string of nemo script calling
        :rtype: str
        """
        choice_model_type, choice_name = self.get_stage_config_choice()
        code_path = self._get_nemo_code_path(choice_model_type)

        hydra_override = self._make_hydra_override()

        command = [
            f"python3 -u {code_path} ",
            f"--config-path={stage_cfg_path.parents[0]}",
            f"--config-name={stage_cfg_path.name}",
            *hydra_override,
        ]
        command_string = " \\\n  ".join(command)
        return command_string

    def _make_hydra_override(self) -> List:
        """
        Override some existing hydra configurations if necessary.

        Example use cases are:
            1. For bcp cluster, `+rank=\${RANK}` is required running some NeMo scripts.
                Existing hydra config doesn't have `rank` field, so we overwrite on the fly.
            2. Auto blend training dataset by overwriting empty `model.data.data_prefix` as
                `model.data.data_prefix=\$({auto_blend_command})`. Existing `model.data.data_prefix`
                could be None in cfg, so we overwrite it in this function.
        """
        hydra_override = []
        if self.cluster == "bcp":
            hydra_override += ["+rank=\${RANK}"]
        return hydra_override

    def _copy_k8s_helm_chart(self, template_root: str, job_path: JobPaths):
        """
        Copy the k8s Helm charts to the results directory.

        :param str template_root: path to where the k8s template files are located
        :param JobPaths job_path: JobPaths object
        """
        template_file = os.path.join(template_root, "training.yaml")
        chart_file = os.path.join(template_root, "Chart.yaml")
        training_path = Path(
            job_path.folder / "k8s_template" / "templates" / "training.yaml"
        )
        training_path.parent.mkdir(parents=True, exist_ok=True)
        config_path = Path(job_path.folder / "k8s_template" / "config")
        config_path.mkdir(parents=True, exist_ok=True)
        chart_path = Path(job_path.folder / "k8s_template" / "Chart.yaml")
        training_config_file = os.path.join(template_root, "training-config.yaml")
        training_config_path = Path(
            job_path.folder / "k8s_template" / "templates" / "training-config.yaml"
        )
        hydra_config_path = Path(job_path.folder / "k8s_template" / "config")

        shutil.copy2(template_file, training_path)
        shutil.copy2(chart_file, chart_path)
        shutil.copy2(training_config_file, training_config_path)
        shutil.copy2(job_path.config_file, hydra_config_path)

    def _add_wandb_key_to_chart(self) -> str:
        """
        Read the WandB API key file and return it to be placed in the Helm chart.

        :return: a string of the WandB API key.
        :rtype: str
        """
        with open(self.cfg.wandb_api_key_file, "r") as f:
            wandb_api_key = f.readline().rstrip()
        return wandb_api_key

    def _make_k8s_spec_file(
        self, template_root: str, cluster_parameters: Dict, job_path: JobPaths
    ):
        """
        Create a spec file for a Kubernetes training job.
        The spec file is generated based on the parameters in the cluster and training config files.

        :param str template_root: path to where the k8s template files are located
        :param Dict cluster_parameters: settings specific to the cluster that is being used
        :param JobPaths job_path: JobPaths object
        """
        with open(os.path.join(template_root, "values.yaml")) as value_file:
            values_template = OmegaConf.load(value_file)

        values_template.image.trainingImage = cluster_parameters["container_image"]
        values_template.image.pullSecret = cluster_parameters["pull_secret"]
        values_template.image.numGPUs = self.stage_cfg.trainer.devices
        values_template.image.nodes = self.stage_cfg.trainer.num_nodes
        values_template.trainingConfig.shmSize = cluster_parameters["shm_size"]
        # TODO: NFSServer and NFSPath will eventually be deprecated
        values_template.trainingConfig.NFSServer = cluster_parameters["nfs_server"]
        values_template.trainingConfig.NFSPath = cluster_parameters["nfs_path"]
        values_template.volumes = cluster_parameters["volumes"]
        values_template.trainingConfig.ibResourceName = cluster_parameters[
            "ib_resource_name"
        ]
        values_template.trainingConfig.ibCount = cluster_parameters["ib_count"]
        values_template.trainingConfig.ibNetworkAnnotation = cluster_parameters[
            "ib_network_annotation"
        ]
        values_template.trainingConfig.envVars = cluster_parameters["env_vars"]

        if cluster_parameters["dns_policy"] is not None:
            values_template.trainingConfig.dnsPolicy = cluster_parameters["dns_policy"]

        if self.cfg.wandb_api_key_file is not None:
            values_template.trainingConfig.wandbKey = self._add_wandb_key_to_chart()

        k8s_template_path = job_path.folder
        k8s_template_file = Path(k8s_template_path / "k8s_template" / "values.yaml")
        k8s_template_file.parent.mkdir(parents=True, exist_ok=True)

        conf = OmegaConf.create(values_template)
        OmegaConf.save(conf, k8s_template_file)

    def get_env_vars(self) -> Dict:
        """
        Set up dictionary for environment variables
        The environment variables from hydra config will be set inside the job scripts.
        For Example:
            Set `env_vars.NVTE_BIAS_DROPOUT_FUSION=1` while calling nemo_launcherlauncher-scripts,
            `NVTE_BIAS_DROPOUT_FUSION=1` will be set while running the job.

        :return: a dictionary of env vars while running the job.
        :rtype: Dict
        """
        env_vars = super().get_env_vars()
        devices = self.stage_cfg.trainer.get("devices", 1)
        if self.cluster != "bcm":
            env_vars["SLURM_NTASKS_PER_NODE"] = devices
        if self.cluster in ["bcp", "k8s"]:  # Set env prefix as env var on BCP
            for env_var_str in [
                self._cuda_device_max_connections,
                self._cuda_visible_devices,
                self._set_ln_sm_margin,
                self._skip_ag_overlap,
            ]:
                if env_var_str:
                    var_name, var_val = env_var_str.split("=")
                    env_vars[var_name] = var_val
        return env_vars


class Training(NeMoStage):
    """Stage class of pretraining with NeMo scripts"""

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "training"
        self.stage_cfg = cfg.get("training")

    def _make_hydra_override(self) -> List:
        """
        Override some existing hydra configurations if necessary.
        Example use cases are:
            1. For bcp cluster, `+rank=\${RANK}` is required running some NeMo scripts.
                Existing hydra config doesn't have `rank` field, so we overwrite on the fly.
            2. Auto blend training dataset by overwriting empty `model.data.data_prefix` as
                `model.data.data_prefix=\$({auto_blend_command})`. Existing `model.data.data_prefix`
                could be None in cfg, so we overwrite it in this function.

        :return: hydra override string added in nemo script calling
        :rtype: str
        """
        hydra_override = []
        choice_model_type, choice_name = self.get_stage_config_choice()
        if self.cluster == "bcp":
            hydra_override += ["+rank=\${RANK}"]
        if (
            choice_model_type in __LANGUAGE_MODELS_LIST__
            and self.stage_cfg.model.data.get("data_prefix", None) is None
        ):
            preprocessed_dir = self.stage_cfg.run.get("preprocessed_dir")
            blending_alpha = self.stage_cfg.run.get("blending_alpha")
            auto_blend_command = (
                f"python3 {self._launcher_scripts_path / 'nemo_launcher/collections/auto_blend.py'} "
                f"model_type={choice_model_type} "
                f"preprocessed_dir={preprocessed_dir} "
                f"blending_alpha={blending_alpha}"
            )
            hydra_override += [f"model.data.data_prefix=\$({auto_blend_command})"]
        if self.stage_cfg.model.get("gc_interval", 0) > 1:
            gc_interval = min(
                self.stage_cfg.model.get("gc_interval"),
                self.cfg.training.trainer.get("val_check_interval"),
            )
            hydra_override += [f"model.gc_interval={gc_interval}"]
        return hydra_override

    def _get_nemo_code_path(self, model_type: str) -> Path:
        """
        Provide the essential nemo code path for running the stage, usually different model types use different nemo scripts.
        For example, `megatron_t5_pretraining.py` for t5 and `megatron_gpt_pretraining.py` for gpt3.

        :param str model_type: i.e. `gpt3`, `t5`, `mt5`, etc.
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """
        model_type_to_code_path = {
            "t5": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_t5_pretraining.py",
            "mt5": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_t5_pretraining.py",
            "gpt3": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
            "llama": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
            "baichuan2": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
            "nemotron": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
            "bert": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_bert_pretraining.py",
            "falcon": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
            "chatglm": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
            "starcoder2": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
            "retro": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_retro_pretraining.py",
            "vit": self._nemo_code_path
            / "examples/vision/vision_transformer/megatron_vit_classification_pretrain.py",
            "clip": self._nemo_code_path
            / "examples/multimodal/vision_language_foundation/clip/megatron_clip_pretrain.py",
            "nsfw": self._nemo_code_path
            / "examples/multimodal/vision_language_foundation/nsfw/megatron_nsfw_pretrain.py",
            "stable_diffusion": self._nemo_code_path
            / "examples/multimodal/text_to_image/stable_diffusion/sd_train.py",
            "instruct_pix2pix": self._nemo_code_path
            / "examples/multimodal/text_to_image/instruct_pix2pix/sd_finetune.py",
            "imagen": self._nemo_code_path
            / "examples/multimodal/text_to_image/imagen/imagen_training.py",
            "dreambooth": self._nemo_code_path
            / "examples/multimodal/text_to_image/dreambooth/dreambooth.py",
            "controlnet": self._nemo_code_path
            / "examples/multimodal/text_to_image/controlnet/controlnet_train.py",
            "nerf": self._nemo_code_path / "examples/multimodal/x_to_nerf/nerf/main.py",
            "neva": self._nemo_code_path
            / "examples/multimodal/multimodal_llm/neva/neva_pretrain.py",
            "video_neva": self._nemo_code_path
            / "examples/multimodal/multimodal_llm/neva/neva_pretrain.py",
            "mistral": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
            "mixtral": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
            "qwen2": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
        }
        return model_type_to_code_path[model_type]


class FineTuning(NeMoStage):
    """Stage class of fine-tuning with NeMo scripts"""

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "fine_tuning"
        self.stage_cfg = cfg.get("fine_tuning")

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders and fine-tuning/prompt-learning dataset"""
        super().setup_folder_and_data()
        choice_model_type, choice_name = self.get_stage_config_choice()

        if choice_model_type in __LANGUAGE_MODELS_LIST__:
            # Prepare fine-tuning dataset
            data_dir = self.cfg.get("data_dir")
            task_name = self.stage_cfg.run.get("task_name")

            # GLUE for internal use
            download_glue_script_path = (
                self._launcher_scripts_path
                / "nemo_launcher/utils/data_utils/download_glue.py"
            )
            if download_glue_script_path.exists():
                from nemo_launcher.utils.data_utils.download_glue import (
                    TASKS_LOWER,
                    download_glue,
                )

                if task_name in TASKS_LOWER:
                    download_glue(
                        data_dir=os.path.join(data_dir, "glue_data"), tasks=task_name
                    )

            # Prepare dataset for squad
            if task_name in ["squad", "xquad"]:
                prepare_squad_for_fine_tuning(
                    data_dir=os.path.join(data_dir, "squad_data")
                )

    def _get_nemo_code_path(self, model_type: str) -> Path:
        """
        Provide the essential nemo code path for running the stage, usually different model types use different nemo scripts.
        For example, `megatron_t5_pretraining.py` for t5 and `megatron_gpt_pretraining.py` for gpt3.

        :param str model_type: i.e. `gpt3`, `t5`, `mt5`, etc.
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """

        model_type_to_code_path = {
            "bert_embedding": self._nemo_code_path
            / "examples/nlp/information_retrieval/megatron_bert_embedding_finetuning.py",
            "gpt3": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "llama": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "code_llama": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "t5": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_t5_seq2seq_finetune.py",
            "mt5": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_t5_seq2seq_finetune.py",
            "falcon": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "chatglm": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "starcoder2": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "gemma": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "vit": self._nemo_code_path
            / "examples/vision/vision_transformer/megatron_vit_classification_finetune.py",
            "neva": self._nemo_code_path
            / "examples/multimodal/multimodal_llm/neva/neva_finetune.py",
            "nsfw": self._nemo_code_path
            / "examples/multimodal/vision_language_foundation/nsfw/megatron_nsfw_pretrain.py",
            "baichuan2": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "mistral": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "mixtral": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "qwen2": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
        }
        return model_type_to_code_path[model_type]


class PEFT(NeMoStage):
    """Stage class of PEFT with NeMo scripts"""

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "peft"
        self.stage_cfg = cfg.get("peft")

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders and fine-tuning/prompt-learning dataset"""
        # Setup folders
        super().setup_folder_and_data()
        choice_model_type, choice_name = self.get_stage_config_choice()

        if choice_model_type in __LANGUAGE_MODELS_LIST__:
            # Prepare prompt learning dataset
            data_dir = self.cfg.get("data_dir")
            task_name = self.stage_cfg.run.get("task_name")

            # Prepare dataset for squad
            if task_name in ("squad", "xquad"):
                self._task_data_dir = os.path.join(data_dir, "squad_data")
                if self.cfg.cluster_type == "k8s":
                    # Skip downloading since on k8s the data is downloaded in a
                    # pre-install job since user may not be using a volume type
                    # that's not available locally, e.g., PVC
                    return
                prepare_squad_for_fine_tuning(data_dir=self._task_data_dir)

    def _copy_k8s_helm_chart(self, template_root: str, job_path: JobPaths):
        """
        Copy the k8s Helm charts to the results directory.

        :param str template_root: path to where the k8s template files are located
        :param JobPaths job_path: JobPaths object
        """
        template_file = os.path.join(template_root, "peft.yaml")
        chart_file = os.path.join(template_root, "Chart.yaml")
        prompt_path = Path(job_path.folder / "k8s_template" / "templates" / "peft.yaml")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        config_path = Path(job_path.folder / "k8s_template" / "config")
        config_path.mkdir(parents=True, exist_ok=True)
        chart_path = Path(job_path.folder / "k8s_template" / "Chart.yaml")
        prompt_config_file = os.path.join(template_root, "peft-config.yaml")
        prompt_config_path = Path(
            job_path.folder / "k8s_template" / "templates" / "peft-config.yaml"
        )
        hydra_config_path = Path(job_path.folder / "k8s_template" / "config")

        shutil.copy2(template_file, prompt_path)
        shutil.copy2(chart_file, chart_path)
        shutil.copy2(prompt_config_file, prompt_config_path)
        shutil.copy2(job_path.config_file, hydra_config_path)

    def _make_k8s_spec_file(
        self, template_root: str, cluster_parameters: Dict, job_path: JobPaths
    ):
        """
        Create a spec file for a Kubernetes PEFT job.

        The spec file is generated based on the parameters in the cluster and
        PEFT config files.

        :param str template_root: path to where the k8s template files are located
        :param Dict cluster_parameters: settings specific to the cluster that is being used
        :param JobPaths job_path: JobPaths object
        """
        with open(os.path.join(template_root, "values.yaml")) as value_file:
            values_template = OmegaConf.load(value_file)

        choice_model_type, _ = self.get_stage_config_choice()

        values_template.image.trainingImage = cluster_parameters["container_image"]
        values_template.image.pullSecret = cluster_parameters["pull_secret"]
        values_template.image.gpuNum = self.stage_cfg.trainer.devices
        values_template.image.nodes = self.stage_cfg.trainer.num_nodes
        values_template.trainingConfig.shmSize = cluster_parameters["shm_size"]
        # TODO: NFSServer and NFSPath will eventually be deprecated
        values_template.trainingConfig.NFSServer = cluster_parameters["nfs_server"]
        values_template.trainingConfig.NFSPath = cluster_parameters["nfs_path"]
        values_template.volumes = cluster_parameters["volumes"]
        values_template.trainingConfig.scriptPath = str(
            self._get_nemo_code_path(choice_model_type)
        )
        values_template.trainingConfig.envVars = cluster_parameters["env_vars"]

        values_template.datasetConfig.prepare_task_name = self.stage_cfg.run.get(
            "task_name"
        )
        values_template.datasetConfig.task_data_dir = self._task_data_dir

        if cluster_parameters["dns_policy"] is not None:
            values_template.trainingConfig.dnsPolicy = cluster_parameters["dns_policy"]

        if self.cfg.wandb_api_key_file is not None:
            values_template.trainingConfig.wandbKey = self._add_wandb_key_to_chart()

        k8s_template_path = job_path.folder
        k8s_template_file = Path(k8s_template_path / "k8s_template" / "values.yaml")
        k8s_template_file.parent.mkdir(parents=True, exist_ok=True)

        conf = OmegaConf.create(values_template)
        OmegaConf.save(conf, k8s_template_file)

    def _get_nemo_code_path(self, model_type: str) -> Path:
        """
        Provide the essential nemo code path for running the stage, usually different model types use different nemo scripts.
        For example, `megatron_t5_pretraining.py` for t5 and `megatron_gpt_pretraining.py` for gpt3.

        :param str model_type: i.e. `gpt3`, `t5`, `mt5`, etc.
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """

        if model_type == "mt5":
            raise NotImplementedError(
                "PEFT is not supported in NeMo Megatron mt5 models."
            )
        model_type_to_code_path = {
            "mistral_embedding": self._nemo_code_path
            / "examples/nlp/information_retrieval/megatron_gpt_embedding_finetuning.py",
            "gpt3": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "llama": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "baichuan2": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "chatglm": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "t5": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_t5_finetuning.py",
            "falcon": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "starcoder2": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "gemma": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "griffin": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_griffin_finetuning.py",
            "nemotron": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "neva": self._nemo_code_path
            / "examples/multimodal/multimodal_llm/neva/neva_peft.py",
            "mistral": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "mixtral": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
            "qwen2": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
        }
        return model_type_to_code_path[model_type]


class PromptLearning(NeMoStage):
    """Stage class of prompt-learning with NeMo scripts"""

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "prompt_learning"
        self.stage_cfg = cfg.get("prompt_learning")

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders and fine-tuning/prompt-learning dataset"""
        # Setup folders
        super().setup_folder_and_data()

        # Prepare prompt learning dataset
        data_dir = self.cfg.get("data_dir")
        task_name = self.stage_cfg.run.get("task_name")
        # Prepare squad dataset
        if task_name == "squad":
            prepare_squad_for_prompt_learning(
                os.path.join(data_dir, "prompt_data"), self._launcher_scripts_path,
            )

    def _get_nemo_code_path(self, model_type: str) -> Path:
        """
        Provide the essential nemo code path for running the stage, usually different model types use different nemo scripts.
        For example, `megatron_t5_pretraining.py` for t5 and `megatron_gpt_pretraining.py` for gpt3.

        :param str model_type: i.e. `gpt3`, `t5`, `mt5`, etc.
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """
        model_type_to_code_path = {
            "gpt3": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_prompt_learning.py",
            "llama": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_prompt_learning.py",
            "baichuan2": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_prompt_learning.py",
            "t5": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_t5_prompt_learning.py",
            "mt5": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_t5_prompt_learning.py",
            "chatglm": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_prompt_learning.py",
            "mixtral": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_prompt_learning.py",
        }
        return model_type_to_code_path[model_type]


class AdapterLearning(PromptLearning):
    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "adapter_learning"
        self.stage_cfg = cfg.get("adapter_learning")

    def _get_nemo_code_path(self, model_type: str) -> Path:
        """
        Provide the essential nemo code path for running the stage, usually different model types use different nemo scripts.
        For example, `megatron_t5_pretraining.py` for t5 and `megatron_gpt_pretraining.py` for gpt3.

        :param str model_type: i.e. `gpt3`, `t5`, `mt5`, etc.
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """
        model_type_to_code_path = {
            "gpt3": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_adapter_tuning.py",
            "llama": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_adapter_tuning.py",
            "baichuan2": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_adapter_tuning.py",
            "t5": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_t5_adapter_tuning.py",
            "chatglm": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_adapter_tuning.py",
            "mistral": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_adapter_tuning.py",
            "mixtral": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_adapter_tuning.py",
        }
        return model_type_to_code_path[model_type]


class IA3Learning(PromptLearning):
    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "ia3_learning"
        self.stage_cfg = cfg.get("ia3_learning")

    def _get_nemo_code_path(self, model_type: str) -> Path:
        """
        Provide the essential nemo code path for running the stage, usually different model types use different nemo scripts.
        For example, `megatron_t5_pretraining.py` for t5 and `megatron_gpt_pretraining.py` for gpt3.

        :param str model_type: i.e. `gpt3`, `t5`, `mt5`, etc.
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """
        model_type_to_code_path = {
            "gpt3": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_ia3_tuning.py",
            "llama": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_ia3_tuning.py",
            "baichuan2": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_ia3_tuning.py",
            "t5": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_t5_ia3_tuning.py",
            "chatglm": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_ia3_tuning.py",
        }
        return model_type_to_code_path[model_type]


class FWInference(NeMoStage):
    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "fw_inference"
        self.stage_cfg = cfg.get("fw_inference")

    def _get_nemo_code_path(self, model_type: str) -> Path:
        """
        Provide the essential nemo code path for running the stage, usually different model types use different nemo scripts.
        For example, `megatron_t5_pretraining.py` for t5 and `megatron_gpt_pretraining.py` for gpt3.

        :param str model_type: i.e. `gpt3`, `t5`, `mt5`, etc.
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """
        model_type_to_code_path = {
            "vit": self._nemo_code_path
            / "examples/vision/vision_transformer/megatron_vit_classification_infer.py",
            "clip": self._nemo_code_path
            / "examples/multimodal/vision_language_foundation/clip/megatron_clip_infer.py",
            "nsfw": self._nemo_code_path
            / "examples/multimodal/vision_language_foundation/nsfw/megatron_nsfw_infer.py",
            "stable_diffusion": self._nemo_code_path
            / "examples/multimodal/text_to_image/stable_diffusion/sd_infer.py",
            "instruct_pix2pix": self._nemo_code_path
            / "examples/multimodal/text_to_image/instruct_pix2pix/sd_edit_cli.py",
            "dreambooth": self._nemo_code_path
            / "examples/multimodal/text_to_image/dreambooth/dreambooth_infer.py",
            "imagen": self._nemo_code_path
            / "examples/multimodal/text_to_image/imagen/imagen_infer.py",
            "controlnet": self._nemo_code_path
            / "examples/multimodal/text_to_image/controlnet/controlnet_infer.py",
            "neva": self._nemo_code_path
            / "examples/multimodal/multimodal_llm/neva/neva_evaluation.py",
            "video_neva": self._nemo_code_path
            / "examples/multimodal/multimodal_llm/neva/neva_evaluation.py",
            "retro": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_retro_eval.py",
        }
        return model_type_to_code_path[model_type]


class RAGIndexing(NeMoStage):
    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "rag_indexing"
        self.stage_cfg = cfg.get("rag_indexing")

    def _get_nemo_code_path(self, model_type: str) -> Path:
        model_type_to_code_path = {
            "bert": self._nemo_code_path / "examples/nlp/rag/rag_indexing.py",
        }
        return model_type_to_code_path[model_type]

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
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
        # Training has one command group
        # Shared with fine-tuning and prompt learning
        command_groups = [[]]
        command_groups[0] += self._make_wandb_login_command()
        command_groups[0] += self._make_nemo_path_command()
        command_groups[0] += self._make_git_log_command(stage_cfg_path)
        # command_groups[0] += self._make_numa_mapping_command()

        # commands for installing dependencies
        package_command_string = "pip install llama-index==0.10.33"
        command_groups[0] += [package_command_string]

        # _cuda_device_max_connections and _cuda_visible_devices cannot be used as command prefix on BCP
        if self.cluster == "bcp":
            core_command = []
        else:
            core_command = [
                self._cuda_device_max_connections,
                self._cuda_visible_devices,
                self._set_ln_sm_margin,
                self._skip_ag_overlap,
                self._nvte_bias_gelu_nvfusion,
            ]

        core_command += [
            self._make_api_log_command_prefix(
                results_dir=self.get_job_path().results_folder
            ),
            self._make_nsys_command_prefix(
                results_dir=self.get_job_path().results_folder
            ),
            self._make_nemo_call_string(stage_cfg_path),
        ]
        core_command_string = " ".join([c for c in core_command if c])
        command_groups[0] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class RAGGenerating(NeMoStage):
    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "rag_generating"
        self.stage_cfg = cfg.get("rag_generating")

    def _get_nemo_code_path(self, model_type: str) -> Path:
        model_type_to_code_path = {
            "gpt3": self._nemo_code_path / "examples/nlp/rag/rag_generating.py",
        }
        return model_type_to_code_path[model_type]

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
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
        # Training has one command group
        # Shared with fine-tuning and prompt learning
        command_groups = [[]]
        command_groups[0] += self._make_wandb_login_command()
        command_groups[0] += self._make_nemo_path_command()
        command_groups[0] += self._make_git_log_command(stage_cfg_path)
        # command_groups[0] += self._make_numa_mapping_command()

        # commands for installing dependencies
        package_command_string = "pip install llama-index==0.10.33"
        command_groups[0] += [package_command_string]

        # _cuda_device_max_connections and _cuda_visible_devices cannot be used as command prefix on BCP
        if self.cluster == "bcp":
            core_command = []
        else:
            core_command = [
                self._cuda_device_max_connections,
                self._cuda_visible_devices,
                self._set_ln_sm_margin,
                self._skip_ag_overlap,
                self._nvte_bias_gelu_nvfusion,
            ]

        core_command += [
            self._make_api_log_command_prefix(
                results_dir=self.get_job_path().results_folder
            ),
            self._make_nsys_command_prefix(
                results_dir=self.get_job_path().results_folder
            ),
            self._make_nemo_call_string(stage_cfg_path),
        ]
        core_command_string = " ".join([c for c in core_command if c])
        command_groups[0] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class Conversion(NemoMegatronStage):
    """Stage class of converting training checkpoints to .nemo format"""

    def setup_stage_vars(self, cfg: OmegaConf):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "conversion"
        self.stage_cfg = cfg.get("conversion")

    def _make_hparams_override_command(self):
        """
        Make the command string to override some fields in hparams.yaml file while converting checkpoint into .nemo format

        :return: command string for hparams override with the script in collections
        :rtype: str
        """
        choice_model_type, choice_name = self.get_stage_config_choice()
        model_cfg = self.stage_cfg.get("model")

        if choice_model_type not in __LANGUAGE_MODELS_LIST__ + ["stable_diffusion"]:
            hparams_file = model_cfg.get("hparams_file")
            output_path = self.get_job_path().results_folder
            hparams_override = output_path / "hparams_override.yaml"
            return [f"cp {hparams_file} {hparams_override}"]

        hparams_file = model_cfg.get("hparams_file", "null")
        vocab_file = model_cfg.get("vocab_file", "null")
        merge_file = model_cfg.get("merge_file", "null")
        tokenizer_model = model_cfg.get("tokenizer_model", "null")
        override_configs = {
            "hparams_file": hparams_file,
            "output_path": self.get_job_path().results_folder,
            "vocab_file": vocab_file,
            "merge_file": merge_file,
            "tokenizer_model": tokenizer_model,
        }
        hparams_override = [f"{k}={v}" for k, v in override_configs.items()]
        override_command = [
            f"python3 -u {self._launcher_scripts_path / 'nemo_launcher/collections/hparams_override.py'}",
            *hparams_override,
        ]
        override_command = " \\\n  ".join(override_command)
        return [override_command]

    def _make_checkpoint_search_command(self, **kwargs: Any) -> str:
        """
        Make the command string to search for the latest checkpoint inside checkpoint folder

        :param Path **kwargs: checkpoint search script's argument override
        :return: command string for searching for latest checkpoint with the script in collections
        :rtype: str
        """
        checkpoint_override = [f"{k}={v}" for k, v in kwargs.items()]
        return (
            f"python3 {self._launcher_scripts_path / 'nemo_launcher/collections/checkpoint_search.py'} "
            f"{' '.join(checkpoint_override)}"
        )

    def _make_k8s_spec_file(
        self, template_root: str, cluster_parameters: Dict, job_path: JobPaths
    ):
        """
        Create a spec file for a Kubernetes conversion job.
        The spec file is generated based on the parameters in the cluster and conversion config files.

        :param str template_root: path to where the k8s template files are located
        :param Dict cluster_parameters: settings specific to the cluster that is being used
        :param JobPaths job_path: JobPaths object
        """
        with open(os.path.join(template_root, "values.yaml")) as value_file:
            values_template = OmegaConf.load(value_file)

        num_gpus = (
            self.cfg.conversion.model.pipeline_model_parallel_size
            * self.cfg.conversion.model.tensor_model_parallel_size
        )

        values_template.image.trainingImage = cluster_parameters["container_image"]
        values_template.image.pullSecret = cluster_parameters["pull_secret"]
        values_template.image.gpuNum = num_gpus
        values_template.trainingConfig.shmSize = cluster_parameters["shm_size"]
        # TODO: NFSServer and NFSPath will eventually be deprecated
        values_template.trainingConfig.NFSServer = cluster_parameters["nfs_server"]
        values_template.trainingConfig.NFSPath = cluster_parameters["nfs_path"]
        values_template.volumes = cluster_parameters["volumes"]
        values_template.trainingConfig.vocabPath = self.cfg.conversion.model.vocab_file
        values_template.trainingConfig.mergesPath = self.cfg.conversion.model.merge_file
        values_template.trainingConfig.resultsDirectory = str(job_path.folder)
        values_template.trainingConfig.trainingDirectory = (
            self.cfg.conversion.run.train_dir
        )
        values_template.trainingConfig.launcherScriptsPath = (
            self.cfg.launcher_scripts_path
        )
        values_template.trainingConfig.tensorParallelism = (
            self.cfg.conversion.model.tensor_model_parallel_size
        )
        values_template.trainingConfig.pipelineParallelism = (
            self.cfg.conversion.model.pipeline_model_parallel_size
        )
        values_template.trainingConfig.envVars = cluster_parameters["env_vars"]

        if cluster_parameters["dns_policy"] is not None:
            values_template.trainingConfig.dnsPolicy = cluster_parameters["dns_policy"]

        k8s_template_path = job_path.folder
        k8s_template_file = Path(k8s_template_path / "k8s_template" / "values.yaml")
        k8s_template_file.parent.mkdir(parents=True, exist_ok=True)

        conf = OmegaConf.create(values_template)
        OmegaConf.save(conf, k8s_template_file)

    def _copy_k8s_helm_chart(self, template_root: str, job_path: JobPaths):
        """
        Copy the k8s Helm charts to the results directory.

        :param str template_root: path to where the k8s template files are located
        :param JobPaths job_path: JobPaths object
        """
        template_file = os.path.join(template_root, "conversion.yaml")
        chart_file = os.path.join(template_root, "Chart.yaml")
        conversion_path = Path(
            job_path.folder / "k8s_template" / "templates" / "conversion.yaml"
        )
        conversion_path.parent.mkdir(parents=True, exist_ok=True)
        chart_path = Path(job_path.folder / "k8s_template" / "Chart.yaml")

        shutil.copy2(template_file, conversion_path)
        shutil.copy2(chart_file, chart_path)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
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
        choice_model_type, choice_name = self.get_stage_config_choice()

        command_groups = [[], []]
        command_groups[0] += self._make_hparams_override_command()
        run_cfg = self.stage_cfg.get("run")
        model_cfg = self.stage_cfg.get("model")
        checkpoint_search_command = self._make_checkpoint_search_command(
            checkpoint_folder=model_cfg.get("checkpoint_folder"),
            checkpoint_name=model_cfg.get("checkpoint_name"),
            tensor_model_parallel_size=model_cfg.get("tensor_model_parallel_size", 1),
            pipeline_model_parallel_size=model_cfg.get(
                "pipeline_model_parallel_size", 1
            ),
        )
        command_groups[-1] += [f"export CKPT_NAME=$({checkpoint_search_command})"]

        nemo_file_name = run_cfg.get("nemo_file_name")
        hparams_override_file = (
            self.get_job_path().results_folder / "hparams_override.yaml"
        )
        nemo_file_path = self.get_job_path().results_folder / nemo_file_name

        if choice_model_type in __LANGUAGE_MODELS_LIST__:
            code_path = (
                self._nemo_code_path
                / "examples/nlp/language_modeling/megatron_ckpt_to_nemo.py"
            )
        elif choice_model_type in __VISION_MODELS_LIST__:
            code_path = self._nemo_code_path / "examples/vision/convert_ckpt_to_nemo.py"
        elif choice_model_type in __MULTIMODAL_MODELS_LIST__:
            code_path = (
                self._nemo_code_path / "examples/multimodal/convert_ckpt_to_nemo.py"
            )
        else:
            raise ValueError(f"Model type: {choice_model_type} not recognized")

        args = create_args_list(
            replace_underscore=False,
            gpus_per_node=run_cfg.get("ntasks_per_node"),
            model_type=model_cfg.get("model_type"),
            checkpoint_folder=model_cfg.get("checkpoint_folder"),
            checkpoint_name="\${CKPT_NAME}",
            hparams_file=hparams_override_file,
            nemo_file_path=nemo_file_path,
            tensor_model_parallel_size=model_cfg.get("tensor_model_parallel_size", 1),
            pipeline_model_parallel_size=model_cfg.get(
                "pipeline_model_parallel_size", 1
            ),
        )
        if model_cfg.get("pipeline_model_parallel_split_rank") is not None:
            args += create_args_list(
                replace_underscore=False,
                pipeline_model_parallel_split_rank=model_cfg.get(
                    "pipeline_model_parallel_split_rank"
                ),
            )
        if not run_cfg.get("pack_nemo_file", True):
            args += create_args_list(
                replace_underscore=False, no_pack_nemo_file="store_true",
            )

        args += ["--bcp"] if self.cluster == "bcp" else []

        core_command = [f"python3 -u {code_path}", *args]
        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class ExternalConversion(NemoMegatronStage):
    """Stage class of converting external checkpoints to .nemo format"""

    def setup_stage_vars(self, cfg: OmegaConf):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "external_conversion"
        self.stage_cfg = cfg.get("external_conversion")

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
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
        choice_model_type, choice_name = self.get_stage_config_choice()
        if choice_model_type == "clip":
            code_path = (
                self._nemo_code_path
                / "examples/multimodal/vision_language_foundation/clip/convert_external_clip_to_nemo.py"
            )
        else:
            raise NotImplementedError(
                f"Model type `{choice_model_type}` doesn't support conversion from external source."
            )

        command_groups = [[]]
        run_cfg = self.stage_cfg.get("run")
        model_cfg = self.stage_cfg.get("model")
        nemo_file_name = run_cfg.get("nemo_file_name")
        nemo_file_path = self.get_job_path().results_folder / nemo_file_name
        args = create_args_list(
            replace_underscore=False,
            gpus_per_node=run_cfg.get("ntasks_per_node"),
            arch=model_cfg.get("arch"),
            version=model_cfg.get("version"),
            hparams_file=model_cfg.get("hparams_file"),
            nemo_file_path=nemo_file_path,
            tensor_model_parallel_size=model_cfg.get("tensor_model_parallel_size", 1),
            pipeline_model_parallel_size=model_cfg.get(
                "pipeline_model_parallel_size", 1
            ),
        )
        args += ["--bcp"] if self.cluster == "bcp" else []

        core_command = [f"python3 -u {code_path}", *args]
        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class NeMoEvaluation(NeMoStage):
    """
    Stage class of gpt3/t5/mt5 evaluation with NeMo scripts
    Including: fine-tuning eval, prompt-learning eval, adapter/ia3 learning eval
    """

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "evaluation"
        self.stage_cfg = cfg.get("evaluation")

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
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
        command_groups = super().make_stage_command_groups(stage_cfg_path)

        choice_model_type, choice_name = self.get_stage_config_choice()
        if any(
            [
                choice_model_type.startswith(type)
                for type in ["prompt", "ia3", "adapter"]
            ]
        ):
            pred_file_path = self.stage_cfg.get("pred_file_path")
            ground_truth_file_path = self.stage_cfg.get("ground_truth_file_path")
            code_path = (
                self._launcher_scripts_path
                / "nemo_launcher/collections/metric_calculation/squad_metric_calc.py"
            )
            args = create_args_list(
                pred=pred_file_path, ground_truth=ground_truth_file_path,
            )
            split_string = self.stage_cfg.get("split_string", None)
            if split_string:
                args += create_args_list(split_string=f"'{split_string}'")
            calculation_command = [f"python3 {code_path}", *args]
            calculation_command = " \\\n  ".join(calculation_command)
        elif choice_model_type.startswith("peft"):
            calculation_command = None
        elif choice_name == "squad":
            output_file_path_prefix = self.stage_cfg.model.data.validation_ds.get(
                "output_file_path_prefix"
            )
            pred_file_path = (
                output_file_path_prefix
                + "_validation_dataloader0_inputs_preds_labels.jsonl"
            )
            ground_truth_file_path = self.stage_cfg.model.data.validation_ds.get(
                "ground_truth_file_path"
            )
            code_path = (
                self._launcher_scripts_path
                / "nemo_launcher/collections/metric_calculation/fine_tuning_metric_calc.py"
            )
            args = create_args_list(
                replace_underscore=False,
                pred_file=pred_file_path,
                target_file=ground_truth_file_path,
                squad_eval_script_path=self._launcher_scripts_path
                / "nemo_launcher/collections/metric_calculation/squad_metric_calc.py",
            )
            calculation_command = [f"python3 {code_path}", *args]
            calculation_command = " \\\n  ".join(calculation_command)
        else:
            calculation_command = None

        if calculation_command is not None:
            command_groups += [[calculation_command]]
        return command_groups

    def _get_nemo_code_path(self, model_type: str) -> Path:
        """
        Provide the essential nemo code path for running the stage, usually different model types use different nemo scripts.
        For example, `megatron_t5_pretraining.py` for t5 and `megatron_gpt_pretraining.py` for gpt3.

        :param str model_type: i.e. `gpt3`, `t5`, `mt5`, etc.
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """
        if model_type in ["gpt3", "prompt_gpt3"]:
            raise ValueError(
                "Evaluating GPT-3 models needs `EvalHarnessEvaluation` class."
            )
        model_type_to_code_path = {
            "t5": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_t5_seq2seq_eval.py",
            "mt5": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_t5_seq2seq_eval.py",
            "prompt_t5": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_t5_prompt_learning_eval.py",
            "prompt_mt5": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_t5_prompt_learning_eval.py",
            "retro": self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_retro_qatask_eval.py",
            "ia3_t5": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_t5_ia3_eval.py",
            "ia3_gpt3": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_ia3_eval.py",
            "adapter_t5": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_t5_adapter_eval.py",
            "adapter_gpt3": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_adapter_eval.py",
            "peft_llama": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_generate.py",
            "code_llama": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_generate.py",
            "peft_falcon": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_generate.py",
            "peft_starcoder2": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_generate.py",
            "peft_baichuan2": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_generate.py",
            "peft_chatglm": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_generate.py",
            "peft_mistral": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_generate.py",
            "peft_mixtral": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_generate.py",
            "peft_qwen2": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_generate.py",
            "vit": self._nemo_code_path
            / "examples/vision/vision_transformer/megatron_vit_classification_evaluate.py",
            "clip": self._nemo_code_path
            / "examples/multimodal/vision_language_foundation/clip/megatron_clip_imagenet_zeroshot.py",
            "mistral": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_generate.py",
            "mixtral": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_generate.py",
            "starcoder2": self._nemo_code_path
            / "examples/nlp/language_modeling/tuning/megatron_gpt_generate.py",
        }
        return model_type_to_code_path[model_type]


class EvalHarnessEvaluation(NemoMegatronStage):
    """Stage class of gpt-3 evaluation harness"""

    def __init__(self, cfg):
        super().__init__(cfg)
        choice_model_type, choice_name = self.get_stage_config_choice()
        self.prompt_evaluation = True if "prompt" in choice_model_type else False

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "evaluation"
        self.stage_cfg = cfg.get("evaluation")

    def _make_download_command_string(self) -> str:
        """
        Make dataset download command for evaluation harness.

        :return: command string of downloading evaluation data
        :rtype: str
        """
        data_dir = self.cfg.get("data_dir")
        cache_dir = os.path.join(data_dir, "eval_harness_data")
        run_cfg = self.stage_cfg.get("run")
        tasks = run_cfg.get("tasks")

        code_path = (
            self._launcher_scripts_path
            / "nemo_launcher/collections/eval_harness/download.py"
        )
        args = create_args_list(tasks=tasks, cache_dir=cache_dir,)
        download_command = [f"python3 {code_path}", *args]
        download_command_string = " \\\n  ".join(download_command)
        return download_command_string

    def _make_k8s_spec_file(
        self, template_root: str, cluster_parameters: Dict, job_path: JobPaths
    ):
        """
        Create a spec file for a Kubernetes conversion job.
        The spec file is generated based on the parameters in the cluster and conversion config files.

        :param str template_root: path to where the k8s template files are located
        :param Dict cluster_parameters: settings specific to the cluster that is being used
        :param JobPaths job_path: JobPaths object
        """
        with open(os.path.join(template_root, "values.yaml")) as value_file:
            values_template = OmegaConf.load(value_file)

        num_gpus = (
            self.cfg.evaluation.model.pipeline_model_parallel_size
            * self.cfg.evaluation.model.tensor_model_parallel_size
        )

        values_template.image.trainingImage = cluster_parameters["container_image"]
        values_template.image.pullSecret = cluster_parameters["pull_secret"]
        values_template.image.gpuNum = num_gpus
        values_template.trainingConfig.shmSize = cluster_parameters["shm_size"]
        # TODO: NFSServer and NFSPath will eventually be deprecated
        values_template.trainingConfig.NFSServer = cluster_parameters["nfs_server"]
        values_template.trainingConfig.NFSPath = cluster_parameters["nfs_path"]
        values_template.volumes = cluster_parameters["volumes"]
        values_template.trainingConfig.vocabPath = self.cfg.evaluation.model.vocab_file
        values_template.trainingConfig.mergesPath = self.cfg.evaluation.model.merge_file
        values_template.trainingConfig.resultsDirectory = str(job_path.folder)
        values_template.trainingConfig.trainingDirectory = (
            self.cfg.evaluation.run.train_dir
        )
        values_template.trainingConfig.launcherScriptsPath = (
            self.cfg.launcher_scripts_path
        )
        values_template.trainingConfig.tensorParallelism = (
            self.cfg.evaluation.model.tensor_model_parallel_size
        )
        values_template.trainingConfig.pipelineParallelism = (
            self.cfg.evaluation.model.pipeline_model_parallel_size
        )
        values_template.trainingConfig.name = self.cfg.evaluation.run.name
        values_template.trainingConfig.model = self.cfg.evaluation.model.model_type
        values_template.trainingConfig.cacheDir = os.path.join(
            self.cfg.data_dir, "eval_harness_data"
        )
        values_template.trainingConfig.outputPath = os.path.join(
            self.cfg.evaluation.run.results_dir,
            self.cfg.evaluation.run.eval_name,
            "results",
        )
        values_template.trainingConfig.batchSize = (
            self.cfg.evaluation.model.eval_batch_size
        )
        values_template.trainingConfig.precision = self.cfg.evaluation.model.precision
        values_template.trainingConfig.nemoModel = self.cfg.evaluation.model.nemo_model
        values_template.trainingConfig.checkpointFolder = (
            self.cfg.evaluation.model.checkpoint_folder
        )
        values_template.trainingConfig.checkpointName = (
            self.cfg.evaluation.model.checkpoint_name
        )
        values_template.trainingConfig.hparamsFile = (
            self.cfg.evaluation.model.hparams_file
        )
        values_template.trainingConfig.tasks = self.cfg.evaluation.run.tasks
        values_template.trainingConfig.envVars = cluster_parameters["env_vars"]

        if cluster_parameters["dns_policy"] is not None:
            values_template.trainingConfig.dnsPolicy = cluster_parameters["dns_policy"]

        k8s_template_path = job_path.folder
        k8s_template_file = Path(k8s_template_path / "k8s_template" / "values.yaml")
        k8s_template_file.parent.mkdir(parents=True, exist_ok=True)

        conf = OmegaConf.create(values_template)
        OmegaConf.save(conf, k8s_template_file)

    def _copy_k8s_helm_chart(self, template_root: str, job_path: JobPaths):
        """
        Copy the k8s Helm charts to the results directory.

        :param str template_root: path to where the k8s template files are located
        :param JobPaths job_path: JobPaths object
        """
        template_file = os.path.join(template_root, "evaluation.yaml")
        chart_file = os.path.join(template_root, "Chart.yaml")
        evaluation_path = Path(
            job_path.folder / "k8s_template" / "templates" / "evaluation.yaml"
        )
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        config_path = Path(job_path.folder / "k8s_template" / "config")
        config_path.mkdir(parents=True, exist_ok=True)
        chart_path = Path(job_path.folder / "k8s_template" / "Chart.yaml")
        evaluation_config_file = os.path.join(template_root, "evaluation-config.yaml")
        evaluation_config_path = Path(
            job_path.folder / "k8s_template" / "templates" / "evaluation-config.yaml"
        )
        hparams_config_path = Path(job_path.folder / "k8s_template" / "config")

        shutil.copy2(template_file, evaluation_path)
        shutil.copy2(chart_file, chart_path)
        shutil.copy2(evaluation_config_file, evaluation_config_path)
        shutil.copy2(
            os.path.join(self.cfg.evaluation.run.train_dir, "results", "hparams.yaml"),
            hparams_config_path,
        )

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
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
        if self.prompt_evaluation:
            command_groups = [[]]
        else:
            command_groups = [[], []]
            command_groups[0] += [self._make_download_command_string()]

        data_dir = self.cfg.get("data_dir")
        cache_dir = os.path.join(data_dir, "eval_harness_data")
        run_cfg = self.stage_cfg.get("run")
        model_cfg = self.stage_cfg.get("model")

        code_path = (
            self._launcher_scripts_path
            / "nemo_launcher/collections/eval_harness/evaluate.py"
        )
        args = create_args_list(
            replace_underscore=False,
            name=run_cfg.get("name"),
            model=model_cfg.get("model_type"),
            tasks=run_cfg.get("tasks"),
            cache_dir=cache_dir,
            output_path=self.get_job_path().results_folder,
            batch_size=model_cfg.get("eval_batch_size"),
            tensor_model_parallel_size=model_cfg.get("tensor_model_parallel_size"),
            pipeline_model_parallel_size=model_cfg.get("pipeline_model_parallel_size"),
            precision=model_cfg.get("precision"),
        )

        if self.prompt_evaluation:
            args += create_args_list(
                replace_underscore=False,
                nemo_model=model_cfg.get("nemo_model"),
                prompt_dataset_paths=model_cfg.get("prompt_dataset_paths"),
            )
        else:
            # GPT evaluation
            args += create_args_list(
                replace_underscore=False,
                vocab_file=model_cfg.get("vocab_file"),
                merge_file=model_cfg.get("merge_file"),
                nemo_model=model_cfg.get("nemo_model"),
                checkpoint_folder=model_cfg.get("checkpoint_folder"),
                checkpoint_name=model_cfg.get("checkpoint_name"),
                tokenizer_model=model_cfg.get("tokenizer_model"),
                hparams_file=model_cfg.get("hparams_file"),
            )

        core_command = [f"python3 -u {code_path}", *args]
        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class DiffusionModelEvaluation(NemoMegatronStage):
    """
    DiffusionModelEvaluation is class for evaluating generative diffusion models.
    It can hold multiple sub-stages. For example, generation and gathering.
    They have dependencies on each other and will be launched one by one.
    """

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "evaluation"
        self.stage_cfg = cfg.get("evaluation")

    def _make_sub_stages(self) -> List[str]:
        """
        Create a list of sub-stage names which are required to run in current data stage.
        Based on the input config, some of sub stages may not need to run.

        :return: a list of sub-stage names which are required to run
        :rtype: List[str]
        """
        sub_stages = []
        if self.stage_cfg.get("generate_images", False):
            sub_stages += ["generate"]
        if self.stage_cfg.get("compute_fid_scores", False):
            sub_stages += ["fid"]
        if self.stage_cfg.get("compute_clip_scores", False):
            sub_stages += ["clip"]
        if self.stage_cfg.get("plot_fid_clip", False):
            sub_stages += ["plot"]
        return sub_stages

    def _make_sub_stage_command(
        self, stage_cfg_path: Path, sub_stage: str
    ) -> List[str]:
        """Make a command of the specified sub-stage"""

        eval_diffusion_path = (
            self._launcher_scripts_path
            / "nemo_launcher/collections/eval_diffusion_fid_clip"
        )
        stage_to_code_path = {
            "fid": eval_diffusion_path / "eval_fid.py",
            "clip": eval_diffusion_path / "compute_clip_score.py",
            "plot": eval_diffusion_path / "plot.py",
        }
        choice_model_type, choice_name = self.get_stage_config_choice()
        if choice_model_type == "stable_diffusion":
            stage_to_code_path["generate"] = (
                self._nemo_code_path
                / "examples/multimodal/text_to_image/stable_diffusion/generate_fid_images.py"
            )
        elif choice_model_type == "imagen":
            stage_to_code_path["generate"] = (
                self._nemo_code_path
                / "examples/multimodal/text_to_image/imagen/generate_fid_images.py"
            )

        code_path = stage_to_code_path[sub_stage]

        args = []
        stage_cfg = self.stage_cfg
        if sub_stage == "generate":
            args = [
                f"--config-path={stage_cfg_path.parents[0]}",
                f"--config-name={stage_cfg_path.name}",
            ]
        elif sub_stage == "fid":
            args = create_args_list(
                replace_underscore=False,
                coco_images_path=stage_cfg.fid.coco_images_path,
                fid_images_path=stage_cfg.fid.save_path,
                output_path=os.path.join(
                    stage_cfg.run.get("results_dir", "."), "fid_scores.csv"
                ),
            )
        elif sub_stage == "clip":
            args = create_args_list(
                replace_underscore=False,
                captions_path=stage_cfg.fid.coco_captions_path,
                fid_images_path=stage_cfg.fid.save_path,
                output_path=os.path.join(
                    stage_cfg.run.get("results_dir", "."), "clip_scores.csv"
                ),
                clip_version=stage_cfg.clip_version,
            )
        elif sub_stage == "plot":
            args = create_args_list(
                replace_underscore=False,
                fid_scores_csv=os.path.join(
                    stage_cfg.run.get("results_dir", "."), "fid_scores.csv"
                ),
                clip_scores_csv=os.path.join(
                    stage_cfg.run.get("results_dir", "."), "clip_scores.csv"
                ),
                output_path=os.path.join(
                    stage_cfg.run.get("results_dir", "."), "fid_clip_plot.pdf"
                ),
            )

        sub_stage_command = [f"python3 -u {code_path}", *args]
        sub_stage_command = " \\\n  ".join(sub_stage_command)
        return [sub_stage_command]

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
                dependency = f"afterok:{job_id}"
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

    def make_stage_command_groups(
        self, stage_cfg_path: Path, sub_stage: Optional[str] = None,
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

        command_groups[0] += self._make_sub_stage_command(stage_cfg_path, sub_stage)
        command_groups = clean_command_groups(command_groups)
        return command_groups

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
            run_cfg.get("node_array_size") if sub_stage in ["generate"] else 1
        )
        array = f"0-{node_array_size - 1}"
        if sub_stage == "generate":
            ntasks_per_node = run_cfg.get("ntasks_per_node")
        else:
            ntasks_per_node = 1

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        if cluster == "bcm":
            return {
                "nodes": 1,
                "array": f"{array}%{node_array_size}",
                "container_image": container_image,
                "container_mounts": container_mounts,
                "ntasks_per_node": ntasks_per_node,
            }
        if cluster == "bcp":
            return {
                "nodes": node_array_size,
                "ntasks_per_node": ntasks_per_node,
                "bcp_launcher": "'mpirun --allow-run-as-root'",
            }
        return {}

    def _make_cluster_parameters(
        self, cluster: str, sub_stage: Optional[str] = None,
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
        dependency = run_cfg.get("dependency")

        env_vars = self.get_env_vars()
        env_vars[
            "PYTHONPATH"
        ] = f"{self._launcher_scripts_path}:${{PYTHONPATH}}"  # Required by pile download
        env_vars["NGC_ARRAY_TYPE"] = "MPIJob"  # Required by BCP
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
                {**shared_parameters, **private_parameters,}
            )
        elif cluster == "interactive":
            raise ValueError("Data preparation is not supported in interactive mode.")

        return cluster_parameters


def clean_command_groups(command_groups: List[List[str]]) -> List[List[str]]:
    """
    Remove empty command group in command groups

    :param List[List[str]] command_groups: command groups is a list of command group
    :return: cleaned command groups
    :rtype: List[List[str]]
    """
    for ind, command_group in enumerate(command_groups):
        command_groups[ind] = [c for c in command_group if c]
    return command_groups


def _hydra_interpolation(cfg: OmegaConf) -> None:
    """
    Interpolate hydra config values in cfg object, bypassing lazy interpolation

    :param OmegaConf cfg: OmegaConf object with the config to be interpolated
    :return: None
    """

    def interpolate(cfg: OmegaConf):
        if isinstance(cfg, omegaconf.dictconfig.DictConfig):
            for k, v in cfg.items():
                cfg[k] = interpolate(v)
        elif isinstance(cfg, omegaconf.listconfig.ListConfig):
            for i, v in enumerate(cfg):
                cfg[i] = interpolate(v)
        return cfg

    interpolate(cfg)


def create_args_list(
    hydra: bool = False, replace_underscore: bool = True, **kwargs: Any,
) -> List[str]:
    """
    An easy tool function to convert arguments into a list of argument strings.
    For example, `create_args_list(a=123, b=456)` will generate `['--a=123', '--b=456']`.

    :param bool hydra: Either a hydra argument or regular argument, `--` will be added to regular arguments
    :param bool replace_underscore: Whether to replace `_` with `-` in arguments' names.
    :params Any **kwargs: argument name and their value
    :return: A list of argument strings, e.g. `['--a=123', '--b=456', ...]`
    :rtype: List[str]
    """

    args = []
    for k, v in kwargs.items():
        if hydra:
            if isinstance(v, dict) or isinstance(v, omegaconf.dictconfig.DictConfig):
                # remove quotes around keys if the argument is a dict
                # (https://hydra.cc/docs/advanced/override_grammar/basic/)
                # For example, dict {"a":10, "b":20} will become string "'{a:10,b:20}'"
                data = ",".join(
                    f"{inner_key}:{inner_val}" for inner_key, inner_val in v.items()
                )
                args.append(f"'{k}={{{data}}}'")
            elif isinstance(v, list) or isinstance(v, omegaconf.listconfig.ListConfig):
                data = ",".join(v)
                args.append(f"'{k}=[{data}]'")
            else:
                args.append(f"{k}={v}")
        else:
            # use "store_true" to add keys only args
            if replace_underscore:
                k = k.replace("_", "-")
            args.append(f"--{k}" if v == "store_true" else f"--{k}={v}")
    return args


class SteerLMRegSFT(NeMoStage):
    """Stage class of reward model training with NeMo-Aligner scripts"""

    def setup_stage_vars(self, cfg: OmegaConf):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "steerlm_reg"
        self.stage_cfg = cfg.get("steerlm_reg")

    def setup_folder_and_data(self) -> None:
        """Setup job/data folders and OASST train-val splitted dataset"""
        super().setup_folder_and_data()

        # Prepare fine-tuning dataset
        data_dir = self.cfg.get("data_dir")
        task_name = self.stage_cfg.run.get("task_name")

    def _get_nemo_code_path(self, model_type: str) -> Path:
        """
        Provide the essential nemo code path for running the stage, usually different model types use different nemo scripts.
        For example, `megatron_t5_pretraining.py` for t5 and `megatron_gpt_pretraining.py` for gpt3.

        :param str model_type: i.e. `rw_sft`, `ac_sft`... etc.
        :return: path current stage's essential NeMo-Aligner scripts code
        :rtype: Path
        """

        model_type_to_code_path = {
            "rw_sft": f"{self._aligner_code_path}/examples/nlp/gpt/train_reward_model.py",
            "ac_sft": f"{self._aligner_code_path}/examples/nlp/gpt/train_gpt_sft.py",
        }
        return model_type_to_code_path[model_type]


class ConversionHF2NeMo(NeMoStage):
    """Stage class of reward model training with NeMo-Aligner scripts"""

    def setup_stage_vars(self, cfg: OmegaConf):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "conversion_hf2nemo"
        self.stage_cfg = cfg.get("conversion_hf2nemo")

    def _make_hparams_override_command(self):
        """
        Make the command string to override some fields in hparams.yaml file while converting checkpoint into .nemo format

        :return: command string for hparams override with the script in collections
        :rtype: str
        """
        model_cfg = self.stage_cfg.get("model")
        hparams_file = model_cfg.get("hparams_file")
        vocab_file = model_cfg.get("vocab_file")
        merge_file = model_cfg.get("merge_file")
        tokenizer_model = model_cfg.get("tokenizer_model")
        override_configs = {
            "hparams_file": hparams_file,
            "output_path": self.get_job_path().results_folder,
            "vocab_file": vocab_file,
            "merge_file": merge_file,
            "tokenizer_model": tokenizer_model,
        }
        hparams_override = [f"{k}={v}" for k, v in override_configs.items()]
        override_command = [
            f"python3 -u {self._launcher_scripts_path / 'nemo_launcher/collections/hparams_override.py'}",
            *hparams_override,
        ]
        override_command = " \\\n  ".join(override_command)
        return [override_command]

    def _make_checkpoint_search_command(self, **kwargs: Any) -> str:
        """
        Make the command string to search for the latest checkpoint inside checkpoint folder

        :param Path **kwargs: checkpoint search script's argument override
        :return: command string for searching for latest checkpoint with the script in collections
        :rtype: str
        """
        checkpoint_override = [f"{k}={v}" for k, v in kwargs.items()]
        return (
            f"python3 {self._launcher_scripts_path / 'nemo_launcher/collections/checkpoint_search.py'} "
            f"{' '.join(checkpoint_override)}"
        )

    def _make_k8s_spec_file(
        self, template_root: str, cluster_parameters: Dict, job_path: JobPaths
    ):
        """
        Create a spec file for a Kubernetes conversion job.
        The spec file is generated based on the parameters in the cluster and conversion config files.

        :param str template_root: path to where the k8s template files are located
        :param Dict cluster_parameters: settings specific to the cluster that is being used
        :param JobPaths job_path: JobPaths object
        """
        with open(os.path.join(template_root, "values.yaml")) as value_file:
            values_template = OmegaConf.load(value_file)

        num_gpus = (
            self.cfg.conversion.model.pipeline_model_parallel_size
            * self.cfg.conversion.model.tensor_model_parallel_size
        )

        values_template.image.trainingImage = cluster_parameters["container_image"]
        values_template.image.pullSecret = cluster_parameters["pull_secret"]
        values_template.image.gpuNum = num_gpus
        values_template.trainingConfig.shmSize = cluster_parameters["shm_size"]
        # TODO: NFSServer and NFSPath will eventually be deprecated
        values_template.trainingConfig.NFSServer = cluster_parameters["nfs_server"]
        values_template.trainingConfig.NFSPath = cluster_parameters["nfs_path"]
        values_template.trainingConfig.vocabPath = self.cfg.conversion.model.vocab_file
        values_template.trainingConfig.mergesPath = self.cfg.conversion.model.merge_file
        values_template.trainingConfig.resultsDirectory = str(job_path.folder)
        values_template.trainingConfig.trainingDirectory = (
            self.cfg.conversion.run.train_dir
        )
        values_template.trainingConfig.launcherScriptsPath = (
            self.cfg.launcher_scripts_path
        )
        values_template.trainingConfig.tensorParallelism = (
            self.cfg.conversion.model.tensor_model_parallel_size
        )
        values_template.trainingConfig.pipelineParallelism = (
            self.cfg.conversion.model.pipeline_model_parallel_size
        )
        values_template.trainingConfig.envVars = cluster_parameters["env_vars"]

        if cluster_parameters["dns_policy"] is not None:
            values_template.trainingConfig.dnsPolicy = cluster_parameters["dns_policy"]

        k8s_template_path = job_path.folder
        k8s_template_file = Path(k8s_template_path / "k8s_template" / "values.yaml")
        k8s_template_file.parent.mkdir(parents=True, exist_ok=True)

        conf = OmegaConf.create(values_template)
        OmegaConf.save(conf, k8s_template_file)

    def _copy_k8s_helm_chart(self, template_root: str, job_path: JobPaths):
        """
        Copy the k8s Helm charts to the results directory.

        :param str template_root: path to where the k8s template files are located
        :param JobPaths job_path: JobPaths object
        """
        template_file = os.path.join(template_root, "conversion.yaml")
        chart_file = os.path.join(template_root, "Chart.yaml")
        conversion_path = Path(
            job_path.folder / "k8s_template" / "templates" / "conversion.yaml"
        )
        conversion_path.parent.mkdir(parents=True, exist_ok=True)
        chart_path = Path(job_path.folder / "k8s_template" / "Chart.yaml")

        shutil.copy2(template_file, conversion_path)
        shutil.copy2(chart_file, chart_path)

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
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
        command_groups = [[], []]
        run_cfg = self.stage_cfg.get("run")
        model_cfg = self.stage_cfg.get("model")

        nemo_file_name = run_cfg.get("nemo_file_name")
        nemo_file_path = self.get_job_path().results_folder / nemo_file_name
        code_path = (
            self._nemo_code_path
            / "scripts/checkpoint_converters/convert_llama_hf_to_nemo.py"
        )
        args = create_args_list(
            in_file=run_cfg.get("huggingface_ckpt_path"),
            out_file=run_cfg.get("nemo_file_name"),
        )
        if model_cfg.get("pipeline_model_parallel_split_rank") is not None:
            args += create_args_list(
                replace_underscore=False,
                pipeline_model_parallel_split_rank=model_cfg.get(
                    "pipeline_model_parallel_split_rank"
                ),
            )

        args += ["--bcp"] if self.cluster == "bcp" else []

        core_command = [f"python3 -u {code_path}", *args]
        core_command_string = " \\\n  ".join(core_command)
        command_groups[-1] += [core_command_string]
        command_groups = clean_command_groups(command_groups)

        return command_groups


class PostTrainingQuantization(NeMoStage):
    """
    Stage class of post-training quantization.
    """

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "ptq"
        self.stage_cfg = cfg.get("ptq")

    def _get_nemo_code_path(self, model_type: str) -> Path:
        """
        Provide the essential nemo code path for running the stage, usually different model types use different nemo scripts.
        For example, `megatron_t5_pretraining.py` for t5 and `megatron_gpt_pretraining.py` for gpt3.

        :param str model_type: i.e. `gpt3`, `t5`, `mt5`, etc.
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """
        return (
            self._nemo_code_path
            / "examples/nlp/language_modeling/megatron_gpt_ptq.py"
        )
