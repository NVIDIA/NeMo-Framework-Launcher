from pydantic import BaseModel
from nemo_launcher.core.v2.step_k8s import (
    create_pytorchjob_resource,
    create_mpijob_resource,
    delete_pytorchjob,
)
from nemo_launcher.core.v2.config_k8s import (
    K8sClusterConfig,
    instantiate_model_from_omegaconf,
    adapt_volume_to,
)
from omegaconf import DictConfig
from typing import Any, ClassVar, Optional
from hera.workflows import (
    Container,
    Parameter,
    Step,
    Steps,
    Workflow,
    script,
    Parameter,
    DAG,
    models as m,
)
import os
from omegaconf import OmegaConf
from nemo_launcher.utils.job_utils import JobPaths
from nemo_launcher.core.launchers import K8SLauncherV2
from pydantic import computed_field
from pathlib import Path
from textwrap import dedent
from nemo_launcher.core.stages import create_args_list


class Stage(BaseModel):
    # stage_cfg can be None for generic jobs that don't pass-thru configs to the called script
    stage_cfg: Optional[DictConfig]
    cluster_cfg: BaseModel
    stage_name: ClassVar[str]

    class Config:
        arbitrary_types_allowed = True  # For DictConfig

    @computed_field
    @property
    def job_name(self) -> str:
        return self.stage_cfg.run.name

    @computed_field
    @property
    def job_path(self) -> JobPaths:
        """Fetch a JobPaths object for current stage"""
        run_cfg = self.stage_cfg.run
        results_dir = Path(
            run_cfg.get("results_dir")
        )  # TODO: rename this to job dir in config
        return JobPaths(results_dir, self.job_name)

    def make_local_commands(self) -> list[str]:
        raise NotImplementedError

    def make_docker_commands(self) -> list[str]:
        raise NotImplementedError

    def make_slurm_commands(self) -> list[str]:
        raise NotImplementedError

    def make_k8s_workflow(self) -> Workflow:
        raise NotImplementedError

    def setup_local_results_dir(self) -> None:
        """Setup job folders"""
        self.job_path.folder.mkdir(parents=True, exist_ok=True)
        results_folder = self.job_path.results_folder
        results_folder.mkdir(parents=True, exist_ok=True)

    def run(self):
        self.setup_local_results_dir()
        cluster_cfg = self.cluster_cfg

        if isinstance(cluster_cfg, K8sClusterConfig):
            workflow = self.make_k8s_workflow()
            launcher = K8SLauncherV2(job_path=self.job_path)
            launcher.launch(workflow)
        else:
            # Right now gate all v2 stages since each launcher config needs to be supported
            raise NotImplementedError


class Training(Stage):
    image: str
    env: dict[str, Any]
    n_workers: int
    gpus_per_worker: int
    nemo_train_script: str = "examples/nlp/language_modeling/megatron_gpt_pretraining.py"
    wandb_api_key: Optional[str] = None
    stage_name: ClassVar[str] = "training"

    @classmethod
    def _from_omegaconf(cls, cfg: OmegaConf) -> "Training":
        # TODO: Rewrite to use model for validation
        # This constructor is to bridge from the old way of specifying configs

        model_type_to_code_path = {
            "t5": "examples/nlp/language_modeling/megatron_t5_pretraining.py",
            "mt5": "examples/nlp/language_modeling/megatron_t5_pretraining.py",
            "gpt3": "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
            "llama": "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
            "nemotron": "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
            "bert": "examples/nlp/language_modeling/megatron_bert_pretraining.py",
            "falcon": "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
            "retro": "examples/nlp/language_modeling/megatron_retro_pretraining.py",
            "vit": "examples/vision/vision_transformer/megatron_vit_classification_pretrain.py",
            "clip": "examples/multimodal/vision_language_foundation/clip/megatron_clip_pretrain.py",
            "nsfw": "examples/multimodal/vision_language_foundation/nsfw/megatron_nsfw_pretrain.py",
            "stable_diffusion": "examples/multimodal/text_to_image/stable_diffusion/sd_train.py",
            "sdxl": "examples/multimodal/text_to_image/stable_diffusion/sd_xl_train.py",
            "instruct_pix2pix": "examples/multimodal/text_to_image/instruct_pix2pix/sd_finetune.py",
            "imagen": "examples/multimodal/text_to_image/imagen/imagen_training.py",
            "dreambooth": "examples/multimodal/text_to_image/dreambooth/dreambooth.py",
            "controlnet": "examples/multimodal/text_to_image/controlnet/controlnet_train.py",
            "nerf": "examples/multimodal/x_to_nerf/nerf/main.py",
            "neva": "examples/multimodal/multimodal_llm/neva/neva_pretrain.py",
        }
        # Based on Training.get_stage_config_choice
        stage_cfg = cfg.get(cls.stage_name)
        stage_config_choice = cfg.get(f"{cls.stage_name}_config")
        choice_model_type = stage_config_choice.rsplit("/", 1)[0]

        return cls(
            stage_cfg=stage_cfg,
            cluster_cfg=instantiate_model_from_omegaconf(cfg.cluster),
            image=cfg.container,
            env=cfg.env_vars,
            n_workers=stage_cfg.trainer.num_nodes,
            gpus_per_worker=stage_cfg.trainer.devices,
            nemo_train_script=model_type_to_code_path[choice_model_type],
            wandb_api_key=cfg.wandb_api_key_file,
        )

    def make_k8s_workflow(self) -> Workflow:
        assert isinstance(self.cluster_cfg, K8sClusterConfig)
        # First step is to resolve the config since there are absolute/relative references
        OmegaConf.resolve(self.stage_cfg)

        self.cluster_cfg.check_path_in_volumes(
            self.stage_cfg.exp_manager.explicit_log_dir
        )
        for data_item in self.stage_cfg.model.data.data_prefix:
            if not isinstance(data_item, str):
                continue
            self.cluster_cfg.check_path_in_volumes(data_item)

        vols, vol_mounts = adapt_volume_to(self.cluster_cfg.volumes, to_format="hera")

        with Workflow(
            generate_name="training-",
            entrypoint="training-steps",
            namespace=self.cluster_cfg.namespace,
            volumes=vols,
        ) as w:
            pytorchjob = create_pytorchjob_resource(
                generate_name="training-",
                image=self.image,
                image_pull_secret=self.cluster_cfg.pull_secret,
                n_workers=self.n_workers,
                gpus_per_worker=self.gpus_per_worker,
                namespace=self.cluster_cfg.namespace,
                env=self.env,
                command=[
                    "bash",
                    "-euxc",
                    f"""
mkdir -p /config
cat <<"EOF" | tee /config/config.yaml
{OmegaConf.to_yaml(self.stage_cfg)}
EOF
cd /opt/NeMo
nvidia-smi
export PYTHONPATH=/opt/NeMo:${{PYTHONPATH}}
{('wandb login ' + self.wandb_api_key) if self.wandb_api_key else ''}
torchrun {self.nemo_train_script} --config-path=/config --config-name=config.yaml
                    """,
                ],
                volumes=self.cluster_cfg.volumes,
                network_interfaces=self.cluster_cfg.ib_interfaces,
                capabilities=self.cluster_cfg.capabilities,
            )
            with Steps(name="training-steps") as s:
                Step(name="pytorchjob", template=pytorchjob)
        return w


class PEFT(Stage):
    data_dir: str
    image: str
    env: dict[str, Any]
    n_workers: int
    gpus_per_worker: int
    task_name: str = "squad"
    launcher_download_module: str = "nemo_launcher.utils.data_utils.prepare_squad"
    nemo_train_script: str = "examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py"
    wandb_api_key: Optional[str] = None
    stage_name: ClassVar[str] = "peft"

    @classmethod
    def _from_omegaconf(cls, cfg: OmegaConf) -> "PEFT":
        # TODO: Rewrite to use model for validation
        # This constructor is to bridge from the old way of specifying configs

        model_type_to_code_path = {
            "gpt3": "examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py",
            "llama": "examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py",
            "t5": "examples/nlp/language_modeling/tuning/megatron_t5_peft_tuning.py",
            "falcon": "examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py",
            "neva": "examples/multimodal/multimodal_llm/neva/neva_peft.py",
        }
        # Based on PEFT.get_stage_config_choice
        stage_cfg = cfg.get(cls.stage_name)
        stage_config_choice = cfg.get(f"{cls.stage_name}_config")
        choice_model_type = stage_config_choice.rsplit("/", 1)[0]
        if choice_model_type == "mt5":
            raise NotImplementedError(
                f"{cls.__name__} is not supported in NeMo Megatron mt5 models."
            )

        return cls(
            stage_cfg=stage_cfg,
            cluster_cfg=instantiate_model_from_omegaconf(cfg.cluster),
            data_dir=cfg.data_dir,
            image=cfg.container,
            env=cfg.env_vars,
            n_workers=stage_cfg.trainer.num_nodes,
            gpus_per_worker=stage_cfg.trainer.devices,
            task_name=stage_cfg.run.task_name,
            nemo_train_script=model_type_to_code_path[choice_model_type],
            wandb_api_key=cfg.wandb_api_key_file,
        )

    def make_k8s_workflow(self) -> Workflow:
        assert isinstance(self.cluster_cfg, K8sClusterConfig)
        # First step is to resolve the config since there are absolute/relative references
        OmegaConf.resolve(self.stage_cfg)

        self.cluster_cfg.check_path_in_volumes(self.data_dir)
        self.cluster_cfg.check_path_in_volumes(
            self.stage_cfg.exp_manager.explicit_log_dir
        )
        self.cluster_cfg.check_path_in_volumes(self.stage_cfg.model.restore_from_path)

        vols, vol_mounts = adapt_volume_to(self.cluster_cfg.volumes, to_format="hera")

        with Workflow(
            generate_name="peft-",
            entrypoint="peft-steps",
            namespace=self.cluster_cfg.namespace,
            volumes=vols,
        ) as w:
            # TODO: to be backward compatible with current stage_cfg, "squad_data" dir is coded
            # here since it's not parametrized
            squad_data_dir = os.path.join(self.data_dir, "squad_data")
            download_squad = Container(
                name="download-squad",
                image=self.image,
                command=[
                    "bash",
                    "-euxc",
                    dedent(
                        f"""
                    cd /opt/NeMo-Framework-Launcher/launcher_scripts
                    python -c '
                    from {self.launcher_download_module} import *
                    prepare_squad_for_fine_tuning(data_dir="{squad_data_dir}")
                    '
                    """
                    ),
                ],
                volume_mounts=vol_mounts,
                # NOTE: This does not use security contexts
            )
            pytorchjob = create_pytorchjob_resource(
                generate_name="peft-",
                image=self.image,
                image_pull_secret=self.cluster_cfg.pull_secret,
                n_workers=self.n_workers,
                gpus_per_worker=self.gpus_per_worker,
                namespace=self.cluster_cfg.namespace,
                env=self.env,
                command=[
                    "bash",
                    "-euxc",
                    f"""
mkdir -p /config
cat <<"EOF" | tee /config/config.yaml
{OmegaConf.to_yaml(self.stage_cfg)}
EOF
cd /opt/NeMo
nvidia-smi
export PYTHONPATH=/opt/NeMo:${{PYTHONPATH}}
{('wandb login ' + self.wandb_api_key) if self.wandb_api_key else ''}
torchrun {self.nemo_train_script} --config-path=/config --config-name=config.yaml
                    """,
                ],
                volumes=self.cluster_cfg.volumes,
                network_interfaces=self.cluster_cfg.ib_interfaces,
                capabilities=self.cluster_cfg.capabilities,
            )
            with Steps(name="peft-steps") as s:
                if self.task_name:
                    if self.task_name not in ("squad", "xquad"):
                        raise ValueError
                    Step(name="download-squad", template=download_squad)
                Step(name="pytorchjob", template=pytorchjob)
        return w


class PileDataPreparation(Stage):
    data_dir: str
    image: str
    env: dict[str, Any]
    n_workers: int
    n_proc_per_worker: int

    # Set scripts to empty/None to skip the step
    download_script: Optional[
        str
    ] = "/opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/dataprep_scripts/pile_dataprep/download.py"
    extract_script: Optional[
        str
    ] = "/opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/dataprep_scripts/pile_dataprep/extract.py"
    preprocess_script: Optional[
        str
    ] = "/opt/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/dataprep_scripts/pile_dataprep/preprocess.py"

    download_vocab_url: Optional[str] = None
    download_merges_url: Optional[str] = None
    vocab_save_dir: Optional[str] = None
    merges_save_dir: Optional[str] = None
    tokenizer_type: str = "GPT2BPETokenizer"
    tokenizer_library: str = "megatron"
    the_pile_url: str = "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/"  # Source URL to download The Pile dataset from.
    file_numbers: str = "0-29"  # The pile dataset consists of 30 files (0-29), choose which ones to download.
    rm_downloaded: bool = True  # Extract script will remove downloaded zst after extraction
    rm_extracted: bool = True  #  Preprocess script will remove extracted files after preproc.
    script_args: ClassVar[list[str]] = [
        "data_dir",
        "download_vocab_url",
        "download_merges_url",
        "vocab_save_dir",
        "merges_save_dir",
        "tokenizer_type",
        "tokenizer_library",
        "the_pile_url",
        "file_numbers",
        "rm_downloaded",
        "rm_extracted",
    ]
    stage_name: ClassVar[str] = "data_preparation"

    @computed_field
    @property
    def n_total_processes(self) -> int:
        return self.n_workers * self.n_proc_per_worker

    @classmethod
    def _from_omegaconf(
        cls: "PileDataPreparation", cfg: OmegaConf
    ) -> "PileDataPreparation":
        # TODO: Rewrite to use model for validation
        # This constructor is to bridge from the old way of specifying configs
        stage_cfg = cfg.get(cls.stage_name)

        return cls(
            stage_cfg=stage_cfg,
            cluster_cfg=instantiate_model_from_omegaconf(cfg.cluster),
            data_dir=cfg.data_dir,
            image=cfg.container,
            env=cfg.env_vars,
            n_workers=stage_cfg.run.node_array_size,
            # TODO: stage_cfg should depend on a parameter that is launcher independent
            n_proc_per_worker=stage_cfg.run.bcp_preproc_npernode,
            download_vocab_url=stage_cfg.download_vocab_url,
            download_merges_url=stage_cfg.download_merges_url,
            vocab_save_dir=stage_cfg.vocab_save_dir,
            merges_save_dir=stage_cfg.merges_save_dir,
            tokenizer_type=stage_cfg.tokenizer_type,
            tokenizer_library=stage_cfg.tokenizer_library,
            the_pile_url=stage_cfg.the_pile_url,
            file_numbers=stage_cfg.file_numbers,
            rm_downloaded=stage_cfg.rm_downloaded,
            rm_extracted=stage_cfg.rm_extracted,
        )

    def make_k8s_workflow(self) -> Workflow:
        assert isinstance(self.cluster_cfg, K8sClusterConfig)
        assert (
            "TRANSFORMERS_OFFLINE" not in self.env
            or str(self.env["TRANSFORMERS_OFFLINE"]) == "0"
        ), f"pile_dataprep/preprocess.py may fail if it cannot fetch configs from HF. Do not use HF in offline mode: {self.env}"
        # First step is to resolve the config since there are absolute/relative references
        OmegaConf.resolve(self.stage_cfg)

        self.cluster_cfg.check_path_in_volumes(self.data_dir)

        vols, vol_mounts = adapt_volume_to(self.cluster_cfg.volumes, to_format="hera")

        # This is a thin wrapper to avoid the `return` in download_single_file and to be hermetic
        def _download_single_file(
            url: str, save_dir: str, file_name: Optional[str] = None
        ):
            from nemo_launcher.utils.file_utils import download_single_file

            download_single_file(url, save_dir, file_name)

        download_single_file_script = script(
            name="download-tokenizer",
            volume_mounts=vol_mounts,
            image=self.image,
            image_pull_policy="Always",
            env=self.env,
        )(_download_single_file)

        with Workflow(
            generate_name="pile-prep-",
            entrypoint="data-steps",
            namespace=self.cluster_cfg.namespace,
            volumes=vols,
        ) as w:
            # ++overide all parameters to avoid having to create config file in launcher & worker containers
            hydra_config_as_args = [
                f"++{a}"
                for a in create_args_list(
                    hydra=True,
                    replace_underscore=False,
                    # Used by all scripts: download.py/extract.py/preprocess.py
                    cluster_type="k8s",
                    # Set to empty b/c pile_dataprep/preprocess.py assumes vocab is under the launcher, which is ignored
                    # on k8s and since vocab_save_dir and merges_save_dir are absolutes paths, this can be ignored
                    launcher_scripts_path="",
                    **{arg: getattr(self, arg) for arg in self.script_args},
                )
            ]

            mpirun_template = (
                lambda script_name: f'mpirun --allow-run-as-root -np { self.n_total_processes } -npernode { self.n_proc_per_worker } -bind-to none -map-by slot --oversubscribe -x PYTHONPATH -mca pml ob1 -mca btl ^openib python3 {script_name} {" ".join(hydra_config_as_args)}'
            )
            commands = []
            for script_path in (
                self.download_script,
                self.extract_script,
                self.preprocess_script,
            ):
                if not script_path:
                    continue
                commands.append(mpirun_template(script_path))
            assert commands
            commands_str = "\n".join(commands)

            data_resource = create_mpijob_resource(
                generate_name="pile-prep-",
                image=self.image,
                image_pull_secret=self.cluster_cfg.pull_secret,
                n_workers=self.n_workers,
                namespace=self.cluster_cfg.namespace,
                env=self.env,
                command=["bash", "-euxc", commands_str],
                volumes=self.cluster_cfg.volumes,
                network_interfaces=self.cluster_cfg.ib_interfaces,
                capabilities=self.cluster_cfg.capabilities,
            )
            with Steps(name="data-steps") as s:
                # First download any tokenizer files in standalone containers
                if self.download_merges_url:
                    download_single_file_script(
                        name="download-merges",
                        arguments={
                            "url": self.download_merges_url,
                            "save_dir": self.merges_save_dir,
                            "file_name": "merges.txt",
                        },
                    )

                if self.download_vocab_url:
                    download_single_file_script(
                        name="download-vocab",
                        arguments={
                            "url": self.download_vocab_url,
                            "save_dir": self.vocab_save_dir,
                            "file_name": (
                                "vocab.json"
                                if self.download_vocab_url.endswith("json")
                                else "vocab.txt"
                            ),
                        },
                    )

                # Then run preprocessing mpijob
                Step(name="mpijob", template=data_resource)
        return w


class RLHFPPO(Stage):
    image: str
    env: dict[str, Any]
    n_critic_workers: int
    n_critic_gpus_per_worker: Optional[int] = None
    n_actor_workers: int
    n_actor_gpus_per_worker: Optional[int] = None
    critic_port: int = 5567

    wandb_api_key: Optional[str] = None

    critic_script: str = "/opt/NeMo-Aligner/examples/nlp/gpt/serve_ppo_critic.py"
    actor_script: str = "/opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_ppo_actor.py"

    stage_name: ClassVar[str] = "rlhf_ppo"

    @classmethod
    def _from_omegaconf(cls: "RLHFPPO", cfg: OmegaConf) -> "RLHFPPO":
        # TODO: Rewrite to use model for validation
        # This constructor is to bridge from the old way of specifying configs
        stage_cfg = cfg.get(cls.stage_name)

        return cls(
            stage_cfg=stage_cfg,
            cluster_cfg=instantiate_model_from_omegaconf(cfg.cluster),
            image=cfg.container,
            env=cfg.env_vars,
            n_critic_workers=stage_cfg.critic.trainer.num_nodes,
            n_critic_gpus_per_worker=stage_cfg.critic.trainer.devices,
            n_actor_workers=stage_cfg.actor.trainer.num_nodes,
            n_actor_gpus_per_worker=stage_cfg.actor.trainer.devices,
            wandb_api_key=cfg.wandb_api_key_file,
        )

    def make_k8s_workflow(self) -> Workflow:
        assert isinstance(self.cluster_cfg, K8sClusterConfig)
        # First step is to resolve the config since there are absolute/relative references
        OmegaConf.resolve(self.stage_cfg)

        self.cluster_cfg.check_path_in_volumes(
            self.stage_cfg.critic.exp_manager.explicit_log_dir
        )
        self.cluster_cfg.check_path_in_volumes(
            self.stage_cfg.critic.pretrained_checkpoint.restore_from_path
        )
        self.cluster_cfg.check_path_in_volumes(
            self.stage_cfg.actor.exp_manager.explicit_log_dir
        )
        self.cluster_cfg.check_path_in_volumes(
            self.stage_cfg.actor.pretrained_checkpoint.restore_from_path
        )
        for shards in self.stage_cfg.actor.model.data.data_prefix.values():
            for shard in shards:
                if not isinstance(shard, str):
                    continue
                self.cluster_cfg.check_path_in_volumes(shard)

        vols, vol_mounts = adapt_volume_to(self.cluster_cfg.volumes, to_format="hera")

        with Workflow(
            generate_name="rlhf-ppo-",
            entrypoint="rlhf-ppo-steps",
            namespace=self.cluster_cfg.namespace,
            volumes=vols,
        ) as w:
            critic_job = create_pytorchjob_resource(
                generate_name="critic-",
                image=self.image,
                image_pull_secret=self.cluster_cfg.pull_secret,
                n_workers=self.n_critic_workers,
                gpus_per_worker=self.n_critic_gpus_per_worker,
                namespace=self.cluster_cfg.namespace,
                env=self.env,
                command=[
                    "bash",
                    "-euxc",
                    f"""
mkdir -p /config
cat <<"EOF" | tee /config/config.yaml
{OmegaConf.to_yaml(self.stage_cfg["critic"])}
EOF
nvidia-smi
{('wandb login ' + self.wandb_api_key) if self.wandb_api_key else ''}
torchrun {self.critic_script} --config-path=/config --config-name=config.yaml \
    trainer.ppo.port={self.critic_port}
                    """,
                ],
                volumes=self.cluster_cfg.volumes,
                network_interfaces=self.cluster_cfg.ib_interfaces,
                # Set success_condition to None is equivalent to not waiting
                success_condition=None,
                capabilities=self.cluster_cfg.capabilities,
            )
            actor_job = create_pytorchjob_resource(
                generate_name="actor-",
                image=self.image,
                image_pull_secret=self.cluster_cfg.pull_secret,
                n_workers=self.n_actor_workers,
                gpus_per_worker=self.n_actor_gpus_per_worker,
                namespace=self.cluster_cfg.namespace,
                env=self.env,
                command=[
                    "bash",
                    "-euxc",
                    f"""
mkdir -p /config
cat <<"EOF" | tee /config/config.yaml
{OmegaConf.to_yaml(self.stage_cfg["actor"])}
EOF
nvidia-smi
{('wandb login ' + self.wandb_api_key) if self.wandb_api_key else ''}
torchrun {self.actor_script} --config-path=/config --config-name=config.yaml \
    remote_critic_rm.critic.ip={{{{inputs.parameters.critic_job_name}}}}-worker-0.{{{{inputs.parameters.critic_job_namespace}}}}.svc.cluster.local \
    remote_critic_rm.critic.port={self.critic_port}
                    """,
                ],
                volumes=self.cluster_cfg.volumes,
                network_interfaces=self.cluster_cfg.ib_interfaces,
                resource_inputs=[
                    Parameter(name="critic_job_name"),
                    Parameter(name="critic_job_namespace"),
                ],
                capabilities=self.cluster_cfg.capabilities,
            )
            # Critic will hang around so it should be deleted
            delete_critic = delete_pytorchjob()

            with DAG(name="rlhf-ppo-steps"):
                critic_task = critic_job()
                actor_task = actor_job(
                    arguments=[
                        critic_task.get_parameter("metadata_name").with_name(
                            "critic_job_name"
                        ),
                        # Namespace is also required b/c actor uses critic's Pod DNS which includes namespace
                        critic_task.get_parameter("metadata_namespace").with_name(
                            "critic_job_namespace"
                        ),
                    ]
                )
                critic_task >> actor_task
                delete_critic(
                    arguments=[critic_task.get_parameter("metadata_name")],
                    depends=f"{actor_task.name} || {actor_task.name}.Failed || {actor_task.name}.Errored",
                )
        return w


class RLHFRewardModel(Stage):
    image: str
    env: dict[str, Any]
    n_workers: int
    gpus_per_worker: int
    nemo_train_script: str = "/opt/NeMo-Aligner/examples/nlp/gpt/train_reward_model.py"
    wandb_api_key: Optional[str] = None
    stage_name: ClassVar[str] = "rlhf_rm"

    @classmethod
    def _from_omegaconf(cls: "RLHFRewardModel", cfg: OmegaConf) -> "RLHFRewardModel":
        # TODO: Rewrite to use model for validation
        # This constructor is to bridge from the old way of specifying configs

        model_type_to_code_path = {
            "gpt3": "/opt/NeMo-Aligner/examples/nlp/gpt/train_reward_model.py",
        }
        # Based on Training.get_stage_config_choice
        stage_cfg = cfg.get(cls.stage_name)
        stage_config_choice = cfg.get(f"{cls.stage_name}_config")
        choice_model_type = stage_config_choice.rsplit("/", 1)[0]

        return cls(
            stage_cfg=stage_cfg,
            cluster_cfg=instantiate_model_from_omegaconf(cfg.cluster),
            image=cfg.container,
            env=cfg.env_vars,
            n_workers=stage_cfg.trainer.num_nodes,
            gpus_per_worker=stage_cfg.trainer.devices,
            nemo_train_script=model_type_to_code_path[choice_model_type],
            wandb_api_key=cfg.wandb_api_key_file,
        )

    def make_k8s_workflow(self) -> Workflow:
        assert isinstance(self.cluster_cfg, K8sClusterConfig)
        # First step is to resolve the config since there are absolute/relative references
        OmegaConf.resolve(self.stage_cfg)

        self.cluster_cfg.check_path_in_volumes(
            self.stage_cfg.exp_manager.explicit_log_dir
        )
        for shards in self.stage_cfg.model.data.data_prefix.values():
            for shard in shards:
                self.cluster_cfg.check_path_in_volumes(shard)

        vols, vol_mounts = adapt_volume_to(self.cluster_cfg.volumes, to_format="hera")

        with Workflow(
            generate_name="rlhf-rm-",
            entrypoint="rlhf-rm-steps",
            namespace=self.cluster_cfg.namespace,
            volumes=vols,
        ) as w:
            pytorchjob = create_pytorchjob_resource(
                generate_name="rlhf-rm-",
                image=self.image,
                image_pull_secret=self.cluster_cfg.pull_secret,
                n_workers=self.n_workers,
                gpus_per_worker=self.gpus_per_worker,
                namespace=self.cluster_cfg.namespace,
                env=self.env,
                command=[
                    "bash",
                    "-euxc",
                    f"""
mkdir -p /config
cat <<"EOF" | tee /config/config.yaml
{OmegaConf.to_yaml(self.stage_cfg)}
EOF
nvidia-smi
{('wandb login ' + self.wandb_api_key) if self.wandb_api_key else ''}
torchrun {self.nemo_train_script} --config-path=/config --config-name=config.yaml
                    """,
                ],
                volumes=self.cluster_cfg.volumes,
                network_interfaces=self.cluster_cfg.ib_interfaces,
                capabilities=self.cluster_cfg.capabilities,
            )
            with Steps(name="rlhf-rm-steps") as s:
                Step(name="pytorchjob", template=pytorchjob)
        return w
