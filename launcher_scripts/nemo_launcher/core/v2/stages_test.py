from nemo_launcher.core.v2 import stages
from nemo_launcher.core.v2 import config_k8s
from omegaconf import OmegaConf
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
import pytest
import math

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
launcher_root = os.path.join(dir_path, "..", "..", "..")


def make_cfg(overrides: list[str]):
    with initialize(config_path="../../../conf", version_base="1.2"):
        return compose(
            config_name="config", overrides=overrides, return_hydra_config=True
        )


# TODO: Resolvers are repeated here b/c main.py is outside of the nemo_launcher package
if not OmegaConf.has_resolver("multiply"):
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
if not OmegaConf.has_resolver("divide_ceil"):
    OmegaConf.register_new_resolver(
        "divide_ceil", lambda x, y: int(math.ceil(x / y)), replace=True
    )
if not OmegaConf.has_resolver("divide_floor"):
    OmegaConf.register_new_resolver(
        "divide_floor", lambda x, y: int(math.floor(x / y)), replace=True
    )


@pytest.mark.parametrize(
    "stage_cls,overrides",
    [
        (
            stages.Training,
            [
                "stages=[training]",
                "training=gpt3/1b_improved",
                "training.exp_manager.explicit_log_dir=/nemo/results",
                "training.model.data.data_prefix=[0.5,/nemo/doc_1,0.5,/nemo/doc_2]",
            ],
        ),
        (
            stages.PEFT,
            [
                "stages=[peft]",
                "peft=llama/squad",
                "data_dir=/nemo/data",
                "peft.exp_manager.explicit_log_dir=/nemo/results",
                "peft.model.restore_from_path=/nemo/llama.nemo",
            ],
        ),
        (
            stages.PileDataPreparation,
            [
                "stages=[data_preparation]",
                "data_preparation=gpt3/download_gpt3_pile",
                "env_vars.TRANSFORMERS_OFFLINE=0",
                "data_dir=/nemo/pile",
            ],
        ),
        (
            stages.RLHFPPO,
            [
                "stages=[rlhf_ppo]",
                "rlhf_ppo=gpt3/2b_ppo",
                "rlhf_ppo.critic.pretrained_checkpoint.restore_from_path=/nemo/megatron_gpt.nemo",
                "rlhf_ppo.critic.exp_manager.explicit_log_dir=/nemo/critic_results",
                "rlhf_ppo.actor.pretrained_checkpoint.restore_from_path=/nemo/megatron_gpt.nemo",
                "rlhf_ppo.actor.exp_manager.explicit_log_dir=/nemo/actor_results",
                "rlhf_ppo.actor.model.data.data_prefix={train: [/nemo/train_prompt], validation: [/nemo/val_prompt], test: [/nemo/val_prompt]}",
            ],
        ),
        (
            stages.RLHFRewardModel,
            [
                "stages=[rlhf_rm]",
                "rlhf_rm=gpt3/2b_rm",
                "rlhf_rm.exp_manager.explicit_log_dir=/nemo/results",
                "rlhf_rm.model.data.data_prefix={train: [/nemo/train_comparisons.jsonl], validation: [/nemo/test_comparisons.jsonl], test: [/nemo/test_comparisons.jsonl]}",
            ],
        ),
    ],
)
def test_k8s_stage_functional(stage_cls, overrides):
    common_overrides = [
        f"launcher_scripts_path={launcher_root}",
        "cluster=k8s_v2",
        "+cluster.volumes.workspace.mount_path=/nemo",
    ]
    cfg = make_cfg(overrides=common_overrides + overrides)
    # Note: This is needed if using compose API and not hydra.main b/c we rely on hydra resolver
    # Open issue tracking fix https://github.com/facebookresearch/hydra/issues/2017
    HydraConfig.instance().set_config(cfg)
    stage = stage_cls._from_omegaconf(cfg)

    w = stage.make_k8s_workflow()
