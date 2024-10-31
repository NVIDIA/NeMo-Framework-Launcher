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

import math
import sys

import hydra
import omegaconf
from nemo_launcher.core.data_curation_stages import DataCurationStage
from nemo_launcher.core.data_stages import (
    CustomDataPreparation,
    FIDEvaluationDataPreparation,
    HumanEvalDataPreparation,
    MC4DataPreparation,
    MultimodalDataPreparation,
    PileDataPreparation,
    SlimPajamaDataPreparation,
    SteerLMDataPreparation,
)
from nemo_launcher.core.export_stages import Export
from nemo_launcher.core.rlhf_stages import RLHFPPO, RLHFRewardModel
from nemo_launcher.core.stages import (
    PEFT,
    AdapterLearning,
    Conversion,
    DiffusionModelEvaluation,
    EvalHarnessEvaluation,
    ExternalConversion,
    FineTuning,
    FWInference,
    IA3Learning,
    NeMoEvaluation,
    PromptLearning,
    Training,
    SteerLMRegSFT,
    ConversionHF2NeMo,
    PostTrainingQuantization,
    RAGIndexing,
    RAGGenerating,
)
from nemo_launcher.core.v2 import stages as stages_v2
from nemo_launcher.core.v2.config_k8s import K8sClusterConfig

omegaconf.OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
omegaconf.OmegaConf.register_new_resolver(
    "divide_ceil", lambda x, y: int(math.ceil(x / y)), replace=True
)
omegaconf.OmegaConf.register_new_resolver(
    "divide_floor", lambda x, y: int(math.floor(x / y)), replace=True
)

STR2STAGECLASSV2 = {
    "peft": stages_v2.PEFT,
    "data_preparation": {
        stages_v2.PileDataPreparation: ["gpt3", "t5", "bert", "llama", "falcon"],
    },
    "training": stages_v2.Training,
    "rlhf_ppo": stages_v2.RLHFPPO,
    "rlhf_rm": stages_v2.RLHFRewardModel,
}
STR2STAGECLASS = {
    "training": Training,
    "fine_tuning": FineTuning,
    "peft": PEFT,
    "prompt_learning": PromptLearning,
    "adapter_learning": AdapterLearning,
    "ia3_learning": IA3Learning,
    "conversion": Conversion,
    "conversion_hf2nemo": ConversionHF2NeMo,
    "external_conversion": ExternalConversion,
    "export": Export,
    "fw_inference": FWInference,
    "evaluation": {
        EvalHarnessEvaluation: [
            "gpt3",
            "prompt_gpt3",
            "llama",
            "prompt_llama",
            "falcon",
            "baichuan2",
            "chatglm",
            "mistral",
            "mixtral",
            "qwen2",
        ],
        NeMoEvaluation: [
            "t5",
            "mt5",
            "retro",
            "prompt_t5",
            "prompt_mt5",
            "adapter_t5",
            "adapter_gpt3",
            "ia3_t5",
            "ia3_gpt3",
            "peft_llama",
            "code_llama",
            "peft_falcon",
            "vit",
            "clip",
            "peft_baichuan2",
            "peft_chatglm",
            "starcoder2",
            "peft_mistral",
            "peft_mixtral",
            "peft_qwen2",
            "peft_t5",
        ],
        DiffusionModelEvaluation: ["stable_diffusion", "imagen"],
    },
    "data_preparation": {
        SlimPajamaDataPreparation: ["gpt"],
        PileDataPreparation: [
            "gpt3",
            "t5",
            "bert",
            "llama",
            "falcon",
            "baichuan2",
            "chatglm",
            "qwen2",
            "mistral",
            "mixtral",
        ],
        MC4DataPreparation: ["mt5"],
        SteerLMDataPreparation: ["steerlm"],
        CustomDataPreparation: ["generic"],
        MultimodalDataPreparation: ["multimodal"],
        FIDEvaluationDataPreparation: ["fid_evaluation"],
        HumanEvalDataPreparation: ["code_llama"],
    },
    "rlhf_rm": RLHFRewardModel,
    "rlhf_ppo": RLHFPPO,
    "data_curation": DataCurationStage,
    "steerlm_reg": SteerLMRegSFT,
    "ptq": PostTrainingQuantization,
    "rag_indexing": RAGIndexing,
    "rag_generating": RAGGenerating,
}


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: omegaconf.DictConfig):
    requested_stages = cfg.get("stages")

    dependency = None
    is_k8s_v2 = "_target_" in cfg.cluster and hydra.utils.get_class(
        cfg.cluster._target_
    ) in (K8sClusterConfig,)
    for stage_name in requested_stages:
        # TODO: User needs to specifically request cluster type k8s_v2 and the stage needs to be supported
        if is_k8s_v2:
            if stage_name not in STR2STAGECLASSV2:
                raise ValueError(
                    f"Using cluster=k8s_v2, but stage '{stage_name}' is not supported yet"
                )
            stage_class = STR2STAGECLASSV2[stage_name]
        else:
            stage_class = STR2STAGECLASS[stage_name]

        if isinstance(stage_class, dict):
            stage_config_choice = cfg.get(f"{stage_name}_config")
            choice_model_type = stage_config_choice.rsplit("/", 1)[0]

            for cls, model_types in stage_class.items():
                if choice_model_type in model_types:
                    stage_class = cls
                    break

        if dependency is not None:
            cfg[stage_name]["run"]["dependency"] = dependency

        # TODO: User needs to specifically request cluster type k8s_v2 and the stage needs to be supported
        if is_k8s_v2:
            if stage_name not in STR2STAGECLASSV2:
                raise ValueError(
                    f"Using cluster=k8s_v2, but stage '{stage_name}' is not supported yet"
                )
            stage = stage_class._from_omegaconf(cfg)
            job_path = stage.job_path
        else:
            stage = stage_class(cfg)
            job_path = stage.get_job_path()
        job_id = stage.run()

        command = " \\\n  ".join(sys.argv)

        with open(job_path.folder / "launcher_cmd.log", "w") as f:
            f.write(command)

        if job_id:
            dependency = f"afterany:{job_id}"


if __name__ == "__main__":
    main()
