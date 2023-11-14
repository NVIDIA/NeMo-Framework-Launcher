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
from nemo_launcher.core.data_curation_stages import QualityFiltering
from nemo_launcher.core.data_stages import (
    CustomDataPreparation,
    MC4DataPreparation,
    PileDataPreparation,
)
from nemo_launcher.core.export_stages import Export
from nemo_launcher.core.rlhf_stages import RLHFPPO, RLHFRewardModel
from nemo_launcher.core.stages import (
    PEFT,
    AdapterLearning,
    Conversion,
    EvalHarnessEvaluation,
    FineTuning,
    IA3Learning,
    NeMoEvaluation,
    PromptLearning,
    Training,
)

omegaconf.OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
omegaconf.OmegaConf.register_new_resolver(
    "divide_ceil", lambda x, y: int(math.ceil(x / y)), replace=True
)
omegaconf.OmegaConf.register_new_resolver(
    "divide_floor", lambda x, y: int(math.floor(x / y)), replace=True
)

STR2STAGECLASS = {
    "training": Training,
    "fine_tuning": FineTuning,
    "peft": PEFT,
    "prompt_learning": PromptLearning,
    "adapter_learning": AdapterLearning,
    "ia3_learning": IA3Learning,
    "conversion": Conversion,
    "export": Export,
    "evaluation": {
        EvalHarnessEvaluation: ["gpt3", "prompt_gpt3", "llama", "prompt_llama"],
        NeMoEvaluation: [
            "t5",
            "mt5",
            "prompt_t5",
            "prompt_mt5",
            "adapter_t5",
            "adapter_gpt3",
            "ia3_t5",
            "ia3_gpt3",
            "peft_llama",
        ],
    },
    "data_preparation": {
        PileDataPreparation: ["gpt3", "t5", "bert", "llama"],
        MC4DataPreparation: ["mt5"],
        CustomDataPreparation: ["generic"],
    },
    "rlhf_rm": RLHFRewardModel,
    "rlhf_ppo": RLHFPPO,
    "quality_filtering": QualityFiltering,
}


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    requested_stages = cfg.get("stages")

    dependency = None
    for stage_name in requested_stages:
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
        stage = stage_class(cfg)
        job_id = stage.run()

        job_path = stage.get_job_path()
        command = " \\\n  ".join(sys.argv)
        with open(job_path.folder / "launcher_cmd.log", "w") as f:
            f.write(command)

        if job_id:
            dependency = f"afterany:{job_id}"


if __name__ == "__main__":
    main()
