# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Entry point, main file to run to launch fine-tuning autoconfigurator jobs."""

import hydra
import omegaconf
from src.search import run_search


@hydra.main(config_path="conf", config_name="config")
def main(cfg: omegaconf.dictconfig.DictConfig) -> None:
    """
    Entry point for the fine-tuning autoconfigurator pipeline. Reads the config using
      hydra, runs fine-tuning hyperparemeter search.
    :param omegaconf.dictconfig.DictConfig cfg: OmegaConf object, read using
      the @hydra.main decorator.
    :return: None
    """
    run_search(cfg=cfg)


if __name__ == "__main__":
    main()
