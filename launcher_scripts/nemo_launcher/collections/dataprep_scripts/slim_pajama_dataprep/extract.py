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

"""
Slim Pajama data downloading.
Example usage:
    python main.py
"""

from glob import glob
import os

import hydra
import json

import nemo_launcher.utils.file_utils as utils
from nemo_launcher.core.logger import logger


SOURCES_LIST = [
    "RedPajamaCommonCrawl",
    "RedPajamaC4",
    "RedPajamaGithub",
    "RedPajamaBook",
    "RedPajamaArXiv",
    "RedPajamaWikipedia",
    "RedPajamaStackExchange",
]


def approve_source(filename: str, source_list: list):
    """
    Function to remove data from non approved sources.
    Books data is removed by default due to copyright issues

    Arguments:
        filename: path to jsonl file with the data
        source_list: list of sources that are allowed to be included in the dataset
    """

    with open(filename, "r") as i:
        with open(filename + ".tmp", "w") as o:
            for line in i.read().splitlines():
                j = json.loads(line)
                if j["meta"]["redpajama_set_name"] in source_list:
                    json.dump(j, o)
                    o.write("\n")
    os.rename(filename + ".tmp", filename)
    return


def split_shards(dataset: list, w_size: int) -> list:
    shards = []
    for shard in range(w_size):
        idx_start = (shard * len(dataset)) // w_size
        idx_end = ((shard + 1) * len(dataset)) // w_size
        shards.append(dataset[idx_start:idx_end])
    return shards


def get_shard_list(data_dir: str, w_size: int) -> list:
    dataset = glob("*example_train*zst", root_dir=data_dir)
    return split_shards(dataset, w_size)


def run_extraction(
    data_dir: str,
    shards_to_extract: list,
    w_rank: int,
    approved_sources: list,
    rm_downloaded: bool = False,
) -> int:
    shards_extracted = 0
    source_list = []
    for source in approved_sources:
        if source in SOURCES_LIST:
            source_list.append(source)
        else:
            logger.warning(
                f"Source: {source} is not recognized, set the approved_sources flag in launcher_scripts/conf/data_preparation/gpt/download_slim_pajama.yaml"
            )

    logger.info(f"Task :{w_rank} is extracting shards {shards_to_extract}")
    for shard in shards_to_extract[w_rank]:
        file_path = os.path.join(data_dir, shard)
        utils.extract_single_zst_file(file_path, data_dir, shard[:-4], rm_downloaded)
        shard_path = os.path.join(data_dir, shard[:-4])
        approve_source(shard_path, source_list)
        shards_extracted += 1
    return shards_extracted


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg):
    """
    Function to download the pile dataset files on Slurm.

    Arguments:
        cfg: main config file.
    conf variables being used:
        data_dir
    """
    data_dir = cfg.get("data_dir")
    rm_downloaded = cfg.get("rm_downloaded")
    approved_sources = cfg.get("approved_sources")
    num_tasks = int(os.environ["SLURM_STEP_NUM_TASKS"])
    array_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    rank = int(os.environ["RANK"])
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    w_rank = rank + (task_id * num_tasks)
    w_size = num_tasks * array_count

    shards_to_extract = get_shard_list(data_dir, w_size)
    shards_extracted = run_extraction(
        data_dir, shards_to_extract, w_rank, approved_sources, rm_downloaded
    )
    logger.info(f"Extracted {shards_extracted} shards out of {len(shards_to_extract)}")


if __name__ == "__main__":
    main()
