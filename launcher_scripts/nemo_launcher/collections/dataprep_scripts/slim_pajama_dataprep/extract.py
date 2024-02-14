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

import nemo_launcher.utils.file_utils as utils


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
    data_dir: str, shards_to_extract: list, w_rank: int, rm_downloaded: bool = False
) -> int:
    shards_extracted = 0
    print(f"Task :{w_rank} is extracting shards {shards_to_extract}")
    for shard in shards_to_extract[w_rank]:
        file_path = os.path.join(data_dir, shard)
        utils.extract_single_zst_file(file_path, data_dir, shard[:-4], rm_downloaded)
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
    num_tasks = int(os.environ["SLURM_STEP_NUM_TASKS"])
    array_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    rank = int(os.environ["RANK"])
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    w_rank = rank + (task_id * num_tasks)
    w_size = num_tasks * array_count

    shards_to_extract = get_shard_list(data_dir, w_size)
    shards_extracted = run_extraction(
        data_dir, shards_to_extract, w_rank, rm_downloaded
    )
    print(f"Extracted {shards_extracted} shards out of {len(shards_to_extract)}")


if __name__ == "__main__":
    main()
