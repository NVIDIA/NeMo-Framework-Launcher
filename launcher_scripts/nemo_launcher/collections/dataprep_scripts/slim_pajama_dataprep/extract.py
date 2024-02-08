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

# Global Variables and Environment Variables
WSIZE = int(os.environ['SLURM_STEP_NUM_TASKS']) * int(os.environ['SLURM_ARRAY_TASK_COUNT'])
WRANK = int(os.environ['RANK']) + ( int(os.environ['SLURM_ARRAY_TASK_ID']) * int(os.environ['SLURM_STEP_NUM_TASKS']))


def split_shards(dataset: list) -> list:
    shards = []
    for shard in range(WSIZE):
        idx_start = (shard * len(dataset)) // WSIZE
        idx_end = ((shard + 1) * len(dataset)) // WSIZE
        shards.append(dataset[idx_start:idx_end])
    return shards


def get_shard_list(data_dir: str) -> list:
    dataset = glob('*example_train*zst', root_dir=data_dir)
    return split_shards(dataset)


def run_extraction(data_dir: str, shards_to_extract: list, rm_downloaded: bool=False) -> int:
    shards_extracted = 0
    print(f'Task :{WRANK} is extracting shards {shards_to_extract}')
    for shard in shards_to_extract[WRANK]:
        file_path = os.path.join(data_dir,shard)
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
    data_dir = cfg.get('data_dir')
    rm_downloaded = cfg.get('rm_downloaded')
    shards_to_extract = get_shard_list(data_dir)
    shards_extracted = run_extraction(data_dir, shards_to_extract, rm_downloaded)
    print(f'Extracted {shards_extracted} shards out of {len(shards_to_extract)}')

if __name__ == "__main__":
    main()