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
"""

import os

import hydra

import nemo_launcher.utils.file_utils as utils

# Global Variables and Environment Variables
WSIZE = int(os.environ['SLURM_STEP_NUM_TASKS']) * int(os.environ['SLURM_ARRAY_TASK_COUNT'])
WRANK = int(os.environ['RANK']) + ( int(os.environ['SLURM_ARRAY_TASK_ID']) * int(os.environ['SLURM_STEP_NUM_TASKS']))
CHUNKS = 10
SHARDS = 6000


def find_missing_files( data_dir: str, shards_to_download: list) -> list:
    """
    Function to check if all files are downloaded. 
    Commit diffs found here: 
    https://huggingface.co/datasets/cerebras/SlimPajama-627B/tree/main/train
    chunk 1: train/chunk1/example_train_0.jsonl.zst  0-5911
    chunk 2: train/chunk2/example_train_5910.jsonl.zst 0-5910
    chunk 3: train/chunk3/example_train_5918.jsonl.zst 0-5918
    chunk 4: train/chunk4/example_train_5916.jsonl.zst 0-5916
    chunk 5: train/chunk5/example_train_5932.jsonl.zst 0-5932
    chunk 6: train/chunk6/example_train_5914.jsonl.zst 0-5914
    chunk 7: train/chunk7/example_train_5905.jsonl.zst 0-5905
    chunk 8: train/chunk8/example_train_5920.jsonl.zst 0-5920
    chunk 9: train/chunk9/example_train_5919.jsonl.zst 0-5919
    chunk 10: train/chunk10/example_train_5911.jsonl.zst 0-5911
    total: 59166
    """

    missing_shards = []
    chunk_sizes = [
        5911,
        5910,
        5918,
        5916,
        5932,
        5914,
        5905,
        5920,
        5919,
        5911
    ]
    for chunk, chunk_size in enumerate(chunk_sizes):
        for shard in shards_to_download[WRANK]:
            if shard > chunk_size:
                break
            filename = f'example_train_chunk{chunk+1}_shard{shard}.jsonl.zst'
            if not os.path.exists(data_dir + '/' + filename):
                missing_shards.append((chunk, shard))
    return missing_shards


def split_shards() -> list:
    """
    Function to split up the 60000 downloads evenly across all tasks on all nodes.
    """
    shards = []
    shards_to_download = list(range(SHARDS))

    for shard in range(WSIZE):
        idx_start = (shard * SHARDS) // WSIZE
        idx_end = ((shard + 1) * SHARDS) // WSIZE
        shards.append(shards_to_download[idx_start:idx_end])
    return shards

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg) -> None:
    """
    Function to download the slim pajama dataset files on Slurm.

    Arguments:
        cfg: main config file.
    conf variables being used:
        data_dir
        slim_pajama_url
    """
    data_dir = cfg.get('data_dir')
    slim_pajama_url = cfg.get('slim_pajama_url')
    assert data_dir is not None, 'DATA_DIR must be a valid path.'

    shards_to_download = split_shards()

    for chunk in range(1, CHUNKS + 1):
        print(f'Task :{WRANK} is downloading shards {shards_to_download} in chunk {chunk}')
        for shard in shards_to_download[WRANK]:
            filename = f'example_train_chunk{chunk}_shard{shard}.jsonl.zst'
            url = f'{slim_pajama_url}/chunk{chunk}/example_train_{shard}.jsonl.zst'
            save_path = utils.download_single_file(url, data_dir, filename)
            if os.path.getsize(save_path) < 100: # If an empty file gets downloaded, delete it. Empty files are roughly 15 bytes long
                os.remove(save_path)

    print('Checking for missing shards')
    missing_shards = find_missing_files(data_dir, shards_to_download)
    if missing_shards:
        print(f'Attempting to redownload missing shards: {missing_shards}')
    for chunk, shard in missing_shards:
        filename = f'example_train_chunk{chunk}_shard{shard}.jsonl.zst'
        url = f'{slim_pajama_url}/chunk{chunk}/example_train_{shard}.jsonl.zst'
        save_path = utils.download_single_file(url, data_dir, filename)
        if os.path.getsize(save_path) < 100:
            os.remove(save_path)


if __name__ == "__main__":
    main()