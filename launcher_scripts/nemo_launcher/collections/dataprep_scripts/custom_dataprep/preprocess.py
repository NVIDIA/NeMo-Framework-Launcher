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

"""
Multi-worker data preprocessing.
Example usage:
 python preprocess.py \
    --worker-mapping-file=<path/to/preprocess_mapping_file> \
    --output-path=<output/path> \
    --tokenizer-library <some_tokenizer_lib> \
    --tokenizer-model <some_tokenizer_model> \
    --dataset-impl mmap \
    --workers 80  \
    --apply-ftfy
"""

import argparse
import os
import subprocess
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess custom dataset", allow_abbrev=False)

    parser.add_argument("--output-path", help="Path to store output bin files", required=True)
    parser.add_argument("--worker-mapping-file", help="Decide which worker download which languages", required=True)
    parser.add_argument(
        "--workers-per-node",
        default=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)),
        help="Number of workers per node in preprocessing step",
        type=int,
    )
    parser.add_argument("--bcp", action="store_true", help="Whether on BCP platform")
    args, other_args = parser.parse_known_args()

    workers_per_node = args.workers_per_node  # local world size
    if args.bcp:
        global_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        task_id = global_rank // workers_per_node
        rank = global_rank % workers_per_node
    else:  # on slurm based platforms
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        rank = int(os.environ.get("LOCAL_RANK", 0))

    with open(args.worker_mapping_file) as f:
        mapping = f.readlines()
    data_files = []
    if task_id * workers_per_node + rank < len(mapping):
        data_files = mapping[task_id * workers_per_node + rank].strip().split(",")
    print(f" ****** Task ID {task_id:02d} Rank {rank:02d} is preparing to preprocess {data_files}...")

    os.makedirs(args.output_path, exist_ok=True)
    start_time = time.time()
    cmd = [
        "python",
        "/opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py",
    ]
    for split in data_files:
        if not split:  # Remove empty split
            continue
        print(f" ****** Task ID {task_id:02d} Rank {rank:02d} starts to preprocess {os.path.basename(split)}...")
        input_arg = ["--input", split]
        output_arg = ["--output-prefix", os.path.join(args.output_path, os.path.basename(split))]
        subprocess.check_call(cmd + input_arg + output_arg + other_args)
        print(f" ****** Task ID {task_id:02d} Rank {rank:02d} finished preprocessing {os.path.basename(split)}...")
        print(f" ****** Task ID {task_id:02d} Rank {rank:02d} time elapsed {(time.time() - start_time) / 60:.2f} min.")
