#!/bin/bash

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

shards_per_file=800

data_dir=$1
rm_extracted=$2
WSIZE=$((SLURM_STEP_NUM_TASKS * SLURM_ARRAY_TASK_COUNT))
WRANK=$((RANK + (SLURM_ARRAY_TASK_ID * SLURM_STEP_NUM_TASKS)))

num_files=`find $data_dir -maxdepth 1 -type f -name "example_train_chunk*.jsonl" | wc -l`
files=($data_dir/example_train_chunk*.jsonl)

# Find the ceiling of the result
shards=$(((num_files+shards_per_file-1)/shards_per_file ))
if [[ $WRANK -ge $shards ]]; then
  echo "More tasks than shards. Shutting down extra task ${WRANK}."
  exit 0
fi

for ((i=0; i<$shards; i++)); do
  file_start=$((i*shards_per_file))

  if [[ $(((i+1)*shards_per_file)) -ge ${#files[@]} ]]; then
    file_stop=$((${#files[@]}-1))
  else
    file_stop=$(((i+1)*shards_per_file))
  fi
  if [[ $(($i % $WSIZE)) -eq $WRANK ]]; then
    echo "Task $WRANK is building chunk $i with files $file_start to $file_stop"
    if [[ ! -f $data_dir/train_chunk_${i}.jsonl ]]; then
      cat ${files[@]:$file_start:$shards_per_file} > $data_dir/train_chunk_${i}.jsonl
      echo "Task $WRANK finished building chunk $i"
    else
      echo "Chunk $i already exists. Skipping"
    fi
    
  fi
done
