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
Dolly data preprocessing.
Example usage:
python preprocess.py --input=<path/to/data/file>
"""

import json
from argparse import ArgumentParser

import numpy as np


def to_jsonl(path_to_data):
    print(f"Preprocessing data to jsonl format...")
    output_path = f"{path_to_data.split('.')[0]}-output.jsonl"
    with open(path_to_data, "r") as f, open(output_path, "w") as g:
        for line in f:
            line = json.loads(line)
            context = line["context"].strip()
            if context != "":
                # Randomize context and instruction order.
                context_first = np.random.randint(0, 2) == 0
                if context_first:
                    instruction = line["instruction"].strip()
                    assert instruction != ""
                    input = f"{context}\n\n{instruction}"
                    output = line["response"]
                else:
                    instruction = line["instruction"].strip()
                    assert instruction != ""
                    input = f"{instruction}\n\n{context}"
                    output = line["response"]
            else:
                input = line["instruction"]
                output = line["response"]
            g.write(
                json.dumps(
                    {"input": input, "output": output, "category": line["category"]}
                )
                + "\n"
            )
    print(f"Data was successfully preprocessed and saved by {output_path} .")


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to jsonl dataset you want to prepare.",
    )
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    path_to_data = args.input
    to_jsonl(path_to_data)


if __name__ == "__main__":
    main()
