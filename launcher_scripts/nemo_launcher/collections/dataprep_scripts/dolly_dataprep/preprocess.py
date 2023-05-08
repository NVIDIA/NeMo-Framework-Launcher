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
 python preprocess.py \
    --input=<path/to/data/file> \
    --dataset-impl=<dataset_type> \
    --tokenizer-library=<tokenizer_library> \
    --tokenizer-type=<tokenizer_type> \
    --vocab-file=<path_to_vocab> \
    --merges-file=<path_to_merges> \
    --workers=<num_workers>
"""

import os
import json
import subprocess
import numpy as np
from argparse import ArgumentParser

def to_jsonl(path_to_data):
    print(f"Preprocessing data to jsonl format...")
    output_path = f"{path_to_data.split('.')[0]}-output.jsonl"
    with open(path_to_data, 'r') as f, open(output_path, 'w') as g:
        for line in f:
            line = json.loads(line)
            context = line['context'].strip()
            if context != "":
                # Randomize context and instruction order.
                context_first = np.random.randint(0, 2) == 0
                if context_first:
                    instruction = line['instruction'].strip()
                    assert instruction != ""
                    input = f"{context}\n\n{instruction}"
                    output = line['response']
                else:
                    instruction = line['instruction'].strip()
                    assert instruction != ""
                    input = f"{instruction}\n\n{context}"
                    output = line['response']
            else:
                input = line['instruction']
                output = line['response']
            g.write(json.dumps({'input': input, 'output': output, 'category': line['category']}) + '\n')
    print(f"Data was successfully preprocessed and saved by {output_path} .")

def to_mmap(path_to_data, args):
    to_jsonl(path_to_data)
    output_path = f"{path_to_data.split('.')[0]}-output.jsonl"
    text_jsonl = f"{path_to_data.split('.')[0]}-text.jsonl"
    new_lines = []
    with open(output_path, 'r') as f, open(text_jsonl, 'w') as g:
        for line in f:
            line = json.loads(line)
            text = f"{line['input']} {line['output']}"
            g.write(json.dumps({'text': text, 'category': line['category']}) + '\n')
    
    dir_path = os.path.dirname(output_path)
    output = f"{dir_path}/dolly"
    cmd = (
        f"python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py "
        f"--input {text_jsonl} "
        f"--output-prefix {output} "
        f"--vocab-file {args.vocab_file} "
        f"--merge-file {args.merges_file} "
        f"--dataset-impl {args.dataset_impl} "
        f"--tokenizer-library {args.tokenizer_library} "
        f"--tokenizer-type {args.tokenizer_type} "
        f"--workers {args.workers} "
        f"--append-eod "
    )
    os.system(cmd)

    os.system(
        f"rm {output_path} && "
        f"rm {text_jsonl}"
    )

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to jsonl dataset you want to prepare."
    )
    parser.add_argument(
        "--dataset-impl", type=str, required=True, help="Dataset type: mmap or text."
    )
    parser.add_argument(
        "--workers", type=int, required=False, default=1, help="Tokenizer type you want to use."
    )
    parser.add_argument(
        "--vocab-file", type=str, required=False, help="Path to the vocab file."
    )
    parser.add_argument(
        "--merges-file", type=str, required=False, help="Path to the merges file."
    )
    parser.add_argument(
        "--tokenizer-library", type=str, required=False, default='megatron', help="Tokenizer library you want to use."
    )
    parser.add_argument(
        "--tokenizer-type", type=str, required=False, default='GPT2BPETokenizer', help="Tokenizer type you want to use."
    )
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    path_to_data = args.input
    dataset_type = args.dataset_impl

    if dataset_type == "text":
        to_jsonl(path_to_data)
    elif dataset_type == "mmap":
        to_mmap(path_to_data, args)
    else:
        raise ValueError("Input dataset type is not correct. Available list of types: ['text', 'mmap']")

if __name__ == '__main__':
    main()

