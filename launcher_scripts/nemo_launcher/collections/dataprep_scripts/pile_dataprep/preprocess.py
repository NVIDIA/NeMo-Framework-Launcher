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

import os
import subprocess
from time import sleep

import hydra
import nemo_launcher.utils.file_utils as utils  # TODO: check if this in python path
import psutil


def get_model_type(data_config):
    known_types = [
        "t5",
        "bert",
        "gpt3",
        "llama",
        "falcon",
        "baichuan2",
        "chatglm",
        "mistral",
        "mixtral",
    ]
    for m_type in known_types:
        if m_type in data_config:
            return m_type


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg):
    launcher_scripts_path = cfg.get("launcher_scripts_path")
    data_config = cfg.get("data_config")
    data_dir = cfg.get("data_dir")
    rm_extracted = cfg.get("rm_extracted")
    tokenizer_type = cfg.get("tokenizer_type")
    tokenizer_library = cfg.get("tokenizer_library")
    tokenizer_model = cfg.get("tokenizer_model")
    assert data_dir is not None, "data_dir must be a valid path"

    # Vocab
    vocab_dir = cfg.get("vocab_save_dir")
    assert vocab_dir is not None, "vocab_save_dir must be a valid path."
    if "gpt" in tokenizer_type.lower():
        vocab_path = os.path.join(launcher_scripts_path, vocab_dir, "vocab.json")
    elif tokenizer_library.lower() == "huggingface":
        vocab_path = None
    else:
        vocab_path = os.path.join(launcher_scripts_path, vocab_dir, "vocab.txt")

    # Merges
    merges_dir = cfg.get("merges_save_dir")
    assert merges_dir is not None, "merges_save_dir must be a valid path."
    merges_path = os.path.join(launcher_scripts_path, merges_dir, "merges.txt")

    # This compile doesn't seem to do anything. It compiles
    # "helpers.cpython-38-x86_64-linux-gnu.so", but since that file already
    # exists, it doesn't do anything. Force make via: touch helpers.cpp
    megatron_dir = "/opt/NeMo/nemo/collections/nlp/data/language_modeling/megatron"
    compiled_helpers_lib = os.path.join(megatron_dir, "compiled_helpers_lib")
    compilecmd = (
        f"cd /opt/NeMo; git rev-parse HEAD; "
        f"cd {megatron_dir}; "
        f"touch helpers.cpp; make;"
    )

    code_path = (
        "/opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py"
    )
    hf_cache = os.environ.get(
        "TRANSFORMERS_CACHE", os.environ.get("HF_HOME", "/temp_root/.cache/")
    )
    runcmd = (
        f"cd {megatron_dir}; "
        f'export PYTHONPATH="/opt/NeMo/.:$PYTHONPATH"; '
        f'export TRANSFORMERS_CACHE="{hf_cache}"; '
        f"python3 {code_path} "
    )

    if cfg.get("cluster_type") == "bcm":
        file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
        extracted_path = os.path.join(data_dir, f"{file_number:02d}.jsonl")

        model_type = get_model_type(data_config)
        output_prefix = os.path.join(data_dir, f"my-{model_type}_{file_number:02d}")

        flags = (
            f"--input {extracted_path} "
            f"--output-prefix {output_prefix} "
            f"--dataset-impl mmap "
            f"--tokenizer-type {tokenizer_type} "
            f"--tokenizer-library {tokenizer_library} "
            f"--tokenizer-model {tokenizer_model} "
            f"--workers $SLURM_CPUS_ON_NODE "
        )
        if vocab_path is not None:
            flags += f"--vocab {vocab_path} "
        if model_type == "bert":
            # Used for bert binary head (Next sentence predition)
            flags += "--split-sentences "
        else:
            flags += f"--merge-file {merges_path} " f"--append-eod "

        os.system(compilecmd)
        runcmd += f"{flags} "
        os.system(runcmd)
        if rm_extracted:
            os.remove(extracted_path)
    elif cfg.get("cluster_type") in ["bcp", "k8s"]:
        file_numbers = cfg.get("file_numbers")
        files_list = utils.convert_file_numbers(file_numbers)
        if cfg.get("cluster_type") == "bcp":
            wrank = int(os.environ.get("RANK", 0))
            wsize = int(os.environ.get("WORLD_SIZE", 1))
            lrank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            # Assumes launched via mpirun:
            #   mpirun -N <nnodes> -npernode 1 ...
            wrank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
            wsize = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
            lrank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))

        if lrank == 0:
            # Compile once per node. Should be one container instance per node.
            os.system(compilecmd)
            os.system(f"touch {compiled_helpers_lib}")
        else:
            while not os.path.exists(compiled_helpers_lib):
                sleep(1)

        files_list_groups = utils.split_list(files_list, wsize)
        files_to_preproc = files_list_groups[wrank]
        ncpus = psutil.cpu_count(logical=False)
        for file_number in files_to_preproc:
            extracted_path = os.path.join(data_dir, f"{file_number:02d}.jsonl")

            model_type = get_model_type(data_config)
            output_prefix = os.path.join(data_dir, f"my-{model_type}_{file_number:02d}")

            flags = (
                f"--input {extracted_path} "
                f"--output-prefix {output_prefix} "
                f"--dataset-impl mmap "
                f"--tokenizer-type {tokenizer_type} "
                f"--tokenizer-library {tokenizer_library} "
                f"--tokenizer-model {tokenizer_model} "
                f"--workers {ncpus} "
            )
            if vocab_path is not None:
                flags += f"--vocab {vocab_path} "
            if model_type == "bert":
                # Used for bert binary head (Next sentence predition)
                flags += "--split-sentences "
            else:
                flags += f"--merge-file {merges_path} " f"--append-eod "

            proc = subprocess.Popen(runcmd + flags, shell=True)
            proc.wait()
            if rm_extracted:
                os.remove(extracted_path)


if __name__ == "__main__":
    main()
