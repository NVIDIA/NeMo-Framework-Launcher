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
Dolly data downloading.
Example usage:
 python download.py \
    --path_to_save=<path/to/save/dolly> \
    --download_link=<link/to/download>
"""

import os
from argparse import ArgumentParser

default_link = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"


def get_file_name(link):
    file_name = link.split("/")[-1]

    return file_name


def get_args(default_link=default_link):
    parser = ArgumentParser()
    parser.add_argument(
        "--path_to_save",
        type=str,
        required=True,
        help="Specify the path where to save the data.",
    )
    parser.add_argument(
        "--link_to_download",
        type=str,
        required=False,
        default=default_link,
        help="Specify the link where to download the data.",
    )
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    path_to_save = args.path_to_save
    link_to_download = args.link_to_download
    file_name = get_file_name(link_to_download)

    print(f"Downloading Dolly dataset {file_name} to {path_to_save} ...")
    os.system(f"cd {path_to_save} && " f"wget {link_to_download}")
    print(f"Dolly dataset {file_name} was successfully downloaded to {path_to_save} .")


if __name__ == "__main__":
    main()
