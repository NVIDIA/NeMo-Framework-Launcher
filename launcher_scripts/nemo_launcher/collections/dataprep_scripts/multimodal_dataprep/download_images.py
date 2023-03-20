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

import glob, os, subprocess
import hydra


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg):
    """
    Download images from the URLs provided in the parquet files, using img2dataset
    This download step will take a significant amount of time, so it should be parallelized by launching multiple
    slurm tasks each taking on one shard, as well as using multiple processes and threads within each task.
    """
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    download_parquet_dir = cfg.get("download_parquet_dir")
    parquet_pattern = cfg.get("parquet_pattern")
    num_processes = cfg.get("download_num_processes")
    if num_processes <= 0:
        num_processes = int(os.environ.get("SLURM_CPUS_ON_NODE"))

    parquet_file_list = glob.glob(os.path.join(download_parquet_dir, "**", parquet_pattern), recursive=True)
    if len(parquet_file_list) != num_tasks:
        print(f"WARNING: Number of slurm tasks ({num_tasks}) must equal to the number of parquet files "
              f"after subpartitioning ({len(parquet_file_list)})")
        print("WARNING: If you continue executing the script, image data may not be downloaded completely.")

    parquet_file_name = sorted(parquet_file_list)[task_id]
    output_folder_path = os.path.join(cfg.get("download_images_dir"), os.path.basename(parquet_file_name))
    os.makedirs(output_folder_path, exist_ok=True)

    img2dataset_kwargs = {
        "input_format": "parquet",
        "url_col": "URL",
        "caption_col": "TEXT",
        "output_format": "webdataset",
        "url_list": parquet_file_name,
        "output_folder": output_folder_path,
        "processes_count": num_processes,
        "thread_count": cfg.get("download_num_threads"),
    }
    img2dataset_kwargs.update(cfg.get("img2dataset_additional_arguments"))

    cmd_list = ["img2dataset"]
    cmd_list.extend(f"--{k}={v}" for k, v in img2dataset_kwargs.items())
    print("Running: ", cmd_list)
    subprocess.run(cmd_list)


if __name__ == "__main__":
    main()
