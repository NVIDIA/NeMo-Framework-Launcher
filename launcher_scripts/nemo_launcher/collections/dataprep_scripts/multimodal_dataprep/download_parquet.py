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

import glob, os
import hydra
from huggingface_hub import snapshot_download
import dask.dataframe as dd
from tqdm import tqdm


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg):
    """
    Download the parquet files from a huggingface repository, and optionally sub-partition it.
    This download step should only be run in one process. It should be quick (a few minutes to an hour)
    """
    dataset_repo_id = cfg.get("dataset_repo_id")
    download_parquet_dir = cfg.get("download_parquet_dir")
    parquet_subpartitions = cfg.get("parquet_subpartitions", 1)
    parquet_pattern = cfg.get("parquet_pattern")
    downloaded_path = snapshot_download(repo_id=dataset_repo_id, repo_type="dataset",
                                        cache_dir=download_parquet_dir, local_dir=download_parquet_dir,
                                        local_dir_use_symlinks=False, allow_patterns=parquet_pattern)
    parquet_file_list = glob.glob(os.path.join(downloaded_path, "**", parquet_pattern), recursive=True)
    print(f"*** Downloaded {len(parquet_file_list)} parquet files. With {parquet_subpartitions} subpartitions, "
          f"please launch {len(parquet_file_list)*parquet_subpartitions} jobs in the next step. ***")
    print("Sub-partitioning individual parquet files...")
    if parquet_subpartitions > 1:
        for parquet_file in tqdm(parquet_file_list):
            os.makedirs(os.path.basename(parquet_file)+"_parts", exist_ok=True)
            dd.read_parquet(parquet_file) \
              .repartition(parquet_subpartitions) \
              .to_parquet(parquet_file+"_parts")
            os.remove(parquet_file)


if __name__ == "__main__":
    main()
