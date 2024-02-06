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
import glob
import os
import pickle
from typing import Optional

import hydra
from reorganize_tar import reorganize


def generate_wdinfo(tar_folder: str, chunk_size: int, output_path: Optional[str]):
    if not output_path:
        return
    tar_files = []
    for fname in glob.glob(os.path.join(tar_folder, "*.tar")):
        # only glob one level of folder structure because we only write basename to the tar files
        if os.path.getsize(fname) > 0 and not os.path.exists(f"{fname}.INCOMPLETE"):
            tar_files.append(os.path.basename(fname))
    data = {
        "tar_files": sorted(tar_files),
        "chunk_size": chunk_size,
        "total_key_count": len(tar_files) * chunk_size,
    }
    print(data)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print("Generated", output_path)


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg):
    tar_folder = cfg.input_dir
    tar_chunk_size = cfg.tar_chunk_size
    extensions = cfg.file_ext_in_tar
    output_wdinfo_path = cfg.output_wdinfo_path

    # reorganize last few tar files
    incomplete_tarfiles = glob.glob(
        os.path.join(tar_folder, "**", "*.tar.INCOMPLETE"), recursive=True
    )
    incomplete_tarlist = [p.replace(".INCOMPLETE", "") for p in incomplete_tarfiles]

    if len(incomplete_tarlist) > 0:
        reorganize(
            incomplete_tarlist,
            tar_folder,
            extensions=extensions,
            fname_prefix="last_few_",
        )

    generate_wdinfo(tar_folder, tar_chunk_size, output_wdinfo_path)


if __name__ == "__main__":
    main()
