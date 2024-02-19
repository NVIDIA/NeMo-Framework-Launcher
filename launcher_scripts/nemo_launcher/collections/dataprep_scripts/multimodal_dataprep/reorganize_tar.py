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
import tarfile
import time

import hydra
from tqdm import tqdm


def reorganize(
    files,
    save_dir,
    chunksize=1000,
    offset=0,
    extensions=(".jpg", ".txt"),
    fname_prefix="",
):
    """
    Takes a list of tar files, and reorganizes the files inside them based on their file name ignoring extension,
    such that each output tarfile contains exactly `chunksize` distinct file names (except the last tar file),
    corresponding to that many training examples.
    A log file is generated containing information about the number of files in the last tar file.

    Args:
    - files (list): A list of strings representing paths to tar files.
    - save_dir (str): The path of the directory where the reorganized tar files and log file will be saved.
    - chunksize (int, optional): The number of files to be included in each new tar file. Defaults to 1000.
    - offset (int, optional): The starting number of the new tar files' names. Defaults to 0.
    - extensions (tuple, optional): A tuple containing strings representing the file extensions to be included
                                    in the reorganized tar files. Defaults to ('.jpg', '.txt').
    - fname_prefix (str, optional): A prefix to every tar file. Defaults to "".
    """

    def _get_tarname(cur_tar):
        return os.path.join(save_dir, f"{fname_prefix}{cur_tar:05}.tar")

    cur_file = 0
    cur_tar = offset
    os.makedirs(save_dir, exist_ok=True)
    new_tar = tarfile.TarFile(_get_tarname(cur_tar), "w")
    for i, tar_name in enumerate(tqdm(files)):
        try:
            obj_name = tar_name
            if os.path.isdir(obj_name):
                continue
            with tarfile.open(obj_name) as tar_obj:
                names = tar_obj.getnames()
                all_filenames = set([os.path.splitext(name)[0] for name in names])

                for name in all_filenames:
                    file_pair = []
                    for post_script in extensions:
                        f = tar_obj.extractfile(name + post_script)
                        file_pair.append((f, name + post_script))
                    if len(file_pair) == len(extensions):
                        for f, filename in file_pair:
                            info = tarfile.TarInfo(name=filename)
                            info.size = f.seek(0, os.SEEK_END)
                            f.seek(0)
                            new_tar.addfile(tarinfo=info, fileobj=f)
                        cur_file += 1
                    if cur_file == chunksize:
                        cur_file = 0
                        cur_tar += 1
                        new_tar.close()
                        new_tar = tarfile.TarFile(_get_tarname(cur_tar), "w")
        except Exception as e:
            print(f"Failed to process tarfile {tar_name} due to {str(e)}")

    new_tar.close()
    with open(os.path.join(save_dir, "log.txt"), "w") as f:
        f.write(f"Last tar {cur_tar} only has {cur_file} files!")
    if cur_file > 0:
        open(
            _get_tarname(cur_tar) + ".INCOMPLETE", "w"
        ).close()  # mark file as incomplete
    else:
        os.remove(_get_tarname(cur_tar))


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg) -> None:
    """
    Reorganize the contents of tar files from the download_images step, so that the tar files are uniform
    (i.e. each containing an equal number (usually 1000) of training examples (image-text pairs)).
    The tar files created from the download_images step are not uniform, because there is always a portion of images
    that fail to download or are no long available.
    Uniform tar files are important if a sequential sampler is used during training (i.e. not infinite sampler).
    Uniform tar files are also important for precaching because a sequential sampler is used there.
    """

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    ntasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    root = cfg.get("input_dir")
    save_dir = cfg.get("output_dir")
    extensions = cfg.get("file_ext_in_tar")
    tar_chunk_size = cfg.get("tar_chunk_size")

    abs_save_dir = os.path.join(save_dir, f"task{task_id:04d}")
    os.makedirs(abs_save_dir, exist_ok=True)
    print(f"saving to {abs_save_dir}")

    files = sorted(glob.glob(os.path.join(root, "**", "*.tar"), recursive=True))
    if len(files) == 0:
        raise FileNotFoundError(f"Could not find any tar files in {files}")
    slc_start, slc_end = (
        task_id * len(files) // ntasks,
        (task_id + 1) * len(files) // ntasks,
    )

    start = time.time()
    print(
        f"Task {task_id}/{ntasks} is processing files {slc_start} to {slc_end - 1} (total 0-{len(files) - 1})"
    )

    reorganize(
        files[slc_start:slc_end],
        abs_save_dir,
        chunksize=tar_chunk_size,
        extensions=extensions,
    )

    end = time.time()
    print(f"Task {task_id} finished in {end - start}s")


if __name__ == "__main__":
    main()
