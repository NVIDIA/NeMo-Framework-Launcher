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
import multiprocessing
import os
import re
import tarfile
from functools import partial

import hydra
from tqdm import tqdm


class TarFile2(tarfile.TarFile):
    """
    Custom subclass of TarFile to ignore all ReadError
    Useful in dealing with truncated tarfiles
    """

    def _load(self):
        while True:
            try:
                tarinfo = self.next()
                if tarinfo is None:
                    break
            except tarfile.ReadError as ignored:
                # as soon as a ReadError happens, ignore the error and treat whatever has been read so far as the
                # full tarfile
                break
        self._loaded = True


def reset_tarfile(tar_path, replace_current=True, sorted_filenames=False):
    """
    Rewrite the entire content of `tar_path` into a new tarfile, optionally replacing the current one
    The reason this function exists is that Python's tarfile module unpredictably produces tarfiles that are
    unable to be opened for append, due to EmptyHeaderError. See these related discussions
    https://stackoverflow.com/questions/44550090/python-tarfile-unpredictable-error-tarfile-readerror-empty-header
    https://stackoverflow.com/questions/45376065/python-3-5-tarfile-append-mode-readerror-on-empty-tar
    Therefore this function exists as a way to "fix" the tarfile while keeping all its content
    """
    try:
        new_tar_path = tar_path.replace(".tar", "_tmp.tar")
        with tarfile.open(new_tar_path, "w") as new_tar_obj:
            with TarFile2.open(tar_path) as tar_obj:
                names = tar_obj.getnames()
                if sorted_filenames:
                    names.sort()
                for name in names:
                    try:
                        f = tar_obj.extractfile(name)
                        info = tarfile.TarInfo(name=name)
                        info.size = f.seek(0, os.SEEK_END)
                        f.seek(0)
                        new_tar_obj.addfile(tarinfo=info, fileobj=f)
                    except tarfile.ReadError:
                        print(f"reset_tarfile: Skipping {name} due to ReadError")
                        continue

        if replace_current:
            os.remove(tar_path)
            os.rename(new_tar_path, tar_path)
    except Exception as e:
        print(f"Failed to reset tarfile {tar_path} due to: {str(e)}.")
        return False

    return True


def retrieve_source_objects_from_one_tar(
    url, source_dir, source_extensions, skip_incomplete=True
):
    """
    Retrieve objects that fall into `source_extensions`
    from source tar files specified as the object name in the current tar file `url`

    e.g. source_extensions = ['mp4', 'json']
    tar file `url` contains the following file objects
    | 001.tar/00001.pickle
    | 001.tar/00002.pickle
    | 002.tar/00001.pickle
    | ...
    Then this function will locate os.path.join(source_dir, '001.tar') and retrieve these objects,
    so that the merged tar file `url` will contain
    | 001.tar/00001.pickle  001.tar/00001.mp4  001.tar/00001.json
    | 001.tar/00002.pickle  001.tar/00002.mp4  001.tar/00002.json
    | 002.tar/00001.pickle  002.tar/00001.mp4  002.tar/00001.json
    | ...
    """
    if skip_incomplete and os.path.exists(url + ".INCOMPLETE"):
        print(f"✔️ Skipping {url} because it is marked as INCOMPLETE")
        return True

    def split_obj_name(obj_name):
        """
        splits 'part_000/001.tar/something/123' into
        ('part_000/001.tar', 'something/123')
        """
        return re.match(r"(.*\.tar)/(.*)", obj_name).groups()

    source_extensions = [
        "." + x if not x.startswith(".") else x for x in source_extensions
    ]

    # fix empty header issue
    try:
        tarfile.open(url, "a").close()
    except tarfile.ReadError:
        print(f"Resetting tar file for {url}")
        reset_tarfile(url)
    tar_name = "unavailable"
    try:
        with tarfile.open(url, "a") as tar_obj:
            names = tar_obj.getnames()
            names_set = set(names)
            all_obj_names = sorted(
                list(set(os.path.splitext(name)[0] for name in names))
            )

            prev_tar_name = ""
            cur_tar_source = None
            for obj_name in tqdm(all_obj_names, disable=True):
                tar_name, inner_obj_name = split_obj_name(obj_name)
                if tar_name != prev_tar_name:
                    if cur_tar_source is not None:
                        cur_tar_source.close()
                    source_tar_path = os.path.join(source_dir, tar_name)
                    if os.path.isdir(source_tar_path):
                        # weird folder structure for some tar files. not needed in general but just keep it here
                        source_tar_path = os.path.join(source_tar_path, tar_name)

                    cur_tar_source = TarFile2.open(source_tar_path)
                for ext in source_extensions:
                    # ext already starts with a dot
                    new_obj_name = tar_name + "/" + inner_obj_name + ext
                    if new_obj_name not in names_set:
                        f = cur_tar_source.extractfile(inner_obj_name + ext)
                        info = tarfile.TarInfo(name=new_obj_name)
                        info.size = f.seek(0, os.SEEK_END)
                        f.seek(0)
                        tar_obj.addfile(tarinfo=info, fileobj=f)

                prev_tar_name = tar_name
    except Exception as e:
        print(f"Failed to process tarfile {url} due to: {str(e)}. Tar name: {tar_name}")
        return False
    return True


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg):
    task_id = cfg.get("override_task_id", None) or int(
        os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    )
    ntasks = cfg.get("override_task_count", None) or int(
        os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)
    )

    append_tar_dir = (
        cfg.append_tar_dir
    )  # contains tarfiles with file obj names that correspond to src location
    source_dir = (
        cfg.source_dir
    )  # should be relative to the file obj names in each tar file
    source_extensions = cfg.source_extensions  # e.g. [.mp4, .json, .transcript]

    urls = glob.glob(os.path.join(append_tar_dir, "**", "*.tar"), recursive=True)
    if len(urls) == 0:
        raise FileNotFoundError(f"Could not find any tar files in {append_tar_dir}")
    slc_start, slc_end = (
        task_id * len(urls) // ntasks,
        (task_id + 1) * len(urls) // ntasks,
    )
    print(
        f"Task {task_id}/{ntasks} is processing files {slc_start} to {slc_end - 1} (total 0-{len(urls) - 1})"
    )

    num_processes = min(len(urls), multiprocessing.cpu_count(), 32)
    print(f"Retrieve source objects: multiprocessing with {num_processes} processes")
    with multiprocessing.Pool(num_processes) as p:
        success = p.map(
            partial(
                retrieve_source_objects_from_one_tar,
                source_dir=source_dir,
                source_extensions=source_extensions,
            ),
            urls[slc_start:slc_end],
        )
    print("success:", success.count(True), "failed:", success.count(False))

    print(f"Sorting filenames: multiprocessing with {num_processes} processes")
    with multiprocessing.Pool(num_processes) as p:
        success = p.map(
            partial(reset_tarfile, replace_current=True, sorted_filenames=True),
            urls[slc_start:slc_end],
        )
    print("success:", success.count(True), "failed:", success.count(False))


if __name__ == "__main__":
    main()
