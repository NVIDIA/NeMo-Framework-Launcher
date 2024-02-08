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
#
# Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.

import glob
import os
from multiprocessing import Pool

import hydra
import numpy as np
from PIL import Image
from PIL.Image import Resampling
from pycocotools.coco import COCO
from tqdm import tqdm


def preprocess_one_image(input_url, output_url=None):
    im = Image.open(input_url)
    width, height = im.size
    new_width = new_height = min(width, height)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    im = im.crop((left, top, right, bottom))
    im = im.resize((256, 256), resample=Resampling.BICUBIC)
    if output_url is None:
        # follow existing paths
        output_url = input_url.replace(
            "val2014/COCO_val2014_", "coco2014_val/images_256/"
        )
    im.save(output_url, quality=95)


def preprocess_images(root_dir, num_processes):
    """
    Center-crop and resize all images in the coco 2014 validation set.
    """
    if num_processes <= 0:
        num_processes = int(os.environ.get("SLURM_CPUS_ON_NODE"))

    input_dir = os.path.join(root_dir, "val2014")
    output_dir = os.path.join(root_dir, "coco2014_val", "images_256")
    os.makedirs(output_dir, exist_ok=True)

    with Pool(num_processes) as p:
        p.map(preprocess_one_image, glob.glob(os.path.join(input_dir, "*.jpg")))


def preprocess_captions(root_dir):
    """
    randomly select 30k captions from the validation dataset of coco2014,
    and save each to a separate txt file
    """
    output_dir = os.path.join(root_dir, "coco2014_val_sampled_30k", "captions")
    os.makedirs(output_dir, exist_ok=True)

    coco_caps = COCO(os.path.join(root_dir, "captions_val2014.json"))

    # randomly select 30k captions
    caption_ids = list(coco_caps.anns.keys())
    caption_ids_30k = np.random.choice(caption_ids, size=30_000, replace=False)

    # write each caption to a file
    for cap_id in tqdm(caption_ids_30k):
        with open(os.path.join(output_dir, f"{cap_id:012d}.txt"), "w") as f:
            f.write(coco_caps.anns[cap_id]["caption"])


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg):
    if cfg.preprocess_images:
        preprocess_images(cfg.root_dir, cfg.num_processes)
    if cfg.preprocess_captions:
        preprocess_captions(cfg.root_dir)


if __name__ == "__main__":
    main()
