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
import io
import os
import pickle
import re
import tarfile
import time
from typing import Optional

import hydra
import numpy as np
import lightning.pytorch as pl
import torch
import torch.utils.data as data
import webdataset as wds
from nemo.collections.multimodal.data.stable_diffusion.augmentation.augmentations import (
    construct_image_augmentations,
)
from nemo.collections.multimodal.modules.stable_diffusion.distributions.distributions import (
    DiagonalGaussianDistribution,
)
from nemo.core import Serialization
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

_IMG_EXTENSIONS = "jpg jpeg png ppm pgm pbm pnm".split()


class EncodingCacher(pl.LightningModule, Serialization):
    def __init__(self, output_dir, tar_chunk_size, precache_cfg, tar_prefix=""):
        super().__init__()
        self.automatic_optimization = False  # a LightningModule parameter
        self.output_tar_folder = output_dir
        # need to delete tarfiles from this task folder, because we're appending to existing tarfiles
        if self.global_rank == 0:
            for fname in glob.glob(f"{self.output_tar_folder}/{tar_prefix}_r*.*"):
                try:
                    os.remove(fname)
                    print("Deleted", fname)
                except FileNotFoundError:
                    pass
        else:
            while len(glob.glob(f"{self.output_tar_folder}/{tar_prefix}_r*.*")) > 0:
                time.sleep(1)

        os.makedirs(output_dir, exist_ok=True)
        self.encodings_config: ListConfig[DictConfig] = precache_cfg.encodings
        self.save_orig_image = "image" in (precache_cfg.save_original_in_tar or [])
        self.save_orig_text = "text" in (precache_cfg.save_original_in_tar or [])
        self.save_orig_video = "video" in (precache_cfg.save_original_in_tar or [])

        for m in self.encodings_config:
            if m.modality == "image":
                self.image_ext = m.extension
                self.image_ext_PIL = (
                    "JPEG" if self.image_ext.upper() == "JPG" else self.image_ext
                )
            elif m.modality == "text":
                self.text_ext = m.extension
            elif m.modality == "video":
                self.video_ext = m.extension

        self.encoder_models = torch.nn.ModuleList()
        for encoding_config in self.encodings_config:
            self.encoder_models.append(
                self.instantiate_encoder(encoding_config.get("encoder_config", None))
            )

        self.output_tar_chunk_size = tar_chunk_size
        self.cur_tar_num = 0
        self.cur_tar_size = 0
        self.tar_prefix = tar_prefix
        self.pickle_encodings = precache_cfg.get("pickle_encodings", True)

    def instantiate_encoder(self, config):
        if not config:
            return None
        model = self.from_config_dict(config).eval()  # from the serialization class
        for param in model.parameters():
            param.requires_grad = False
        return model

    def cast_precision(self, t, precision):
        if isinstance(t, list):
            return [self.cast_precision(elem, precision) for elem in t]
        if not isinstance(t, torch.Tensor):
            return t
        if precision in [16, "16"]:
            return t.half()
        elif precision in [32, "32"]:
            return t.float()
        else:
            raise NotImplementedError

    @torch.no_grad()
    def encode(self, batch, batch_idx):
        print(f"get_input with batch_idx {batch_idx}")
        all_tensors = {}
        for encoding_config, encoder_model in zip(
            self.encodings_config, self.encoder_models
        ):
            if encoder_model is None:
                continue
            x = batch[encoding_config.modality]
            x = self.cast_precision(x, encoding_config.precision)
            if hasattr(encoder_model, "encode") and callable(encoder_model.encode):
                encoded = encoder_model.encode(x)
            elif callable(encoder_model):
                encoded = encoder_model(x)
            else:
                raise ValueError(
                    f"Provided class {encoder_model.__class__} does not look like an encoder"
                )

            # postprocess the encoder output for different cases
            if not encoding_config.get("store_pad_tokens", True) and len(encoded) == 2:
                # if we ignore pad tokens for caching, then the provided encoder should return a text mask.
                text_encoded, text_mask = encoded
                encoded = [
                    text_encoded[i][text_mask[i] == 1]
                    for i in range(text_encoded.size(0))
                ]  # list of tensors

            if isinstance(encoded, DiagonalGaussianDistribution):
                encoded = encoded.parameters

            all_tensors[encoding_config.key] = self.cast_precision(
                encoded, encoding_config.precision
            )
        return all_tensors

    def _get_tarname(self):
        return f"{self.output_tar_folder}/{self.tar_prefix}_r{self.global_rank}_{self.cur_tar_num}.tar"

    @torch.no_grad()
    def save_tarfiles(self, batch, all_tensors, source_names=None):
        batch_size = len(list(batch.values())[0])

        def write_tar_content(idx):
            tar_name = self._get_tarname()
            open(tar_name + ".INCOMPLETE", "w").close()  # mark file as incomplete
            with tarfile.open(tar_name, "a") as tar:
                print("writing to tar:", tar_name)
                while (
                    idx < batch_size and self.cur_tar_size < self.output_tar_chunk_size
                ):
                    fileobj_name = (
                        source_names[idx]
                        if source_names is not None
                        else f"{self.cur_tar_size:04d}"
                    )
                    if self.pickle_encodings:
                        tensors = {
                            k: v[idx].to("cpu").numpy() for k, v in all_tensors.items()
                        }
                        # serialize data into a bytestream
                        with io.BytesIO() as abuf:
                            pickle.dump(tensors, abuf)
                            write_fileobj_to_tar(tar, f"{fileobj_name}.pickle", abuf)
                    else:
                        for k, v in all_tensors.items():
                            with io.BytesIO() as abuf:
                                np.save(abuf, v[idx].to("cpu").numpy())
                                write_fileobj_to_tar(tar, f"{fileobj_name}.{k}", abuf)

                    if self.save_orig_image:
                        with io.BytesIO() as abuf:
                            batch["image_original"][idx].save(
                                abuf, format=self.image_ext_PIL
                            )
                            write_fileobj_to_tar(
                                tar, f"{fileobj_name}.{self.image_ext}", abuf
                            )
                    if self.save_orig_text:
                        with io.BytesIO(bytes(batch["text"][idx], "utf-8")) as abuf:
                            write_fileobj_to_tar(
                                tar, f"{fileobj_name}.{self.text_ext}", abuf
                            )
                    if self.save_orig_video:
                        with io.BytesIO(bytes(batch["video"][idx])) as abuf:
                            write_fileobj_to_tar(
                                tar, f"{fileobj_name}.{self.video_ext}", abuf
                            )
                    idx += 1
                    self.cur_tar_size += 1
            return idx

        idx = 0
        idx = write_tar_content(idx)

        if self.cur_tar_size == self.output_tar_chunk_size:
            os.remove(self._get_tarname() + ".INCOMPLETE")
            # this tar file is finished
            self.cur_tar_size = 0
            self.cur_tar_num += 1

        if idx == batch_size:
            return

        # there is remaining items in the batch that didn't fit in the last tar file
        idx = write_tar_content(idx)
        # assert idx == batch_size

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        all_tensors = self.encode(batch, batch_idx)
        if "__url__" in batch and "__key__" in batch:
            source_names = [
                os.path.basename(url) + "/" + key
                for url, key in zip(batch["__url__"], batch["__key__"])
            ]
        else:
            source_names = None
        self.save_tarfiles(batch, all_tensors, source_names)


def pil_loader(key, data):
    r"""
    Function to load an image.
    If the image is corrupt, it returns a black image.
    Args:
        key: Image key.
        data: Image data stream.
    """

    extension = re.sub(r".*[.]", "", key)
    if extension.lower() not in _IMG_EXTENSIONS:
        return None

    with io.BytesIO(data) as stream:
        img = Image.open(stream)
        img.load()
        img = img.convert("RGB")

    return img


def write_fileobj_to_tar(tar: tarfile.TarFile, file_name: str, fileobj: io.BytesIO):
    """
    Write data as a pickle file named "file_name" to the specified tar file
    This is used for precaching image/text encodings in webdataset
    Reference: https://stackoverflow.com/questions/32074161/write-data-directly-to-a-tar-archive
    """
    fileobj.seek(0)
    info = tarfile.TarInfo(name=file_name)
    info.size = len(fileobj.getbuffer())
    tar.addfile(tarinfo=info, fileobj=fileobj)


def caching_collate_fn(batch):
    """
    In precaching, sometimes it is necessary to pass in the original image (as a PIL Image object) to the batch
    Modify collate function such that it accepts PIL images in the batch (without trying to convert it to a tensor)
    Following the example of default_collate_fn in collate.py
    """
    from PIL.Image import Image
    from torch.utils.data._utils.collate import collate, default_collate_fn_map

    def collate_Image_fn(batch, *, collate_fn_map=None):
        return batch

    caching_collate_fn_map = default_collate_fn_map.copy()
    caching_collate_fn_map[Image] = collate_Image_fn
    return collate(batch, collate_fn_map=caching_collate_fn_map)


def get_webdataset_loader(precache_cfg, urls, input_tar_dir=""):
    img_transform = construct_image_augmentations(
        {
            "resize_smallest_side": "512",
            "center_crop_h_w": "512,512",
            "horizontal_flip": False,
        }
    )

    def tuple_to_dict(inp):
        for input in inp:
            out_dict = {}
            for (i, modality_cfg) in enumerate(precache_cfg.encodings):
                if modality_cfg.modality.startswith("image"):
                    out_dict[modality_cfg.modality] = img_transform(input[i])
                else:
                    out_dict[modality_cfg.modality] = input[i]
            # e.g. {'images': input[0], 'text': input[1]}
            out_dict["__key__"] = input[-1]
            out_dict["__url__"] = input[-2].replace(os.path.join(input_tar_dir, ""), "")
            # url only stores identifying information about the tarfile, no extraneously long path
            yield out_dict

    modality_exts = " ".join(
        modality_cfg.extension for modality_cfg in precache_cfg.encodings
    )
    dataset = wds.DataPipeline(
        wds.SimpleShardList(urls),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.split_by_node,
        wds.split_by_worker,
        wds.decode(pil_loader, handler=wds.warn_and_continue),
        wds.to_tuple(modality_exts + " __url__ __key__", handler=wds.warn_and_continue),
    ).compose(tuple_to_dict)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=precache_cfg.batch_size_per_GPU,
        num_workers=precache_cfg.dataloader_num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=caching_collate_fn,
    )

    return dataloader


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg):
    task_id = cfg.get("override_task_id", None) or int(
        os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    )
    ntasks = cfg.get("override_task_count", None) or int(
        os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)
    )

    precache_cfg = OmegaConf.load(cfg.precache_config_path)
    input_tar_dir = cfg.input_dir
    tar_chunk_size = cfg.tar_chunk_size
    output_tar_folder = cfg.output_dir

    urls = sorted(glob.glob(os.path.join(input_tar_dir, "**", "*.tar"), recursive=True))
    if len(urls) == 0:
        raise FileNotFoundError(f"Could not find any tar files in {input_tar_dir}")
    slc_start, slc_end = (
        task_id * len(urls) // ntasks,
        (task_id + 1) * len(urls) // ntasks,
    )
    print(
        f"Task {task_id}/{ntasks} is processing files {slc_start} to {slc_end - 1} (total 0-{len(urls) - 1})"
    )

    dataloader = get_webdataset_loader(
        precache_cfg, urls[slc_start:slc_end], input_tar_dir
    )

    pl.seed_everything(42)
    # we use pytorch lightning so make multi-node precaching easy
    trainer = pl.Trainer(**precache_cfg.lightning)
    trainer.fast_dev_run = True

    model = EncodingCacher(
        output_tar_folder, tar_chunk_size, precache_cfg, tar_prefix=f"t{task_id}"
    )
    trainer.predict(model, dataloader)
    print(f"Task {task_id} finished.")


if __name__ == "__main__":
    main()
