# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS"" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import os
import time

import torch
import triton_python_backend_utils as pb_utils
from nemo.collections.multimodal.models.imagen.precond import ContinousDDPMPrecond, EDMPrecond
from nemo.collections.multimodal.modules.imagen.sampler.sampler import DDPMSampler, EDMSampler
from omegaconf import OmegaConf
from polygraphy import cuda
from transformers import T5Tokenizer

from .Engine import Engine, device_view


class UNetWrapper(torch.nn.Module):
    def __init__(self, engine, forward_func):
        super().__init__()
        self.engine = engine
        self.forward_func = forward_func

    def forward(
        self, x, time, text_embed, text_mask, x_low_res=None, time_low_res=None,
    ):
        input_dict = {
            "x": device_view(x),
            "time": device_view(time),
            "text_embed": device_view(text_embed),
            "text_mask": device_view(text_mask),
        }

        if x_low_res is not None:
            input_dict["x_low_res"] = device_view(x_low_res)

        if time_low_res is not None:
            input_dict["time_low_res"] = device_view(time_low_res)

        (logits,) = self.forward_func(engine=self.engine, inputs=input_dict, output_names=["logits"])

        return logits


class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        base_path = f"{os.path.dirname(os.path.abspath(__file__))}/plan/"
        with open(os.path.join(base_path, "conf.yaml"), "rb") as fp:
            self.config = OmegaConf.load(fp.name)

        # Get output configuration
        images_config = pb_utils.get_output_config_by_name(model_config, "images")

        # Convert Triton types to numpy types
        self.images_dtype = pb_utils.triton_string_to_numpy(images_config["data_type"])

        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-11b", model_max_length=self.config.t5.model_seq_len)
        self.batch_size = self.config["batch_size"]
        self._init_engines(base_path)

    def _init_engines(self, base_path):
        stream = cuda.Stream()

        self.t5_trt = Engine(os.path.join(base_path, "t5.plan"))
        shape_dict = {
            "input_ids": (1, self.config.t5.model_seq_len),
            "attn_mask": (1, self.config.t5.model_seq_len),
            "encoded_text": (1, *self.config.models[0].text_embed[1:]),
            "text_mask": (1, *self.config.models[0].text_mask[1:]),
        }
        self.t5_trt.set_engine(stream, shape_dict)

        low_res_size = None
        self.models = []
        for i, model_cfg in enumerate(self.config.models):
            engine = Engine(os.path.join(base_path, f"unet{i}.plan"))
            self.models.append(UNetWrapper(engine, self.get_result_from_engine))
            shape_dict = {
                "x": (self.batch_size, *model_cfg.x[1:]),
                "time": (self.batch_size,),
                "text_embed": (self.batch_size, *self.config.models[0].text_embed[1:]),
                "text_mask": (self.batch_size, *self.config.models[0].text_mask[1:]),
                "logits": (self.batch_size, *model_cfg.x[1:]),
            }

            if low_res_size is not None:
                shape_dict["x_low_res"] = (self.batch_size, *low_res_size[1:])

                if model_cfg.noise_cond_aug:
                    shape_dict["time_low_res"] = (self.batch_size,)

            engine.set_engine(stream, shape_dict)
            low_res_size = model_cfg.x

    def tokenize(self, text_batch):
        encoded = self.t5_tokenizer.batch_encode_plus(
            text_batch,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.t5.model_seq_len,
            truncation=True,
        )

        input_ids = encoded.input_ids.int().cuda()
        attn_mask = encoded.attention_mask.int().cuda()

        return input_ids, attn_mask

    def get_result_from_engine(self, engine, inputs, output_names):
        result = engine.infer(inputs)
        output_list = []
        for name in output_names:
            output_list.append(result[name].clone())

        return tuple(output_list)

    def sample_image(self, noise_map, text_encoding, text_mask, x_low_res=None, sampling_steps=None, cfg=-1):
        unet = self.models[self.model_idx]
        model_cfg = self.config.models[self.model_idx]
        cfg = model_cfg.cond_scale if cfg == -1 else cfg
        unet_type = "sr" if self.model_idx != 0 else "base"
        noise_cond_aug = model_cfg.noise_cond_aug
        if model_cfg.preconditioning_type == 'DDPM':
            model = ContinousDDPMPrecond(unet=unet, **model_cfg.preconditioning, noise_cond_aug=noise_cond_aug)
            sampler = DDPMSampler(unet_type=unet_type, denoiser=model.scheduler)
        elif model_cfg.preconditioning_type == 'EDM':
            model = EDMPrecond(unet=unet, **model_cfg.preconditioning, noise_cond_aug=noise_cond_aug)
            model.inference = True
            sampler = EDMSampler(unet_type=unet_type)

        return sampler(
            model,
            noise_map,
            text_encoding,
            text_mask,
            x_low_res,
            cfg,
            sampling_steps,
            self.config.thresholding_method,
        )

    def execute(self, requests):
        responses = []

        device = torch.device("cuda")

        for request in requests:
            text_input = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()
            seed = pb_utils.get_input_tensor_by_name(request, "seed")
            cfg = pb_utils.get_input_tensor_by_name(request, "cfg")
            seed = 2000 if seed is None else seed.as_numpy()[0][0]
            cfg = -1 if cfg is None else cfg.as_numpy()[0][0]
            text_input = text_input.tolist()[0][0].decode("utf-8")

            promptlist = [text_input]

            input_ids, attn_mask = self.tokenize(promptlist)

            feed_dict = {"input_ids": device_view(input_ids), "attn_mask": device_view(attn_mask)}

            encoded_text, text_mask = self.get_result_from_engine(
                engine=self.t5_trt, inputs=feed_dict, output_names=["encoded_text", "text_mask"]
            )

            for bnum in range(encoded_text.shape[0]):
                nvalid_elem = text_mask[bnum].sum().item()
                encoded_text[bnum][nvalid_elem:] = 0

            encoded_text = encoded_text.repeat(self.batch_size, 1, 1)
            text_mask = text_mask.repeat(self.batch_size, 1)
            torch.random.manual_seed(seed)
            noise_maps = []
            steps = []
            for model_cfg in self.config.models:
                noise_map = torch.randn((self.batch_size, *model_cfg.x[1:]), device=device)
                noise_maps.append(noise_map)
                steps.append(model_cfg.step)

            x_low_res = None
            for idx, (noise_map, step) in enumerate(zip(noise_maps, steps)):
                self.model_idx = idx
                t0 = time.perf_counter()
                generated_images = self.sample_image(
                    noise_map=noise_map,
                    text_encoding=encoded_text,
                    text_mask=text_mask,
                    x_low_res=x_low_res,
                    sampling_steps=step,
                    cfg=cfg,
                )
                total_time = time.perf_counter() - t0
                print((f"model{idx} time: %.2f s" % (total_time)))
                x_low_res = generated_images

            imageList = ((x_low_res + 1) / 2).clamp_(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            images = pb_utils.Tensor("images", imageList.astype(self.images_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[images])
            responses.append(inference_response)

        return responses
