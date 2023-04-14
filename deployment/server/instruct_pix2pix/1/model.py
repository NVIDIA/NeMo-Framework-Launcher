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
import os
import einops
import math
import numpy as np
import os
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from einops import rearrange, repeat
import json
from .sampler import DiscreteEpsDDPMDenoiser, sample_euler_ancestral, DiagonalGaussianDistribution
from .utils import make_beta_schedule, Engine, device_view
from transformers import CLIPTokenizer
from omegaconf import OmegaConf
import triton_python_backend_utils as pb_utils
from polygraphy import cuda

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "b ... -> (n b) ...", n=3)
        cfg_sigma = einops.repeat(sigma, "b ... -> (n b) ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        out = out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
        return out



class TritonPythonModel:
    # stable diffusion model
    def initialize(self, args):
        config = json.loads(args['model_config'])
        base_path = f"{os.path.dirname(os.path.abspath(__file__))}/plan/"
        with open(os.path.join(base_path, "conf.yaml"), "rb") as fp:
            self.config = OmegaConf.load(fp.name)
        self.max_batch_size = self.config['batch_size']
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.cond_stage_model = Engine(f"{base_path}/clip.plan")
        self.encode_model = Engine(f"{base_path}/vae_encode.plan")
        self.decode_model = Engine(f"{base_path}/vae_decode.plan")
        self.denoise_model = Engine(f"{base_path}/unet.plan")
        self.engines = [self.cond_stage_model, self.encode_model, self.decode_model, self.denoise_model]
        loadEngines(self.engines)
        loadResources(self.engines, self.config)


        beta_schedule = "linear"
        timesteps = 1000
        linear_start=0.00085
        linear_end=0.0120
        cosine_s=0.008
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0)).cuda()
        self.model_wrap = DiscreteEpsDDPMDenoiser(self.denoise_model, alphas_cumprod)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)

    def execute(self, requests):

        responses = []
        save_to_file = True
        out_path = "/models/outputs/"
        resolution = self.config.resolution
        max_length = self.config.clip.max_length
        num_images_per_prompt = self.max_batch_size
        # for decode
        scale_factor = 0.18215
        #Sampler params
        beta_schedule = "linear"
        timesteps = 1000
        linear_start=0.00085
        linear_end=0.0120
        cosine_s=0.008

        for request in requests:

            prompt = [p.decode() for p in (pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy())]
            input_image = Image.fromarray(pb_utils.get_input_tensor_by_name(request, "input_image").as_numpy())
            seed = pb_utils.get_input_tensor_by_name(request, "seed")
            seed = 0 if seed is None else seed.as_numpy()[0]
            steps = pb_utils.get_input_tensor_by_name(request, "steps")
            steps = self.config.steps if steps is None else steps.as_numpy()[0]
            text_cfg_scale = pb_utils.get_input_tensor_by_name(request, "text_cfg_scale")
            text_cfg_scale = self.config.text_cfg_scale if text_cfg_scale is None else text_cfg_scale.as_numpy()[0]
            image_cfg_scale = pb_utils.get_input_tensor_by_name(request, "image_cfg_scale")
            image_cfg_scale = self.config.image_cfg_scale if image_cfg_scale is None else image_cfg_scale.as_numpy()[0]

            torch.manual_seed(seed)
            null_token = clip_encode(self.cond_stage_model, self.tokenizer, [""], max_length)
            width, height = input_image.size
            factor = resolution / max(width, height)
            factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            width = int((width * factor) // 64) * 64
            height = int((height * factor) // 64) * 64
            input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

            cond = {}
            cond["c_crossattn"] = [
                repeat(clip_encode(self.cond_stage_model, self.tokenizer, prompt, max_length),
                        "1 ... -> n ...", n=num_images_per_prompt)
            ]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").cuda(non_blocking=True)
            torch.cuda.synchronize()
            cond["c_concat"] = [
                repeat(encode_images(self.encode_model, input_image).mode(),
                        "1 ... -> n ...", n=num_images_per_prompt)
            ]

            uncond = {}
            uncond["c_crossattn"] = [
                repeat(null_token, "1 ... -> n ...", n=num_images_per_prompt)
            ]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = self.model_wrap.get_sigmas(steps).float()

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": text_cfg_scale,
                "image_cfg_scale": image_cfg_scale,
            }
            z = torch.randn_like(cond["c_concat"][0])
            z = z * sigmas[0]
            z = sample_euler_ancestral(self.model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = decode_images(self.decode_model, z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = rearrange(x, "n c h w -> n h w c")

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("generated_image", np.asarray(x.cpu().numpy()))
                ]
            )
            responses.append(inference_response)
        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list m ust match the length of `requests` list.
        return responses


def clip_encode(cond_stage_model, tokenizer, text, max_length):

    batch_encoding = tokenizer(text, truncation=True, max_length=max_length, return_length=True,
                                    return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"].to("cuda", non_blocking=True)
    z = cond_stage_model.infer({"tokens": device_view(tokens.type(torch.int32))})['logits'].clone()
    seq_len = (z.shape[1] + 8 - 1) // 8 * 8
    z = torch.nn.functional.pad(z, (0, 0, 0, seq_len-z.shape[1]), value=0.0)
    return z


def encode_images(model, images):
    logits =  model.infer({"x": device_view(images.contiguous().type(torch.float32))})['logits'].clone()
    out = DiagonalGaussianDistribution(logits)
    return out


def decode_images(model, samples):
    # Run Engine here
    z = samples
    z = 1. / 0.18215 * z
    images = model.infer({"z": device_view(z.type(torch.float32))})['logits'].clone()

    return images


def loadEngines(engines):
    for engine in engines:
        engine.load()
        engine.activate()

def runEngine(engines, model_name, feed_dict):
    engine = engines[model_name]
    return engine.infer(feed_dict, self.stream)


def loadResources(engines, config):
    max_batch_size = config.batch_size
    height = config.height
    width = config.width
    stream = cuda.Stream()
    for e in engines:
        e.stream = stream
    engines[0].allocate_buffers(shape_dict={'tokens': config.clip.tokens,'logits': config.clip.logits}, device="cuda")
    engines[1].allocate_buffers(shape_dict={'x': config.vaee.x,'logits': config.vaee.logits}, device="cuda")
    engines[2].allocate_buffers(shape_dict={'z': config.vaed.z,'logits': config.vaed.logits}, device="cuda")
    engines[3].allocate_buffers(shape_dict={'x': config.unet.x, 't': config.unet.t, 'context': config.unet.context,'logits': config.unet.logits}, device="cuda")
