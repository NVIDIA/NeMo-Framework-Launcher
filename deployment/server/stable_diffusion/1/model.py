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
import json
import os
import time

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from omegaconf import OmegaConf
from PIL import Image
from polygraphy import cuda
from transformers import CLIPTokenizer

from .sampler import DDIMSampler, PLMSSampler
from .utils import Engine, device_view


class TritonPythonModel:
    # stable diffusion model
    def initialize(self, args):
        config = json.loads(args['model_config'])
        base_path = f"{os.path.dirname(os.path.abspath(__file__))}/plan/"
        with open(os.path.join(base_path, "conf.yaml"), "rb") as fp:
            self.config = OmegaConf.load(fp.name)
        self.max_batch_size = self.config['batch_size']
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.cond_stage_model = Engine(os.path.join(base_path, "clip.plan"))
        self.decode_model = Engine(os.path.join(base_path, "vae.plan"))
        self.denoise_model = Engine(os.path.join(base_path, "unet.plan"))
        self.engines = [self.cond_stage_model, self.decode_model, self.denoise_model]
        loadEngines(self.engines)
        loadResources(self.engines, self.config)
        # in_channels = model.model.diffusion_model.in_channels #What is this?
        sampler_type = self.config.sampler.sampler_type
        self.sampler = initialize_sampler(self.denoise_model, sampler_type.upper())

    def execute(self, requests):
        # for each request, generate a response
        responses = []
        max_length_tokens = self.config.clip.max_length
        downsampling_factor = self.config.downsampling_factor
        height = self.config.height
        width = self.config.width
        in_channels = self.config.in_channels
        batch_size = self.max_batch_size
        # for decode
        scale_factor = 0.18215
        # Sampler params
        beta_schedule = "linear"
        timesteps = 1000
        linear_start = 0.00085
        linear_end = 0.0120
        cosine_s = 0.008
        for request in requests:
            # Perform inference on the request and append it to responses list...
            # retrieve input data from input buffers by name
            # need to decode BYTES to string
            prompt = [p.decode() for p in (pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy())]
            seed = pb_utils.get_input_tensor_by_name(request, "seed")
            seed = 0 if seed is None else seed.as_numpy()[0]
            unconditional_guidance_scale = pb_utils.get_input_tensor_by_name(request, "unconditional_guidance_scale")
            unconditional_guidance_scale = (
                self.config.clip.unconditional_guidance_scale
                if unconditional_guidance_scale is None
                else unconditional_guidance_scale.as_numpy()[0]
            )
            inference_steps = pb_utils.get_input_tensor_by_name(request, "inference_steps")
            inference_steps = (
                self.config.sampler.inference_steps if inference_steps is None else inference_steps.as_numpy()[0]
            )
            eta = pb_utils.get_input_tensor_by_name(request, "eta")
            eta = self.config.sampler.eta if eta is None else eta.as_numpy()[0]
            print("[I] Running StableDiffusion TRT pipeline")
            cond, u_cond = encode_prompt(
                self.cond_stage_model,
                self.tokenizer,
                prompt,
                unconditional_guidance_scale,
                batch_size,
                max_length_tokens,
            )
            torch.manual_seed(seed)
            latent_shape = [batch_size, height // downsampling_factor, width // downsampling_factor]
            latents = torch.randn(
                [batch_size, in_channels, height // downsampling_factor, width // downsampling_factor]
            ).to(torch.cuda.current_device())
            samples, intermediates = self.sampler.sample(
                S=inference_steps,
                conditioning=cond,
                batch_size=batch_size,
                shape=latent_shape,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=u_cond,
                eta=eta,
                x_T=latents,
            )
            images = decode_images(self.decode_model, samples)
            images = torch_to_numpy(images)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("generated_image", images)]
            )
            responses.append(inference_response)
        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list m ust match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
        print("Thanks for trying out stable diffusion")


def clip_encode(cond_stage_model, tokenizer, text, max_length):

    batch_encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_length=True,
        return_overflowing_tokens=False,
        padding="max_length",
        return_tensors="pt",
    )
    tokens = batch_encoding["input_ids"].to("cuda", non_blocking=True)
    z = cond_stage_model.infer({"tokens": device_view(tokens.type(torch.int32))})['logits'].clone()
    # z = cond_stage_model(tokens).last_hidden_state
    seq_len = (z.shape[1] + 8 - 1) // 8 * 8
    z = torch.nn.functional.pad(z, (0, 0, 0, seq_len - z.shape[1]), value=0.0)
    return z


def encode_prompt(cond_stage_model, tokenizer, prompt, unconditional_guidance_scale, batch_size, max_length):
    # Run Engines here

    c = clip_encode(cond_stage_model, tokenizer, batch_size * prompt, max_length)
    if unconditional_guidance_scale != 1.0:
        uc = clip_encode(cond_stage_model, tokenizer, batch_size * [""], max_length)
    else:
        uc = None
    return c, uc


def initialize_sampler(model, sampler_type):
    if sampler_type == 'DDIM':
        sampler = DDIMSampler(model)
    elif sampler_type == 'PLMS':
        sampler = PLMSSampler(model)
    else:
        raise ValueError(f'Sampler {sampler_type} is not supported for {cls.__name__}')
    return sampler


def decode_images(model, samples):
    # Run Engine here
    z = samples
    z = 1.0 / 0.18215 * z
    images = model.infer({"z": device_view(z)})['logits'].clone()

    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)

    return images


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def torch_to_numpy(images):
    numpy_images = images.cpu().permute(0, 2, 3, 1).numpy()
    return numpy_images


def loadEngines(engines):
    for engine in engines:
        engine.load()
        engine.activate()


def runEngine(engines, model_name, feed_dict):
    engine = engines[model_name]
    return engine.infer(feed_dict, self.stream)


def loadResources(engines, config):
    height = config.height
    width = config.width
    max_batch_size = config.batch_size
    stream = cuda.Stream()
    for e in engines:
        e.stream = stream
    engines[0].allocate_buffers(shape_dict={'tokens': config.clip.tokens, 'logits': config.clip.logits}, device="cuda")
    engines[1].allocate_buffers(
        shape_dict={'z': config.vae.z, 'logits': (max_batch_size, 3, height, width)}, device="cuda"
    )
    engines[2].allocate_buffers(
        shape_dict={
            'x': config.unet.x,
            't': (max_batch_size * 2,),
            'context': config.unet.context,
            'logits': config.unet.logits,
        },
        device="cuda",
    )
