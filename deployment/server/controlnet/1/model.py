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

from pathlib import Path

import einops
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from omegaconf import OmegaConf
from PIL import Image
from torch.nn import functional as F
from transformers import CLIPTokenizer

from .sampler import DDIMSampler, PLMSSampler
from .utils import Engine, device_view


class TritonPythonModel:
    def initialize(self, args):
        base_path = Path(__file__).parent.resolve() / "plan"
        with (base_path / "conf.yaml").open("rb") as fp:
            self.config = OmegaConf.load(fp.name)

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.cond_stage_model = Engine(base_path / "clip.plan")
        self.decode_model = Engine(base_path / "vae.plan")
        self.denoise_model = Engine(base_path / "unet.plan")
        self.control_model = Engine(base_path / "controlnet.plan")

        load_engines(
            self.cond_stage_model, self.decode_model, self.denoise_model, self.control_model,
        )
        load_resources(
            self.cond_stage_model, self.decode_model, self.denoise_model, self.control_model,
        )

        sampler_type = self.config.sampler.sampler_type
        self.sampler = initialize_sampler(self.denoise_model, sampler_type, self.control_model)

    def _get_single_input(self, request, name, default=None):
        # get an optional input of size [1], or return default if None
        param = pb_utils.get_input_tensor_by_name(request, name)
        return param.as_numpy()[0] if param is not None else default

    def _get_control_scales(self, strength, guess_mode):
        return [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

    def execute(self, requests):
        # for each request, generate a response
        responses = []
        max_length_tokens = self.config.clip.max_length
        downsampling_factor = self.config.downsampling_factor
        height = self.config.height
        width = self.config.width
        in_channels = self.config.in_channels
        batch_size = self.config.batch_size

        for request in requests:
            # Perform inference on the request and append it to responses list...
            # retrieve input data from input buffers by name
            # need to decode BYTES to string
            prompt = self._get_single_input(request, "prompt").decode()

            control = pb_utils.get_input_tensor_by_name(request, "control").as_numpy()  # HWC uint8

            seed = self._get_single_input(request, "seed")
            unconditional_guidance_scale = self._get_single_input(
                request, "unconditional_guidance_scale", default=self.config.clip.unconditional_guidance_scale,
            )
            inference_steps = self._get_single_input(
                request, "inference_steps", default=self.config.sampler.inference_steps
            )
            eta = self._get_single_input(request, "eta", default=self.config.sampler.eta)
            guess_mode = self._get_single_input(request, "guess_mode", default=False)
            hint_image_size = self.config.hint_image_size
            strength = self._get_single_input(request, "strength", default=1)
            control_scales = self._get_control_scales(strength, guess_mode)

            print("[I] Running ControlNet TRT pipeline")
            txt_cond, txt_u_cond = encode_prompt(
                self.cond_stage_model,
                self.tokenizer,
                prompt,
                unconditional_guidance_scale,
                batch_size,
                max_length_tokens,
            )
            control = get_control_input(control, batch_size, hint_image_size)
            cond = {"c_concat": control, "c_crossattn": txt_cond}
            u_cond = {
                "c_concat": None if guess_mode else control,
                "c_crossattn": txt_u_cond,
            }

            if seed is not None:
                torch.manual_seed(seed)

            latent_shape = [
                batch_size,
                height // downsampling_factor,
                width // downsampling_factor,
            ]
            latents = torch.randn(batch_size, in_channels, *latent_shape[-2:], device="cuda")
            samples, intermediates = self.sampler.sample(
                S=inference_steps,
                conditioning=cond,
                control_scales=control_scales,
                batch_size=batch_size,
                shape=latent_shape,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=u_cond,
                eta=eta,
                x_T=latents,
            )
            images = decode_images(self.decode_model, samples)
            images = nchw_torch_to_nhwc_numpy(images)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("generated_image", images)]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
        print("Thanks for trying out ControlNet")


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
    tokens = batch_encoding["input_ids"].to(dtype=torch.int32, device="cuda")
    z = cond_stage_model.infer({"tokens": device_view(tokens)})["logits"].clone()
    seq_len = (z.shape[1] + 8 - 1) // 8 * 8
    z = torch.nn.functional.pad(z, (0, 0, 0, seq_len - z.shape[1]), value=0.0)
    return z


def encode_prompt(
    cond_stage_model, tokenizer, prompt, unconditional_guidance_scale, batch_size, max_length,
):
    c = clip_encode(cond_stage_model, tokenizer, batch_size * [prompt], max_length)
    if unconditional_guidance_scale != 1.0:
        uc = clip_encode(cond_stage_model, tokenizer, batch_size * [""], max_length)
    else:
        uc = None
    return c, uc


def get_control_input(image, batch_size, hint_image_size):
    control = torch.as_tensor(np.array(image), dtype=torch.float32, device="cuda") / 255.0
    control = control.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    control = einops.rearrange(control, "b h w c -> b c h w").contiguous()
    control = F.interpolate(control, (hint_image_size, hint_image_size))
    return control


def decode_images(model, samples):
    z = samples / 0.18215
    images = model.infer({"z": device_view(z)})["logits"].clone()
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    return images


def initialize_sampler(model, sampler_type, control_model):
    if sampler_type.upper() == "DDIM":
        sampler = DDIMSampler(model, control_model=control_model)
    elif sampler_type.upper() == "PLMS":
        sampler = PLMSSampler(model, control_model=control_model)
    else:
        raise ValueError(f"Sampler {sampler_type} is not supported")
    return sampler


def nchw_torch_to_nhwc_numpy(images):
    return images.cpu().permute(0, 2, 3, 1).numpy()


def load_engines(*engines):
    for engine in engines:
        engine.load()
        engine.activate()


def load_resources(clip, vae, unet, control):
    clip.allocate_buffers()
    vae.allocate_buffers()
    unet.allocate_buffers()
    control.allocate_buffers()
