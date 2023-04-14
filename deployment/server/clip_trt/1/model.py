# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
from typing import Any, List, Union

import torch
from torch.utils.dlpack import to_dlpack, from_dlpack
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

import triton_python_backend_utils as pb_utils

class TritonPythonModel:

    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "text_probs")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        self.clip_tokenizer = AutoTokenizer(pretrained_model_name="openai/clip-vit-large-patch14")


    def get_reponse_from_model(self, model_name, inputs, output_names):
        encoding_request = pb_utils.InferenceRequest(
            model_name=model_name,
            requested_output_names=output_names,
            inputs=inputs,
        )

        response = encoding_request.exec()
        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())

        output_list = []
        for name in output_names:
            output = pb_utils.get_output_tensor_by_name(response, name)
            output_list.append(output)

        return tuple(output_list)

    def tokenize(self, texts: Union[str, List[str]], tokenizer: Any, context_length: int = 77) -> torch.IntTensor:
        texts_is_str = False
        if isinstance(texts, str):
            texts = [texts]
            texts_is_str = True

        bos_id = tokenizer.bos_id
        eos_id = tokenizer.eos_id
        all_tokens = [[bos_id] + tokenizer.text_to_ids(text) + [eos_id] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = eos_id
            result[i, :len(tokens)] = torch.tensor(tokens)

        if texts_is_str:
            result = result[0]
        return result

    def execute(self, requests):
        responses = []

        for request in requests:
            images = pb_utils.get_input_tensor_by_name(request, "images").as_numpy()
            texts = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            texts = [s.decode('utf-8') for s in texts.tolist()[0]]

            texts = self.tokenize(texts, tokenizer=self.clip_tokenizer, context_length=80)
            texts = pb_utils.Tensor.from_dlpack("texts", to_dlpack(texts))

            images = pb_utils.Tensor("images", images)

            text_probs = self.get_reponse_from_model(
                model_name="clip_vision_trt",
                inputs=[images, texts],
                output_names=["text_probs"]
            )[0]

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[text_probs])
            responses.append(inference_response)


        return responses
