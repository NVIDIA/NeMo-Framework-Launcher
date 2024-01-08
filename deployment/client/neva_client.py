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
import argparse
import queue
import time
from functools import partial

import nemo.collections.multimodal.data.neva.conversation as conversation_lib
import numpy as np
import tritonclient.grpc as triton_grpc
import tritonclient.utils
from PIL import Image


def generate_conv(prompt, images):
    conv = conversation_lib.conv_llava_llama_2.copy()
    text = (prompt, images, "Crop")
    conv.append_message(conv.roles[0], text)
    return conv.get_prompt(), conv.get_images(), conv.sep2


def stream_callback(queue, result, error):
    if error:
        queue.put(error)
    else:
        response = result.get_response(as_json=True)

        if "outputs" in response:
            # the very last response might have no output, just the final flag
            generated_text = result.as_numpy("generated_text")[0].decode()
            queue.put(generated_text)

        if response["parameters"]["triton_final_response"]["bool_param"]:
            # end of the generation
            queue.put(None)


def main(args):
    if args.prompt is None:
        prompt = "<image>\nCan you describe this image?"
    else:
        prompt = args.prompt

    if args.image_path is None:
        image = Image.fromarray(np.random.rand(224, 224, 3).astype(np.uint8))
    else:
        image = Image.open(args.image_path).convert('RGB')

    temperature = 0.2
    max_new_tokens = 512
    seed = 1
    prompt, images, stop = generate_conv(prompt, image)
    images = np.array(images, dtype=np.object_)
    grpc_client = triton_grpc.InferenceServerClient(url='localhost:8001', verbose=False)
    triton_input_grpc = [
        triton_grpc.InferInput('prompt', (1,), tritonclient.utils.np_to_triton_dtype(np.object_)),
        triton_grpc.InferInput('images', images.shape, tritonclient.utils.np_to_triton_dtype(np.object_)),
        triton_grpc.InferInput('stop', (1,), tritonclient.utils.np_to_triton_dtype(np.object_)),
        triton_grpc.InferInput('max_new_tokens', (1,), tritonclient.utils.np_to_triton_dtype(np.uint16)),
        triton_grpc.InferInput('temperature', (1,), tritonclient.utils.np_to_triton_dtype(np.float32)),
        triton_grpc.InferInput('random_seed', (1,), tritonclient.utils.np_to_triton_dtype(np.uint32)),
    ]
    triton_input_grpc[0].set_data_from_numpy(np.array([prompt], dtype=np.object_))
    triton_input_grpc[1].set_data_from_numpy(images)
    triton_input_grpc[2].set_data_from_numpy(np.array([stop], dtype=np.object_))
    triton_input_grpc[3].set_data_from_numpy(np.array([max_new_tokens], dtype=np.uint16))
    triton_input_grpc[4].set_data_from_numpy(np.array([temperature], dtype=np.float32))
    triton_input_grpc[5].set_data_from_numpy(np.array([seed], dtype=np.uint32))
    triton_output_grpc = triton_grpc.InferRequestedOutput('generated_text')

    result_queue = queue.Queue()
    grpc_client.start_stream(callback=partial(stream_callback, result_queue))
    grpc_client.async_stream_infer(
        model_name='neva', inputs=triton_input_grpc, outputs=[triton_output_grpc], enable_empty_final_response=True
    )

    while True:
        response_streaming = result_queue.get()

        if type(response_streaming) == tritonclient.utils.InferenceServerException:
            print(f"Error: {response_streaming.message()}")
            grpc_client.stop_stream()
            break

        if response_streaming is not None:
            output = response_streaming.strip()
            print(f"{output}")
        else:
            break

        time.sleep(0.03)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--image_path", type=str)
    args = parser.parse_args()
    main(args)
