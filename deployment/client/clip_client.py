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
import numpy as np
import tritonclient.grpc as triton_grpc
import tritonclient.utils


def main():
    grpc_client = triton_grpc.InferenceServerClient(url='localhost:8001', verbose=False)
    prompt = "a dog"
    triton_input_grpc = [
        triton_grpc.InferInput('images', (4, 3, 224, 224), tritonclient.utils.np_to_triton_dtype(np.float32)),
        triton_grpc.InferInput('texts', (4, 1), tritonclient.utils.np_to_triton_dtype(np.object_)),
    ]
    triton_input_grpc[0].set_data_from_numpy(np.random.rand(4, 3, 224, 224).astype(np.float32))
    triton_input_grpc[1].set_data_from_numpy(np.array([prompt] * 4, dtype=np.object_).reshape(4, 1))
    triton_output_grpc = triton_grpc.InferRequestedOutput('text_probs')
    request_grpc = grpc_client.infer(
        'clip_trt', model_version='1', inputs=triton_input_grpc, outputs=[triton_output_grpc]
    )
    outt = request_grpc.as_numpy('text_probs')


if __name__ == "__main__":
    main()
