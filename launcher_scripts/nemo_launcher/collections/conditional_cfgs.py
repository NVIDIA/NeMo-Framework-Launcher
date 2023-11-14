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

import sys

import hydra
import pynvml

global cuda_capability
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
cuda_capability, _ = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
pynvml.nvmlShutdown()


@hydra.main(version_base=None, config_path="conf", config_name="get_ln_sm_margin")
def get_ln_sm_margin(cfg):
    """
    Set SM margin to LayerNorm layer at H100. This is to overlap LN kernel with communication kernels.
    """
    global cuda_capability
    if cuda_capability == 9:
        print(8)
    else:
        print(0)


@hydra.main(version_base=None, config_path="conf", config_name="get_ag_overlap")
def get_ag_overlap(cfg):
    """
    Disable AG overlap with P2P ring-exchange at H100 BF16 training.
    FIXME: Fix the bug and remove this conditional setting.
    """
    global cuda_capability
    fp8 = cfg.get("fp8")
    if cuda_capability == 9:
        if fp8:
            print(1)
        else:
            print(0)
    else:
        print(1)


if __name__ == "__main__":
    if sys.argv[1] == "name=get_ln_sm_margin":
        get_ln_sm_margin()
    elif sys.argv[1] == "name=get_ag_overlap":
        get_ag_overlap()
    else:
        raise ValueError("The provided conditional config function does not exist.")
