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


import math
import pynvml
import os
import sys
from collections import defaultdict

import hydra

global cuda_capability
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
cuda_capability, _ = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
pynvml.nvmlShutdown()


@hydra.main(config_path="conf", config_name="get_ub_cfg_file")
def get_ub_cfg_file(cfg):
    """
    Find and return the userbuffer config file. If it doesn't exist return `null`.
    """
    global cuda_capability
    device_name = None
    if cuda_capability == 8:
        device_name = "a100"
    elif cuda_capability == 9:
        device_name = "h100"    
    ub_cfg_path = cfg.get("ub_cfg_path")
    tp_size = cfg.get("tp_size")
    hidden_size = cfg.get("hidden_size")
    mb_size = cfg.get("mb_size")
    seqlen = cfg.get("seqlen")
    cfg_file_name =  f"ub_cfg_{device_name}_h{hidden_size}_tp{tp_size}_mbs{mb_size}_seqlen{seqlen}.yaml"
    cfg_file = os.path.join(ub_cfg_path, cfg_file_name)

    if os.path.isfile(cfg_file):
        print(f"{cfg_file}")
    else:
        print(f"null")


@hydra.main(config_path="conf", config_name="get_ln_sm_margin")
def get_ln_sm_margin(cfg):
    """
    Set SM margin to LayerNorm layer at H100. This is to overlap LN kernel with communication kernels.
    """
    global cuda_capability
    if cuda_capability == 9:
        print(4)
    else:
        print(0)


@hydra.main(config_path="conf", config_name="get_ag_overlap")
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
    if sys.argv[1] == "name=get_ub_cfg_file":
        get_ub_cfg_file()
    elif sys.argv[1] == "name=get_ln_sm_margin":
        get_ln_sm_margin()
    elif sys.argv[1] == "name=get_ag_overlap":
        get_ag_overlap()
    else:
        raise ValueError("The provided conditional config function does not exist.")
