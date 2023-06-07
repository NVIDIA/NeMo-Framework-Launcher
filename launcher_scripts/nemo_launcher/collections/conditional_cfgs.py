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

global device_arch
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
device_arch = pynvml.nvmlDeviceGetArchitecture(handle)
pynvml.nvmlShutdown()


@hydra.main(config_path="conf", config_name="get_ub_cfg_file")
def get_ub_cfg_file(cfg):
    """
    """
    global device_arch
    device_name = None
    if device_arch == pynvml.NVML_DEVICE_ARCH_AMPERE:
        device_name = "a100"
    elif device_arch == pynvml.NVML_DEVICE_ARCH_HOPPER:
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
    """
    global device_arch
    if device_arch == pynvml.NVML_DEVICE_ARCH_HOPPER:
        print(4)
    else:
        print(0)


if __name__ == "__main__":
    if sys.argv[1] == "name=get_ub_cfg_file":
        get_ub_cfg_file()
    elif sys.argv[1] == "name=get_ln_sm_margin":
        get_ln_sm_margin()