#! /bin/bash

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


#
# Downloads the FastText classifier
#


set -eu

res_dir=$1

## Download the fasttext model
if [ ! -f "${res_dir}/lid.176.bin" ]; then
  wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P ${res_dir}
fi

## Replace template text with path to downloaded model
sed -i "s:<Path to the FasText language id model (e.g., lid.176.bin)>:${res_dir}/lid.176.bin:g" ${res_dir}/fasttext_langid.yaml