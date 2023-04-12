#!/bin/bash
set -euxo pipefail

ROOT=${1}

cd ${ROOT}

# install the basic commands, if needed
apt-get update && apt install make gcc unzip
# install dependencies required by pycocotools and preprocessing script
pip install matplotlib
pip install cython
pip install Pillow

git clone --depth 1  https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI && make install
cd ../..

# download and extract captions
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip -j annotations_trainval2014.zip annotations/captions_val2014.json  # unzip only captions_val2014.json
rm annotations_trainval2014.zip

# download and extract val images
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm val2014.zip
