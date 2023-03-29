#!/bin/bash
set -euxo pipefail

ROOT=${1}
DATASET_NAME=${2}

cd ${ROOT}

# Shallow-clone the InstructPix2Pix repository
git clone --depth 1 https://github.com/timothybrooks/instruct-pix2pix.git
cd instruct-pix2pix

# Follow the download instruction
bash scripts/download_data.sh ${DATASET_NAME}

# Move the downloaded files and remove the repository
mv data/${DATASET_NAME} ..
cd ..
rm -r instruct-pix2pix