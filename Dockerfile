# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

ARG BIGNLP_BACKEND=pytorch
ARG BIGNLP_BACKEND_BRANCH_TAG=23.04

FROM nvcr.io/nvidia/${BIGNLP_BACKEND}:${BIGNLP_BACKEND_BRANCH_TAG}-py3 as pytorch

#########################################
#### Build fastertransformer pytorch ####
#########################################
FROM pytorch as ft_builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends bc git-lfs&& \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# backend build
WORKDIR /workspace/FasterTransformer
RUN git clone https://github.com/NVIDIA/FasterTransformer.git /workspace/FasterTransformer

ENV NCCL_LAUNCH_MODE=GROUP
ARG SM=80
ARG BUILD_MIXED_GEMM=OFF
ARG FORCE_BACKEND_REBUILD=0
ARG SPARSITY_SUPPORT=OFF
ARG BUILD_MULTI_GPU=ON
RUN mkdir /var/run/sshd -p && \
    mkdir build -p && cd build && \
    wget https://developer.download.nvidia.com/compute/libcusparse-lt/0.1.0/local_installers/libcusparse_lt-linux-x86_64-0.1.0.2.tar.gz && \
    tar -xzvf libcusparse_lt-linux-x86_64-0.1.0.2.tar.gz && \
    cmake -DSM=${SM} -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DSPARSITY_SUPPORT=${SPARSITY_SUPPORT} -DMEASURE_BUILD_TIME=ON \
      -DBUILD_CUTLASS_MIXED_GEMM=${BUILD_MIXED_GEMM} -DCUSPARSELT_PATH=/workspace/FasterTransformer/build/libcusparse_lt/ \
      -DBUILD_MULTI_GPU=${BUILD_MULTI_GPU} -DBUILD_TRT=ON .. && \
    make -j"$(grep -c ^processor /proc/cpuinfo)"

########################################################################
#### Build training container and install fastertransformer pytorch ####
########################################################################
FROM pytorch as training

ENV NVIDIA_PRODUCT_NAME="NeMo Megatron"

ARG NVIDIA_BIGNLP_VERSION
ENV NVIDIA_BIGNLP_VERSION=$NVIDIA_BIGNLP_VERSION
LABEL com.nvidia.bignlp.version="${NVIDIA_BIGNLP_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        sox \
        swig \
        openssh-server \
        libb64-dev && \
    rm -rf /var/lib/apt/lists/*

# Install bits from BigNLP sources here...
WORKDIR /opt
### Note: if you don't want to ship the source code,
### you can do this COPY and RUN building in a separate build stage using multistage docker,
### and just install the resulting binary here using COPY --from or RUN --mount=from=
### experimental syntax
#COPY bignlp-scripts/src dst
#RUN ...

# Get fastertransformer_backend
RUN git clone https://github.com/triton-inference-server/fastertransformer_backend.git

# Install SentencePiece
RUN git clone https://github.com/google/sentencepiece.git && \
    cd sentencepiece && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    ldconfig

# Install apex
ARG APEX_COMMIT
RUN pip uninstall -y apex && \
    git clone https://github.com/NVIDIA/apex && \
	cd apex && \
    if [ ! -z $APEX_COMMIT ]; then \
        git fetch origin $APEX_COMMIT && \
        git checkout FETCH_HEAD; \
    fi && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./

# Install NeMo
ARG NEMO_COMMIT
RUN git clone https://github.com/NVIDIA/NeMo.git && \
    cd NeMo && \
    if [ ! -z $NEMO_COMMIT ]; then \
        git fetch origin $NEMO_COMMIT && \
        git checkout FETCH_HEAD; \
    fi && \
    pip uninstall -y nemo_toolkit sacrebleu && \
    pip install -e ".[nlp]" && \
    cd nemo/collections/nlp/data/language_modeling/megatron && \
    make

# HF cache
RUN python -c "from transformers import AutoTokenizer; tok_gpt=AutoTokenizer.from_pretrained('gpt2'); tok_bert=AutoTokenizer.from_pretrained('bert-base-cased'); tok_large_bert=AutoTokenizer.from_pretrained('bert-large-cased'); tok_large_uncased_bert=AutoTokenizer.from_pretrained('bert-large-uncased');"

# Install TE
ARG TE_COMMIT
RUN git clone https://github.com/NVIDIA/TransformerEngine.git && \
    cd TransformerEngine && \
    if [ ! -z $TE_COMMIT ]; then \
        git fetch origin $TE_COMMIT && \
        git checkout FETCH_HEAD; \
    fi && \
    git submodule init && git submodule update && \
    NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .

# Install Megatron-core
ARG MEGATRONCORE_COMMIT
RUN git clone https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && \
    if [ ! -z $MEGATRONCORE_COMMIT ]; then \
        git fetch origin $MEGATRONCORE_COMMIT && \
        git checkout FETCH_HEAD; \
    fi && \
    pip install -e .

# Install launch scripts
COPY . NeMo-Megatron-Launcher
RUN cd NeMo-Megatron-Launcher && \
    pip install --no-cache-dir -r requirements.txt

ENV LAUNCHER_SCRIPTS_PATH=/opt/NeMo-Megatron-Launcher/launcher_scripts
ENV PYTHONPATH=/opt/NeMo-Megatron-Launcher/launcher_scripts:${PYTHONPATH}

# pip install required python packages
RUN pip install --no-cache-dir wandb==0.15.3 \
        'best_download>=0.0.6' \
        black==20.8b1 \
        'click>=8.0.1' \
        'datasets>=1.2.1' \
        jsonlines==2.0.0 \
        lm_dataformat==0.0.19 \
        mock==4.0.3 \
        numexpr==2.7.2 \
        pybind11==2.8.0 \
        pycountry==20.7.3 \
        pytablewriter==0.58.0 \
        pytest==6.2.5 \
        sacrebleu==1.5.0 \
        'scikit-learn>=0.24.1' \
        spacy==3.1.3 \
        sqlitedict==1.6.0 \
        'transformers>=4.1' \
        tqdm-multiprocess==0.0.11 \
        zstandard==0.17.0 \
        tritonclient[all]~=2.22.4 \
	'nvidia-pytriton==0.1.5' \
        'nltk>=3.6.7' \
        'ipython>=7.31.1' \
        'torchmetrics==0.9.1'

# Copy FasterTransformer
COPY --from=ft_builder /workspace/FasterTransformer FasterTransformer

# Setup SSH config to allow mpi-operator to communicate with containers in k8s
RUN echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config && \
    sed -i 's/#   StrictHostKeyChecking ask/    StrictHostKeyChecking no/' /etc/ssh/ssh_config && \
    mkdir -p /var/run/sshd

# Examples
WORKDIR /workspace
#COPY any user-facing example scripts should go in here
RUN chmod -R a+w /workspace

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"
