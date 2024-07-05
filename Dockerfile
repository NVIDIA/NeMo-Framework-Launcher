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

ARG LAUNCHER_BACKEND=pytorch
ARG LAUNCHER_BACKEND_BRANCH_TAG=24.03

FROM nvcr.io/nvidia/${LAUNCHER_BACKEND}:${LAUNCHER_BACKEND_BRANCH_TAG}-py3 as pytorch

##################################
#### Build training container ####
##################################
FROM pytorch as training

ENV NVIDIA_PRODUCT_NAME="NeMo Framework"

ARG NVIDIA_LAUNCHER_VERSION
ENV NVIDIA_LAUNCHER_VERSION=$NVIDIA_LAUNCHER_VERSION
LABEL com.nvidia.launcher.version="${NVIDIA_LAUNCHER_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        sox \
        swig \
        openssh-server \
        libb64-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt
### Note: if you don't want to ship the source code,
### you can do this COPY and RUN building in a separate build stage using multistage docker,
### and just install the resulting binary here using COPY --from or RUN --mount=from=
### experimental syntax

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
RUN pip install packaging
ARG APEX_COMMIT
RUN pip uninstall -y apex && \
    git clone https://github.com/NVIDIA/apex && \
	cd apex && \
    if [ ! -z $APEX_COMMIT ]; then \
        git fetch origin $APEX_COMMIT && \
        git checkout FETCH_HEAD; \
    fi && \
    pip install install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam" ./

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

# Install launch scripts
ARG ALIGNER_COMMIT
RUN git clone https://github.com/NVIDIA/NeMo-Aligner.git && \
    cd NeMo-Aligner && \
    git pull && \
    if [ ! -z $ALIGNER_COMMIT ]; then \
        git fetch origin $ALIGNER_COMMIT && \
        git checkout FETCH_HEAD; \
    fi && \
    pip install --no-deps -e .

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
    pip install -e . && \
    cd megatron/core/datasets && \
    make && \
    pip install git+https://github.com/fanshiqing/grouped_gemm@v1.0

# Install launch scripts
ARG LAUNCHER_COMMIT
RUN git clone https://github.com/NVIDIA/NeMo-Framework-Launcher.git && \
    cd NeMo-Framework-Launcher && \
    git pull && \
    if [ ! -z $LAUNCHER_COMMIT ]; then \
        git fetch origin $LAUNCHER_COMMIT && \
        git checkout FETCH_HEAD; \
    fi && \
    pip install --no-cache-dir -r requirements.txt

ENV LAUNCHER_SCRIPTS_PATH=/opt/NeMo-Framework-Launcher/launcher_scripts
ENV PYTHONPATH=/opt/NeMo-Framework-Launcher/launcher_scripts:${PYTHONPATH}

# HF cache
RUN python -c "from transformers import AutoTokenizer; tok_gpt=AutoTokenizer.from_pretrained('gpt2'); tok_bert=AutoTokenizer.from_pretrained('bert-base-cased'); tok_large_bert=AutoTokenizer.from_pretrained('bert-large-cased'); tok_large_uncased_bert=AutoTokenizer.from_pretrained('bert-large-uncased');"

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