#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
set -eu

#Users should specify the following directories
NEMO_MEGATRON_LAUNCHER_DIR=$(readlink -f ${SCRIPT_DIR}/../../../..)
DATA_DIR=${DATA_DIR}
RESTORE_FROM_PATH=${RESTORE_FROM_PATH}
RUN_NAME=${RUN_NAME:-llama-7b-peft-lora}
PEFT_CONFIG=${PEFT_CONFIG:-llama/squad}

# peft.model.megatron_amp_O2=false is needed on containers earlier than 23.11 that
# do not include https://github.com/NVIDIA/NeMo/pull/7971
TRANSIENT_OVERRIDES="peft.model.megatron_amp_O2=false"

HYDRA_FULL_ERROR=1 python3 ${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts/main.py \
cluster=k8s_v2 \
cluster_type=k8s \
cluster.ib_interfaces=null \
container=nvcr.io/nvidia/nemo:24.03.01.framework \
stages=[peft] \
peft=${PEFT_CONFIG} \
launcher_scripts_path=${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts \
data_dir=${DATA_DIR} \
peft.run.name="${RUN_NAME}" \
peft.trainer.num_nodes=1 \
peft.trainer.devices=4 \
peft.trainer.max_epochs=null \
peft.trainer.max_steps=2000 \
peft.model.global_batch_size=128 \
peft.model.micro_batch_size=1 \
peft.model.restore_from_path=$RESTORE_FROM_PATH \
$TRANSIENT_OVERRIDES \
$@
