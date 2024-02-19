#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
set -u

##################################################
# Example invocations:
#   HOST_PATH=<...>
#   DATA_DIR=$HOST_PATH/data RESTORE_FROM_PATH=$HOST_PATH/llama2_7b.nemo ../examples/peft/lora_llama2_7b_4A100_k8s.sh cluster.volumes.hostPath.path=$HOST_PATH
#   
#   NFS_SERVER=<...>
#   NFS_PATH=<...>
#   DATA_DIR=$NFS_PATH/data RESTORE_FROM_PATH=$NFS_PATH/llama2_7b.nemo ../examples/peft/lora_llama2_7b_4A100_k8s.sh cluster.volumes.nfs.server=$NFS_SERVER cluster.volumes..nfs.path=$NFS_PATH
#
#   PVC_NAME=<...>
#   PVC_SUBPATH=<...>
#   DATA_DIR=/$PVC_SUBPATH/data RESTORE_FROM_PATH=/$PVC_SUBPATH/llama2_7b.nemo ../examples/peft/lora_llama2_7b_4A100_k8s.sh cluster.volumes.persistentVolumeClaim.claimName=$PVC_NAME cluster.volumes.persistentVolumeClaim.subPath=$PVC_SUBPATH
##################################################

#Users should specify the following directories
NEMO_MEGATRON_LAUNCHER_DIR=$(readlink -f ${SCRIPT_DIR}/../..)
DATA_DIR=${DATA_DIR}
RESTORE_FROM_PATH=${RESTORE_FROM_PATH}
HELM_RELEASE_NAME=${HELM_RELEASE_NAME:-llama-7b-peft-lora}
PEFT_CONFIG=${PEFT_CONFIG:-llama/squad}

# peft.model.megatron_amp_O2=false is needed on containers earlier than 23.11 that
# do not include https://github.com/NVIDIA/NeMo/pull/7971
TRANSIENT_OVERRIDES="peft.model.megatron_amp_O2=false"

HYDRA_FULL_ERROR=1 python3 ${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts/main.py \
cluster=k8s \
cluster_type=k8s \
"cluster.ib_count='0'" \
container=nvcr.io/nvidia/nemo:24.01.framework \
stages=[peft] \
peft=${PEFT_CONFIG} \
launcher_scripts_path=${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts \
data_dir=${DATA_DIR} \
peft.run.name="${HELM_RELEASE_NAME}" \
peft.trainer.num_nodes=1 \
peft.trainer.devices=4 \
peft.trainer.max_epochs=null \
peft.trainer.max_steps=2000 \
peft.model.global_batch_size=128 \
peft.model.micro_batch_size=1 \
peft.model.restore_from_path=$RESTORE_FROM_PATH \
$TRANSIENT_OVERRIDES \
$@

cat <<EOF
To run again, you need to clean up the helm chart that was just deployed.

  $ helm uninstall ${HELM_RELEASE_NAME}
EOF

