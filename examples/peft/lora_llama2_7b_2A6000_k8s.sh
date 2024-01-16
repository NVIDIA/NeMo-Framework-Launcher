#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
set -u

##################################################
# Example invocations:
#   HOST_PATH=<...>
#   DATA_DIR=$HOST_PATH/data RESTORE_FROM_PATH=$HOST_PATH/llama2_7b.nemo ../examples/peft/lora_llama2_7b_2A6000_k8s.sh cluster.host_path=$HOST_PATH
#   
#   NFS_SERVER=<...>
#   NFS_PATH=<...>
#   DATA_DIR=$NFS_PATH/data RESTORE_FROM_PATH=$NFS_PATH/llama2_7b.nemo ../examples/peft/lora_llama2_7b_2A6000_k8s.sh cluster.nfs_server=$NFS_SERVER cluster.nfs_path=$NFS_PATH
##################################################

#Users should specify the following directories
NEMO_MEGATRON_LAUNCHER_DIR=$(readlink -f ${SCRIPT_DIR}/../..)
DATA_DIR=${DATA_DIR:-}
RESTORE_FROM_PATH=${RESTORE_FROM_PATH}
HELM_RELEASE_NAME=${HELM_RELEASE_NAME:-a6000-7b-2gpu}

# peft.model.megatron_amp_O2=false is needed on containers earlier than 23.11 that
# do not include https://github.com/NVIDIA/NeMo/pull/7971
TRANSIENT_OVERRIDES="peft.model.megatron_amp_O2=false"

if [[ -z "${DATA_DIR}" ]] || [[ ! -d "${DATA_DIR}" ]]; then
  cat <<EOF
DATA_DIR=$DATA_DIR needs to be set to a directory that is available on both this node and the workers.

If hostPath:
  1. mkdir -p ${DATA_DIR:-\$DATA_DIR}
  2. DATA_DIR should be a subdir of "cluster.host_path=..."
If NFS:
  1. DATA_DIR should be a subdir of "cluster.nfs_path=..."
EOF
  exit 1
fi

# gbs=128 worked for 8 A100
# gbs=16 should work for 1 A100 assuming 80G, which is total what 2A6000 can handle, but this OOMd so gbs=4
#Users should setup their cluster type in /launcher_scripts/conf/config.yaml
HYDRA_FULL_ERROR=1 python3 ${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts/main.py \
cluster=k8s \
cluster_type=k8s \
"cluster.ib_count='0'" \
container=nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.11 \
stages=[peft] \
peft=llama/squad \
launcher_scripts_path=${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts \
data_dir=${DATA_DIR} \
base_results_dir=${NEMO_MEGATRON_LAUNCHER_DIR}/results \
peft.run.name="${HELM_RELEASE_NAME}" \
peft.trainer.num_nodes=1 \
peft.trainer.devices=2 \
peft.model.global_batch_size=2 \
peft.model.micro_batch_size=1 \
peft.model.restore_from_path=$RESTORE_FROM_PATH \
$TRANSIENT_OVERRIDES \
$@

cat <<EOF
To run again, you need to clean up the helm chart that was just deployed.

  $ helm uninstall ${HELM_RELEASE_NAME}
EOF

