#!/bin/bash

#Users should setup their cluster type in /launcher_scripts/conf/config.yaml
NEMO_FRAMEWORK_LAUNCHER_DIR=${NEMO_FRAMEWORK_LAUNCHER_DIR:-"/opt/NeMo-Framework-Launcher"}
DATA_DIR=${DATA_DIR}
TOK_PATH=${TOK_PATH}

HYDRA_FULL_ERROR=1 python3 ${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts/main.py \
training=mixtral/mixtral_8x3b \
stages=[training] \
data_dir=${DATA_DIR} \
launcher_scripts_path=${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts \
base_results_dir=${NEMO_FRAMEWORK_LAUNCHER_DIR}/results \
training.run.name="mixtral_8x3b_bf16" \
training.run.time_limit=0:30:00 \
training.model.tokenizer.model=${TOK_PATH} \
+env_vars.NCCL_P2P_NET_CHUNKSIZE=2097152 \
