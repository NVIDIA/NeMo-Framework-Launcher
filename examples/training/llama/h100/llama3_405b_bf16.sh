#!/bin/bash

#Users should setup their cluster type in /launcher_scripts/conf/config.yaml
NEMO_FRAMEWORK_LAUNCHER_DIR=${NEMO_FRAMEWORK_LAUNCHER_DIR:-"/opt/NeMo-Framework-Launcher"}
DATA_DIR=${DATA_DIR}
TOK_PATH=${TOK_PATH}

HYDRA_FULL_ERROR=1 python3 ${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts/main.py \
training=llama/llama3_405b \
stages=[training] \
data_dir=${DATA_DIR} \
launcher_scripts_path=${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts \
base_results_dir=${NEMO_FRAMEWORK_LAUNCHER_DIR}/results \
training.run.name="llama3_1_405b_bf16" \
training.run.time_limit=0:30:00 \
training.trainer.num_nodes=72 \
training.model.global_batch_size=252 \
training.model.tokenizer.model=${TOK_PATH} \
+training.model.optim.grad_sync_dtype=bf16 \
