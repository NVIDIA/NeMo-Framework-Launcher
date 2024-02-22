#!/bin/bash

#Users should setup their cluster type in /launcher_scripts/conf/config.yaml
NEMO_MEGATRON_LAUNCHER_DIR=${NEMO_MEGATRON_LAUNCHER_DIR:-"/opt/NeMo-Megatron-Launcher"}
DATA_DIR=${DATA_DIR}
TOK_PATH=${TOK_PATH}

HYDRA_FULL_ERROR=1 python3 ${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts/main.py \
training=llama/llama2_7b \
stages=[training] \
data_dir=${DATA_DIR} \
launcher_scripts_path=${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts \
base_results_dir=${NEMO_MEGATRON_LAUNCHER_DIR}/results \
training.run.name="llama2_7b_fp8" \
training.run.time_limit=0:15:00 \
training.trainer.num_nodes=1 \
training.model.global_batch_size=128 \
training.model.fp8=True \
training.model.fp8_hybrid=True \
training.model.tokenizer.model=${TOK_PATH} \
+training.model.gc_interval=100 \
