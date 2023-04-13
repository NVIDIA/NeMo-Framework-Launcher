#!/bin/bash

#Users should specify the following directories
NEMO_MEGATRON_LAUNCHER_DIR=/opt/NeMo-Megatron-Launcher
DATA_DIR=

#Users should setup their cluster type in /launcher_scripts/conf/config.yaml
python3 ${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts/main.py \
training=gpt3/20b \
stages=[training] \
launcher_scripts_path=${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts \
data_dir=${DATA_DIR} \
base_results_dir=${NEMO_MEGATRON_LAUNCHER_DIR}/results \
training.run.name="a100_20b_1node" \
training.trainer.num_nodes=1 \
training.model.global_batch_size=32 \
training.model.tensor_model_parallel_size=2 \
training.model.pipeline_model_parallel_size=4 \
training.model.transformer_engine=True \
training.run.time_limit=0:20:00 \
