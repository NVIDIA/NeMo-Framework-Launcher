#!/bin/bash

# Users should specify the path to the launcher directory and the dataset in the
# commandline or in this run script.
NEMO_FRAMEWORK_LAUNCHER_DIR=${NEMO_FRAMEWORK_LAUNCHER_DIR:-"/opt/NeMo-Framework-Launcher"}
DATA_DIR=${DATA_DIR}

#Users should setup their cluster type in /launcher_scripts/conf/config.yaml
python3 ${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts/main.py \
training=gpt3/20b \
stages=[training] \
launcher_scripts_path=${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts \
data_dir=${DATA_DIR} \
base_results_dir=${NEMO_FRAMEWORK_LAUNCHER_DIR}/results \
training.run.name="20b_h100_fp8_1node" \
training.trainer.num_nodes=1 \
training.model.global_batch_size=256 \
training.model.micro_batch_size=4 \
training.model.tensor_model_parallel_size=4 \
training.model.pipeline_model_parallel_size=1 \
training.model.fp8=true \
training.run.time_limit=0:20:00 \
+training.model.optim.grad_sync_dtype=bf16 \
