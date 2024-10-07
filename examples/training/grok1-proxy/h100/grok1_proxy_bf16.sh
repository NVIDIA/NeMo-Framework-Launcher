#!/bin/bash

#Users should setup their cluster type in /launcher_scripts/conf/config.yaml
NEMO_FRAMEWORK_LAUNCHER_DIR=${NEMO_FRAMEWORK_LAUNCHER_DIR:-"/opt/NeMo-Framework-Launcher"}
DATA_DIR=${DATA_DIR}
TOK_PATH=${TOK_PATH}

HYDRA_FULL_ERROR=1 python3 ${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts/main.py \
training=grok/grok1_proxy \
stages=[training] \
data_dir=${DATA_DIR} \
launcher_scripts_path=${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts \
base_results_dir=${NEMO_FRAMEWORK_LAUNCHER_DIR}/results \
training.run.name="grok1_proxy_bf16" \
training.run.time_limit=0:30:00 \
training.model.tokenizer.model=${TOK_PATH} \
+env_vars.NCCL_P2P_NET_CHUNKSIZE=2097152 \
training.model.moe_grouped_gemm=False \
training.model.gradient_accumulation_fusion=True \
+training.model.optim.grad_sync_dtype=bf16 \
training.trainer.num_nodes=64 \
+training.model.context_parallel_size=2 \
training.model.sequence_parallel=True \
training.model.tensor_model_parallel_size=4 \
training.model.pipeline_model_parallel_size=8 \
training.model.virtual_pipeline_model_parallel_size=8 \
training.model.gc_interval=40
