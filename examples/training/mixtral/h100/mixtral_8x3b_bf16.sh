#!/bin/bash

#Users should setup their cluster type in /launcher_scripts/conf/config.yaml
NEMO_FRAMEWORK_LAUNCHER_DIR=${NEMO_FRAMEWORK_LAUNCHER_DIR:-"/opt/NeMo-Framework-Launcher"}
DATA_DIR=${DATA_DIR}
TOK_PATH=${TOK_PATH}

HYDRA_FULL_ERROR=1 python3 ${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts/main.py \
training==mixtral/mixtral_8x3b \
stages=[training] \
data_dir=${DATA_DIR} \
launcher_scripts_path=${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts \
base_results_dir=${NEMO_FRAMEWORK_LAUNCHER_DIR}/results \
training.run.name="mixtral_8x3b_bf16" \
training.run.time_limit=0:30:00 \
training.model.tokenizer.model=${TOK_PATH} \
training.model.pipeline_model_parallel_size=1 \
training.model.virtual_pipeline_model_parallel_size=null \
training.model.expert_model_parallel_size=8 \
training.model.moe_grouped_gemm=False \
training.model.gradient_accumulation_fusion=True \
training.model.optim.name=mcore_distributed_optim \
+training.model.optim.overlap_grad_sync=True \
+training.model.optim.overlap_param_sync=True \
+training.model.optim.grad_sync_dtype=bf16 \
+env_vars.NCCL_P2P_NET_CHUNKSIZE=2097152 \
