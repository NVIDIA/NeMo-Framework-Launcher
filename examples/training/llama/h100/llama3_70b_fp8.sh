#!/bin/bash

#Users should setup their cluster type in /launcher_scripts/conf/config.yaml
NEMO_MEGATRON_LAUNCHER_DIR=${NEMO_MEGATRON_LAUNCHER_DIR:-"/opt/NeMo-Megatron-Launcher"}
DATA_DIR=${DATA_DIR}
TOK_PATH=${TOK_PATH}

HYDRA_FULL_ERROR=1 python3 ${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts/main.py \
training=llama/llama3_70b \
stages=[training] \
data_dir=${DATA_DIR} \
launcher_scripts_path=${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts \
base_results_dir=${NEMO_MEGATRON_LAUNCHER_DIR}/results \
training.run.name="llama3_70b_fp8" \
training.run.time_limit=0:30:00 \
training.trainer.num_nodes=8 \
training.model.global_batch_size=128 \
training.model.fp8=True \
training.model.fp8_hybrid=True \
training.model.tokenizer.model=${TOK_PATH} \
+training.model.optim.grad_sync_dtype=bf16 \
training/tp_overlap@training.model.ub_tp_comm_overlap_cfg=ub_cfg_h100_fp8_h8192_tp4_mbs1_seqlen8192 \
