#!/bin/bash

#This example does pre-training GPT 40B model using torch FSDP.

# Users should specify the path to the launcher directory and the dataset in the
# commandline or in this run script.
NEMO_FRAMEWORK_LAUNCHER_DIR=${NEMO_FRAMEWORK_LAUNCHER_DIR:-"/opt/NeMo-Framework-Launcher"}
DATA_DIR=${DATA_DIR}

#Users should setup their cluster type in /launcher_scripts/conf/config.yaml
python3 ${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts/main.py \
training=gpt3/40b \
stages=[training] \
launcher_scripts_path=${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts \
data_dir=${DATA_DIR} \
base_results_dir=${NEMO_FRAMEWORK_LAUNCHER_DIR}/results \
training.trainer.precision="bf16-mixed" \
training.run.name="fsdp_40b_a100_32node" \
training.trainer.num_nodes=32 \
training.model.global_batch_size=256 \
training.model.micro_batch_size=4 \
training.model.tensor_model_parallel_size=4 \
training.model.pipeline_model_parallel_size=1 \
training.model.virtual_pipeline_model_parallel_size=null \
training.model.megatron_amp_O2=False \
training.model.use_cpu_initialization=True \
+training.model.fsdp=True \
+training.model.fsdp_sharded_checkpoint=True \
training.model.optim.name="fused_adam" \
~training.model.optim.bucket_cap_mb \
~training.model.optim.overlap_grad_sync \
~training.model.optim.overlap_param_sync \
~training.model.optim.contiguous_grad_buffer \
training.run.time_limit=0:20:00 \
