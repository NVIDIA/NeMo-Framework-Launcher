#!/bin/bash

# Required settings
NEMO_MEGATRON_LAUNCHER_DIR=${NEMO_MEGATRON_LAUNCHER_DIR:-"/opt/NeMo-Megatron-Launcher"}
#MODEL= # pretrained model
#TRAIN_DS= # [/path/to/training.jsonl]
#VALID_DS= # [/path/to/validation.jsonl]

# Optional additional arguments to pass from command line
EXTRA_ARGS=${EXTRA_ARGS:-""}

# Cluster settings may be set under launcher_scripts/conf/config.yaml
python3 ${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts/main.py \
    peft=llama/sft \
    stages=[peft] \
    launcher_scripts_path=${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts \
    base_results_dir=${NEMO_MEGATRON_LAUNCHER_DIR}/results \
    peft.run.name=h100_70b_4node \
    peft.run.time_limit=0:20:00 \
    peft.trainer.devices=8 \
    peft.trainer.num_nodes=4 \
    peft.model.micro_batch_size=1 \
    peft.model.global_batch_size=128 \
    peft.model.tensor_model_parallel_size=4 \
    peft.model.pipeline_model_parallel_size=4 \
    peft.model.virtual_pipeline_model_parallel_size=10 \
    peft.model.sequence_parallel=true \
    ++peft.model.batch_p2p_comm=false \
    peft.model.overlap_p2p_comm=true \
    peft.model.fp8=true \
    peft.model.restore_from_path=${MODEL} \
    peft.model.data.train_ds.file_names=${TRAIN_DS} \
    peft.model.data.train_ds.concat_sampling_probabilities=[1.0] \
    peft.model.data.train_ds.max_seq_length=2048 \
    peft.model.data.validation_ds.file_names=${VALID_DS} \
    ${EXTRA_ARGS}

