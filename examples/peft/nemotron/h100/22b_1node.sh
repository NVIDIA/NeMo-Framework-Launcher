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
    peft=nemotron/sft \
    stages=[peft] \
    launcher_scripts_path=${NEMO_MEGATRON_LAUNCHER_DIR}/launcher_scripts \
    base_results_dir=${NEMO_MEGATRON_LAUNCHER_DIR}/results \
    peft.run.name=h100_nemotron_22b_1node \
    peft.run.time_limit=0:20:00 \
    peft.trainer.devices=8 \
    peft.trainer.num_nodes=1 \
    peft.model.micro_batch_size=1 \
    peft.model.global_batch_size=32 \
    peft.model.tensor_model_parallel_size=2 \
    peft.model.pipeline_model_parallel_size=2 \
    peft.model.sequence_parallel=true \
    peft.model.ub_tp_comm_overlap=true \
    peft.model.fp8=true \
    ++peft.model.fp8_params=true \
    ++peft.model.log_token_counts=true \
    ++peft.model.gc_interval=0 \
    peft.model.restore_from_path=${MODEL} \
    peft.model.data.train_ds.file_names=${TRAIN_DS} \
    peft.model.data.train_ds.packed_sequence=true \
    peft.model.data.train_ds.pad_to_max_length=true \
    peft.model.data.train_ds.concat_sampling_probabilities=[1.0] \
    peft.model.data.train_ds.max_seq_length=4096 \
    peft.model.data.validation_ds.file_names=${VALID_DS} \
    ${EXTRA_ARGS}

