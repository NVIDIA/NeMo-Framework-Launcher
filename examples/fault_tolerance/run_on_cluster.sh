#!/bin/bash

# NOTE: NeMo-Megatron-Launcher requirements should be installed
# e.g. cd /NeMo-Megatron-Launcher && pip install -r requirements.txt

CLUSTER="draco-rno"
CONTAINER="/gpfs/fs1/projects/ent_joc/users/jbieniusiewi/ft/NeMo-Megatron-Launcher-gwe-ft/dl+gwe+fault_tolerance_related+nemo-gwe-ft+test.sqsh" # "gitlab-master.nvidia.com/dl/gwe/fault_tolerance_related/nemo-gwe-ft:test"
RUN_NAME="fault_tol_gpt3_5b_no_err"
NODES=4

FT_ARGS="
    ++training.exp_manager.create_fault_tolerance_callback=True \
    ++training.exp_manager.fault_tolerance.max_subsequent_job_failures=1
"

#    ++training.exp_manager.fault_tolerance.initial_rank_heartbeat_timeout=900 \
#    ++training.exp_manager.fault_tolerance.rank_heartbeat_timeout=600 \
#    ++training.exp_manager.fault_tolerance.max_subsequent_job_failures=3 \
#    ++training.exp_manager.fault_tolerance.max_rank_restarts=0
#    ++training.exp_manager.fault_tolerance.simulated_fault.fault_type=random \
#    ++training.exp_manager.fault_tolerance.simulated_fault.base_delay=900 

if [ "$CLUSTER" == "draco-rno" ]; then
    PARTITION="batch_short_dgx1_m2"
    ACCOUNT="coreai_dlalgo_llm"
    JOB_PREFIX="coreai_dlalgo_llm-test:"
    CLUSTER_SPECIFIC_ARGS="++cluster.nv_meta=\"ml-model.fault_tol_tests\""
    USR_DIR="/gpfs/fs1/projects/ent_joc/users/jbieniusiewi"                                                           
    LAUNCHER_DIR="${USR_DIR}/ft/NeMo-Megatron-Launcher-gwe-ft"
else
    echo "Unknown cluster: $CLUSTER"
    exit 1
fi

# create dummy data this that is required by the launcher     
# we will use mock data
mkdir -p ${LAUNCHER_DIR}/dummy_data_dir                                                                           
                                                                                                                                                                           
HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub HYDRA_FULL_ERROR=1 PYTHONWARNINGS="ignore" python3 ${LAUNCHER_DIR}/launcher_scripts/main.py \
    training=gpt3/5b \
    stages=["training"] \
    numa_mapping.enable=True \
    data_dir=${LAUNCHER_DIR}/dummy_data_dir \
    ++training.model.data.mock_dataset=True \
    ++training.model.data.data_impl="mock" \
    ++training.model.data.data_prefix=[] \
    launcher_scripts_path=${LAUNCHER_DIR}/launcher_scripts  \
    container_mounts=[$USR_DIR:$USR_DIR] \
    container=${CONTAINER} \
    cluster.partition=${PARTITION} \
    cluster.account=${ACCOUNT} \
    cluster.job_name_prefix=${JOB_PREFIX} \
    ${CLUSTER_SPECIFIC_ARGS} \
    ++cluster.gres="gpu:8" \
    ++cluster.signal="TERM@240" \
    training.exp_manager.resume_if_exists=True \
    training.exp_manager.create_checkpoint_callback=True \
    training.exp_manager.checkpoint_callback_params.save_top_k=1 \
    training.exp_manager.resume_ignore_no_checkpoint=True \
    training.run.name=${RUN_NAME} \
    training.run.time_limit=00:20:00 \
    training.trainer.max_time=00:01:30:00 \
    training.trainer.num_nodes=${NODES} \
    training.trainer.devices=8 \
    training.trainer.log_every_n_steps=10 \
    training.trainer.val_check_interval=50 \
    ++training.trainer.precision=16 \
    ++training.model.tokenizer.merge_file="${USR_DIR}/bpe/gpt2-merges.txt" \
    ++training.model.tokenizer.vocab_file="${USR_DIR}/bpe/gpt2-vocab.txt" \
    training.trainer.enable_checkpointing=False \
    training.model.micro_batch_size=1 \
    training.model.global_batch_size=$((${NODES} * 8)) \
    training.model.tensor_model_parallel_size=2 \
    training.model.pipeline_model_parallel_size=4 \
    ${FT_ARGS}

