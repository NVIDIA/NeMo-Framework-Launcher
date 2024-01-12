#!/bin/bash


# setup
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=/workspace/launch_scripts:${PYTHONPATH}

# command 1
bash -c "
  torchrun --nproc_per_node=1 /opt/NeMo-Aligner/examples/nlp/data/steerlm/preprocess_openassistant.py \
  data_config=steerlm_openasst_data_prep \
  cluster_type=interactive \
  launcher_scripts_path=/workspace/launch_scripts \
  output_directory=/workspace/launch_scripts/data/steerlm/ " 2>&1 | tee -a /workspace/tests/download_steerlm_openasstant/preprocess/log-download_steerlm_openasstant_0112_135605.out
