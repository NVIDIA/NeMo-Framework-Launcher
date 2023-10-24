#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:20:00

# Load libraries
export NCCL_TOPO_FILE="/nccl/topo.xml"
export 
LD_LIBRARY_PATH="/opt/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH"

# EFA configurations
export FI_PROVIDER=efa

# NCCL configurations
export NCCL_DEBUG=info
export NCCL_BUFFSIZE=33554432
export NCCL_DEBUG_SUBSYS=init,net,graph,env
export NCCL_MIN_NCHANNELS=32

# ompi conf
export OPAL_PREFIX=/opt/amazon/openmpi
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_pml=^cm
export OMPI_MCA_btl_tcp_if_exclude=lo,docker0

#pmix conf
export PMIX_MCA_gds=^ds12
export PMIX_MCA_btl=tcp,self
export PMIX_MCA_pml=^cm
export PMIX_MCA_btl_tcp_if_exclude=lo,docker0

env | grep "SLURMD_NODENAME="
env | grep "SLURM_NODELIST="

module load openmpi
srun --mpi=pmix --nodes=2 --tasks-per-node=8 --container-image=../../nemo_megatron_training.sqsh \
     --container-mounts="$PWD:/nccl" \
     bash -c "
     /nccl/nccl-tests/build/all_reduce_perf -b 256M -e 8G -f 2 -c 1 -n 10"