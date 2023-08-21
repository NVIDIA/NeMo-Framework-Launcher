#! /bin/bash

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


#
# Starts a redis-cluster on a group of nodes within a Slurm cluster
# The user must provide an input directory that contains a file "hostnames.txt"
# that contains a list of hostnames that the user currently has allocated
# in their Slurm job. Also, the user can provide the number of nodes
# that they desire to use to form the redis-cluster (default is 3)
#


set -eu

base_path=`pwd`
data_dir=${1:-"./workspace/data"}
nodelist=${data_dir}/hostnames.txt

num_nodes=$(wc -l < $nodelist)
num_redis_nodes=${2:-3}
num_compute_nodes=$(($num_nodes - $num_redis_nodes))
echo "Using a total of $num_nodes nodes, $num_redis_nodes of which are used for redis"
echo "The remaining $num_compute_nodes nodes can be used for compute"

cluster_join_wait=${3:-40}

# Container-related variables
base_dir=`pwd`
mounts="${base_dir}:${base_dir}"

# Split nodelist
redis_nodelist=${data_dir}/hostnames_redis.txt
compute_nodelist=${data_dir}/hostnames_compute.txt
cat $nodelist | head -n $num_redis_nodes > ${redis_nodelist}
cat $nodelist | tail -n $num_compute_nodes > ${compute_nodelist}

# Redis config and log dir
redis_conf_dir=${data_dir}/redis_confs
redis_log_dir=${data_dir}/redis_logs

rm -rf ${redis_conf_dir}/* ${redis_log_dir}/*
mkdir -p ${redis_conf_dir} ${redis_log_dir}

# Get max cpus per node to request
# the entire node for each redis intance
max_cpus_per_node=$(srun --nodes=1 bash ${base_path}/ndc/deduplication/redis/get_num_cpus_per_node)

# Start all instances
counter=0
for hostname in $(cat $redis_nodelist)
do
  mkdir -p ${redis_conf_dir}/${counter}
  srun -l \
    --nodes=1 \
    --nodelist=$hostname \
    --ntasks-per-node=1 \
    --cpus-per-task=${max_cpus_per_node} \
    --output=${redis_log_dir}/redis_${counter}.out \
    --error=${redis_log_dir}/redis_${counter}.err \
    --container-image="redis" \
    --container-mounts="${mounts}","${redis_conf_dir}/${counter}:/usr/local/etc/redis" \
      sh -c "sh ${base_path}/ndc/deduplication/redis/start_redis_instance_container" &
  sleep 1
  counter=$((counter+1))
done

# Wait for all instances to finish
echo "Started all redis instances, waiting for ${cluster_join_wait} seconds for nodes to initialize..."
sleep ${cluster_join_wait}

# Get cluster command
echo "Joining redis instances into a redis-cluster..."
join_cmd=$(cat ${redis_conf_dir}/*/host_port.out | tr '\n' ' ')

# Join the cluster
srun -l \
  --nodes=1 \
  --output=${redis_log_dir}/start_redis_cluster.out \
  --error=${redis_log_dir}/start_redis_cluster.err \
  --container-image="redis" \
    sh -c "redis-cli --cluster create ${join_cmd} --cluster-replicas 0 --cluster-yes"

echo "Successfully created the redis-cluster"

# Script to takedown the cluster
shutdown_cmd=$(cat ${redis_conf_dir}/*/job_step.out | tr '\n' ' ')
script_path=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "scancel ${shutdown_cmd}" > ${script_path}/shutdown_redis_cluster.sh
echo "Wrote redis-cluster shutdown script to ${script_path}/shutdown_redis_cluster.sh"
echo "Redis logs and configs are in ${data_dir}"
