export FULL_OUTPUT_DIR=$HOME/$JOB_DIR
export LOGDIR=$FULL_OUTPUT_DIR/logs
export PROFILESDIR=$FULL_OUTPUT_DIR/profiles
echo $RUNSCRIPT
mkdir -p $LOGDIR
mkdir -p $PROFILESDIR


source /opt/conda/etc/profile.d/conda.sh;
conda activate rapids;

# Env vars
export RAPIDS_NO_INITIALIZE="1"
export CUDF_SPILL="1"


export LIBCUDF_CUFILE_POLICY=${LIBCUDF_CUFILE_POLICY:-ALWAYS}
export INTERFACE=ibp12s0
export PROTOCOL=ucx
echo $INTERFACE

# DEDUP CONFIGS
export NUM_FILES=-1

if [[ $SLURM_NODEID == 0 ]]; then
  echo "Starting scheduler"
  DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT=True \
  DASK_DISTRIBUTED__RMM__POOL_SIZE=1GB \
    dask scheduler \
      --scheduler-file $LOGDIR/scheduler.json \
      --protocol $PROTOCOL \
      --interface $INTERFACE >> $LOGDIR/scheduler.log 2>&1 &
fi
sleep 30

echo "Starting workers..."
dask-cuda-worker --scheduler-file $LOGDIR/scheduler.json --rmm-pool-size 72GiB --interface $INTERFACE --rmm-async >> $LOGDIR/worker_$HOSTNAME.log 2>&1 &

sleep 60
