source /opt/conda/etc/profile.d/conda.sh;
conda activate rapids;

# Env vars
export RAPIDS_NO_INITIALIZE="1"
export CUDF_SPILL="1"

export LIBCUDF_CUFILE_POLICY=${LIBCUDF_CUFILE_POLICY:-ALWAYS}

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
dask-cuda-worker --scheduler-file $LOGDIR/scheduler.json --rmm-pool-size $POOL_SIZE --interface $INTERFACE --rmm-async >> $LOGDIR/worker_$HOSTNAME.log 2>&1 &

sleep 60

if [[ $SLURM_NODEID == 0 ]]; then
  echo "Time Check: `date`"
  bash $RUNSCRIPT
  echo "Time Check: `date`"
  touch $LOGDIR/done.txt
fi


while [ ! -f $LOGDIR/done.txt ]
do
  sleep 15
done
