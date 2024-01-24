# PEFT
This example uses the llama2 7b model and finetunes it on squad using the
LoRa method.

# Prerequisite
Applications:
* Helm
* Kubectl
Operators:
* kubeflow/training-operator

## NGC registry secret
```sh
kubectl create secret docker-registry ngc-registry --docker-server=nvcr.io --docker-username=\$oauthtoken --docker-password=<NGC KEY HERE>
```

## Checkpoint
To try this example, you need to prepare the llama2 7b Nemo checkpoint:
```sh
# Assumes huggingface llama2 checkpoint under LOCAL_WORKSPACE
LOCAL_WORKSPACE=<...>
NEMO_TRAINING_IMAGE=nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.11
docker run --rm -it -v $LOCAL_WORKSPACE:/workspace $NEMO_TRAINING_IMAGE \
  python /opt/NeMo/scripts/nlp_language_modeling/convert_hf_llama_to_nemo.py \
  --in-file /workspace/llama2_7b \
  --out-file /workspace/llama2_7b.nemo
```

Other prepared Nemo models:
* [GPT](https://huggingface.co/nvidia/nemotron-3-8b-base-4k)
* [T5](https://huggingface.co/nvidia/nemo-megatron-t5-3B) 

## PVC (PersistentVolumeClaim)
### Prerequisite
Any launcher command assumes your PVC already exists. Here is an example
of how to create a dynamic PV from a `StorageClass` setup by your cluster
admin.
```sh
PVC_NAME=nemo-workspace
STORAGE_CLASS=<...>
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC_NAME}
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: ${STORAGE_CLASS}
  resources:
    requests:
      # Requesting enough storage for a few experiments
      storage: 150Gi
EOF
```
Then copy `llama2_7b.nemo` into the PVC. Here is how you can do it with a
busybox container.
```sh
PVC_SUBPATH=peft-workspace
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: busybox
spec:
  containers:
  - name: busybox
    image: busybox
    command: ["sleep", "3600"]
    volumeMounts:
    - name: workspace
      mountPath: /workspace
      subPath: ${PVC_SUBPATH}
  volumes:
  - name: workspace
    persistentVolumeClaim:
      claimName: ${PVC_NAME}
EOF
```
Then copy it via `kubectl exec`
```sh
( cd $LOCAL_WORKSPACE; tar cf - llama2_7b.nemo | kubectl exec -i busybox -- tar xf - -C /workspace )
```
Then cleanup the busybox container:
```sh
kubectl delete pod/busybox
```
### Run
This will first create a `Job` that downloads squad for you and then
will create a `PytorchJob` that runs the finetuning workload.
```sh
DATA_DIR=/$PVC_SUBPATH RESTORE_FROM_PATH=/$PVC_SUBPATH/llama2_7b.nemo \
  HELM_RELEASE_NAME=llama-7b-peft-lora \
  examples/peft/lora_llama2_7b_2A6000_k8s.sh \
  cluster.volumes.persistentVolumeClaim.claimName=$PVC_NAME \
  cluster.volumes.persistentVolumeClaim.subPath=$PVC_SUBPATH \
  "peft.exp_manager.explicit_log_dir=/$PVC_SUBPATH/\${peft.run.model_train_name}/peft_\${peft.run.name}/results"
```

After this is finished, the experiment logs and checkpoint will be saved in the
PVC under the value of `peft.exp_manager.explicit_log_dir`, which in this case is
`/peft-workspace/llama2_7b/peft_*`.

### Cleanup
```sh
helm uninstall llama-7b-peft-lora
```

## HostPath
Instructions to be available soon!

## NFS
Instructions to be available soon!
