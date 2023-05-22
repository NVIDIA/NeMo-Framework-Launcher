# NeMo Framework Open Beta - Inference Container

# Table of contents
- [1. Support Matrix](#1-support-matrix)
- [2. Model Export](#2-model-export)
  * [2.1. GPT Export](#21-gpt-export)
    + [2.1.1. Common](#211-common)
    + [2.1.2. Slurm](#212-slurm)
    + [2.1.3. Base Command Platform](#213-base-command-platform)
- [3. Deploying the NeMo Framework Model](#3-deploying-the-nemo-framework-model)
  * [3.1. Run NVIDIA Triton Server with Generated Model Repository](#31-run-nvidia-triton-server-with-generated-model-repository)
  * [3.2. GPT Text Generation with Ensemble](#32-gpt-text-generation-with-ensemble)
  * [3.3. UL2 Checkpoint Deployment](#33-ul2-checkpoint-deployment)
- [4. Performance](#4-performance)
  * [4.1. GPT Results](#41-gpt-results)
    + [4.1.1. Inference Performance](#411-inference-performance)

### 1. Support Matrix
<a id="markdown-support-matrix" name="support-matrix"></a>

| Software                | Version          |
|-------------------------|------------------|
| NVIDIA Triton           | 2.33.0           |
| FasterTransformer       | v5.3+57704cd     |
| PyTorch                 | 2.0.1            |
| NeMo                    | 1.18.0+ddbe0f3   |
| PyTorch Lightning       | 1.9.4            |
| Hydra                   | 1.2.0            |
| NCCL                    | 2.14.3           |

### 2. Model Export
<a id="markdown-model-export" name="model-export"></a>

Model export is a prerequisite to enable deployment of the NeMo Framework model on the NVIDIA Triton
Inference Server with FasterTransformer Backend.

The export supports only GPT. You can checkout T5 and mT5 support
in FasterTransformer repository but it is limited to older versions of
NeMo and Megatron-LM.

#### 2.1. GPT Export
<a id="markdown-gpt-export" name="gpt-export"></a>

GPT model is evaluated with `lambada` task which results can be compared with results from evaluation stage.

The configuration used for the export needs to be specified in the
`conf/config.yaml` file, specifying the `export` parameter, which specifies the
file to use for export purposes. The `export` parameter must be inclueded in `stages`
to run the training pipeline export stage. The default value is set to
`gpt3/export_gpt3`, which can be found in `conf/export/gpt3/export_gpt3.yaml`. The
parameters can be modified to adapt different export and set of tests run on prepared Triton Model Repository.
For Base Command Platform, all these parameters should be overridden from the command line.

##### 2.1.1. Common
<a id="markdown-common" name="common"></a>
Other `run` parameters might be used to define the job-specific config:
```yaml
run:
  name: export_${.model_train_name}
  time_limit: "2:00:00"
  model_train_name: "gpt3_5b"
  training_dir: ${base_results_dir}/${.model_train_name}
  config_summary: tp${export.model.tensor_model_parallel_size}_pp${export.triton_deployment.pipeline_model_parallel_size}_${export.model.weight_data_type}_${export.triton_deployment.data_type}
  results_dir: ${base_results_dir}/${.model_train_name}/export_${.config_summary}
  model_type: "gpt3"
```

To specify which trained model checkpoint to use as source model
and parameters of conversion to the FasterTransformer format, use the `model` parameter:

```yaml
model:
  checkpoint_path: ${export.run.training_dir}/checkpoints
  # FT checkpoint will be saved in ${.triton_model_dir}/1/${.tensor_model_parallel_size}-gpu
  tensor_model_parallel_size: 8
  weight_data_type: fp16   # fp32|fp16
  processes: 16
  load_checkpoints_to_cpu: False
```

To specify the NVIDIA Triton Inference Server
[model directory](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#repository-layout) and
[FasterTransformer backend](https://github.com/triton-inference-server/fastertransformer_backend/blob/main/docs/gpt_guide.md#how-to-set-the-model-configuration) parameters,
use the `triton_deployment` parameter.

```yaml
triton_deployment:
  triton_model_dir: ${export.run.results_dir}/model_repo/${export.run.model_train_name}
  max_batch_size: 1
  pipeline_model_parallel_size: 1
  int8_mode: False
  enable_custom_all_reduce: False
  data_type: fp16  # fp32|fp16|bf16
```


##### 2.1.2. Slurm
<a id="markdown-slurm" name="slurm"></a>

Set configuration for a Slurm cluster in the `conf/cluster/bcm.yaml` file:

```yaml
partition: null
account: null
exclusive: True
gpus_per_task: null
gpus_per_node: 8
mem: 0
overcommit: False
job_name_prefix: "nemo-megatron-"
```

**Example:**

To run only the export pipeline, include `export` under `stages` in the `conf/config.yaml`:

```yaml
stages:
  - export
```

then run:
```
python3 main.py
```

##### 2.1.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the export stage on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The export scripts must be launched in a multi-node job.

To run the export pipeline to evaluate a 126M GPT model checkpoint stored in
`/mount/results/gpt3_126m/checkpoints`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py \
stages=[export] \
cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_gpt3 \
base_results_dir=/mount/results \
export.run.model_train_name=gpt3_126m \
export.model.tensor_model_parallel_size=2 \
export.triton_deployment.pipeline_model_parallel_size=1 \
>> /results/export_gpt3_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`.
The stdout and stderr outputs will also be redirected to the `/results/export_gpt3_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

# 3. Deploying the NeMo Framework Model

This section describes the deployment of the NeMo Framework model on the NVIDIA Triton
Inference Server with FasterTransformer Backend on both single and multiple
node environments.    NVIDIA Triton Inference Server supports many inference
scenarios, of which two most important are:
* Offline inference scenario - with a goal to maximize throughput regardless
    of the latency, usually achieved with increasing batch size and using server
    static batching feature.
* Online inference scenario - with a goal to maximize throughput within a given
    latency budget, usually achieved with small batch sizes and increasing
    concurrency requests to the server, using dynamic batching feature.


## 3.1. Run NVIDIA Triton Server with Generated Model Repository
<a id="markdown-run-nvidia-triton-server-with-selected-model-repository"
name="run-nvidia-triton-server-with-selected-model-repository"></a>

The inputs:
* NVIDIA Triton model repository with FasterTransformer checkpoint
     ready for inference at production.
* Docker image with NVIDIA Triton and FasterTransformer backend.

The outputs:
* Running NVIDIA Triton model instance serving model in cluster.

To run at slurm FasterTransformer backend, do the following:
```sh
srun \
     --nodes=<NUMBER OF NODES>\
     --partition=<SLURM PARITION>\
     --mpi <MPI MODE>\
     --container-image <NEMO_LAUNCHER INFERENCE CONTAINER>\
     --container-mounts <TRITON MODEL REPOSITORY>:<TRITON MODEL REPOSITORY> \
     bash -c "export CUDA_VISIBLE_DEVICES=<LIST OF CUDA DEVICES> && tritonserver --model-repository <TRITON MODEL REPOSITORY>"

```

Parameters:
* `NUMBER OF NODES`: Number of machines in cluster, which should be used to run inference.
* `SLURM PARTITION`: Slurm partition with DGX machines for inference.
* `MPI MODE`: FasterTransformer uses MPI for interprocess communication like `pmix` library.
* `NEMO_LAUNCHER INFERENCE CONTAINER`: Separate docker container streamlined for just inference.
* `TRITON MODEL REPOSITORY`: Triton model repository created by FasterTransformer export stage.
* `LIST OF CUDA DEVICES`: List of CUDA devices, which should be used by inference like `0,1,2,3`.

When you run inference, then number of machines and GPUs must match configuration
set during FasterTransformer export. You set tensor parallel (TP) and pipeline
parallel configuration (PP). This created wight files divided between GPUs and machines.
A tensor parallel configuration determines how many GPUs are used to process
one transformer layer. If you set TP to 16 but your cluster contains just 8 GPU
machines, then you need 2 nodes to run inference. FasterTransformer consumes all GPUs
accessible to Triton process. If you set TP to 4 but your machines contain 8 GPUs,
then you must hide some GPUs from the process. An environment variable
`CUDA_VISIVLE_DEVICES` can be used to list devices accessible to CUDA library
for a process, so you can use it to limit number of GPUs used by Triton instance.
The example configuration for 126m can't be run with tensor parallel set to 8
because head number in transformer layer must be divisible by tensor parallel
value.

Table below contains example configurations for DGX 8 GPU machines:

| TP   | PP   | #GPUs | #Nodes | CUDA DEVICES       |
| ---- | ---- | ----- | ------ | ------------------ |
| 1    | 1    | 1     | 1      | 0                  |
| 2    | 1    | 2     | 1      | 0,1                |
| 4    | 1    | 4     | 1      | 0,1,2,3            |
| 8    | 1    | 8     | 1      | Not necessary      |
| 8    | 2    | 16    | 2      | Not necessary      |
| 16   | 1    | 16    | 2      | Not necessary      |
| 8    | 3    | 24    | 3      | Not necessary      |
| 8    | 4    | 32    | 4      | Not necessary      |
| 16   | 2    | 32    | 4      | Not necessary      |



The script saves NVIDIA Triton logs so you can verify what happens when
FasterTransformer loads a checkpoint. The command above starts the server, so
that users can test it with other tools created later. You can use this
script to demo inference. The job does not stop on its own, if you don't stop it
manually, it will stop when the time limit is reached on the cluster.

FasterTransformer backend ignores missing files for weights and uses random
tensors in such a scenario. You should make sure that your NVIDIA Triton
instance is serving requests with real weights by inspecting logs.


If you notice warning about missing files, you should double check your model:

```
[WARNING] file /triton-model-repository/model_name/1/1-gpu/model.wpe.bin cannot be opened, loading model fails!
[WARNING] file /triton-model-repository/model_name/1/1-gpu/model.wte.bin cannot be opened, loading model fails!
[WARNING] file /triton-model-repository/model_name/1/1-gpu/model.final_layernorm.bias.bin cannot be opened, loading model fails!
[WARNING] file /triton-model-repository/model_name/1/1-gpu/model.final_layernorm.weight.bin cannot be opened, loading model fails!
```

## 3.2. GPT Text Generation with Ensemble

FasterTransformer for GPT implements a part of whole text generation application.

An
[ensemble](https://github.com/triton-inference-server/server/blob/main/docs/architecture.md#ensemble-models)
model represents a pipeline of models and the connection of input
and output tensors between those models. Ensemble models are intended to be used
to encapsulate a procedure that involves multiple models, such as
"data preprocessing -> inference -> data postprocessing".
Using ensemble models for this purpose can avoid the overhead of
transferring intermediate tensors and minimize the number of requests
that must be sent to Triton.


A text generation example for GPT is implemented as ensemble example:
[gpt](https://github.com/triton-inference-server/fastertransformer_backend/tree/main/all_models/gpt)
folder. This example contains four folders:
* `ensemble`: ensemble definition folder.
* `fastertransformer`: FasterTransformer backend folder.
* `postprocessing`: Detokeniser to generate text.
* `preprocessing`: Tokenizer to translate text into token IDs.

You should replace your `fastertransformer` folder with model store generated
by FasterTransformer export described above. The ensemble expects a `model name`
to be `fastertransformer` so make sure that your generated configuration uses
such `model name`.

The inference container doesn't contain PyTorch so you need to install dependencies
for ensemble. You can start you compute node for Triton in interactive mode to access terminal directly.


Inside machine running container for Triton Inference server install PyTorch and regex packages:

```
pip install torch regex
```

Execute Triton inference server like described above in point 6.1. You can demonize process.

```
CUDA_VISIBLE_DEVICES=0 mpirun -n 1 --allow-run-as-root tritonserver --model-store /your/folders/fastertransformer_backend/all_models/gpt &
```

Install Triton client:

```
pip install tritonclient[all]
```
Execute `end_to_end_test.py` example:

```
python3 /your/folders/fastertransformer_backend/tools/end_to_end_test.py
```

The `end_to_end_test.py` script contains a string examples, which you can replace with your text.



## 3.3. UL2 Checkpoint Deployment

You can deploy UL2 T5 checkpoints using
[readme](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/t5_guide.md#running-ul2-on-fastertransformer-pytorch-op)
created by FasterTransformer.

You can use huggingface t5 conversion script see below:

```
python3 FasterTransformer/examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
        -in_file <UL2 checkpoint folder from training> \
        -saved_dir <FasterTransformer destination folder> \
        -inference_tensor_para_size <tensor parallel size> \
        -weight_data_type <data type>
```

Triton FasterTransformer backend repo contains configuration example [config.pbtxt](https://github.com/triton-inference-server/fastertransformer_backend/blob/main/all_models/t5/fastertransformer/config.pbtxt).

You can use Triton configuration script
[prepare\_triton\_model\_config.py](nemo_megatron/collections/export_scripts/prepare_triton_model_config.py)
to modify config.pbtxt to match
configuration of your UL2 checkpoint and your cluster configuration.


# 4. Performance
<a id="markdown-performance" name="performance"></a>

## 4.1. GPT Results
<a id="markdown-gpt-results" name="gpt-results"></a>

### 4.1.1. Inference Performance
<a id="markdown-inference-performance" name="inference-performance"></a>

Inference performance was measured for NVIDIA DGX SuperPOD for `batch size = 1`.


<img src="img/infer_model_size_gpt3.svg"/>

| GPT Model size | Use Case | Input Sequence Length (Tokens) | Output Sequence Length (tokens) | Average latency [ms]           | TP | PP | GPUs |
|----------------|----------------------------|--------------------------------|----|----|----|----|------|
| 43B            | Question Answering | 60 | 20  |                            ??? |  32 |  1 |   32 |
| 43B           | Intent Classification | 225 | 20  |                            ??? |  32 |  1 |   32 |
| 43B           |Translation | 200 | 200  |                            ??? |  32 |  1 |   32 |
