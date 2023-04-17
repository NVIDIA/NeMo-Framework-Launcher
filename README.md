# NeMo Megatron

## Open Beta

Scripts and code to provide end-to-end data preparation and training for
NeMo-Megatron.

The most recent version of the README can be found
at [https://ngc.nvidia.com/containers/ea-bignlp:bignlp-training](https://ngc.nvidia.com/containers/ea-bignlp:bignlp-training).

## Table of contents

- [NeMo Megatron](#nemo-megatron)
    - [Open Beta](#open-beta)
    - [Table of contents](#table-of-contents)
    - [1. Model Overview](#1-model-overview)
        - [1.1. Vision Transformer (ViT)](#11-vision-transformer-vit)
        - [1.2. CLIP](#12-clip)
        - [1.3. Stable Diffusion](#13-stable-diffusion)
        - [1.4. Instruct Pix2Pix](#14-instruct-pix2pix)
        - [1.5. DreamBooth](#15-dreambooth)
    - [2. Feature Matrix](#2-feature-matrix)
        - [2.1. ViT Models](#21-vit-models)
        - [2.2 CLIP Models](#22-clip-models)
        - [2.3. Stable Diffusion](#23-stable-diffusion)
        - [2.4. Instruct Pix2Pix / DreamBooth Models](#24-instruct-pix2pix--dreambooth-models)
    - [3. Setup](#3-setup)
        - [3.1. Support Matrix](#31-support-matrix)
    - [4. Cloud Service Providers](#4-cloud-service-providers)
        - [4.1. Cluster Bring-Up](#41-cluster-bring-up)
            - [4.1.1. Common](#411-common)
            - [4.1.2. OCI](#412-oci)
            - [4.1.3. AWS](#413-aws)
        - [4.2. Cluster Validation](#42-cluster-validation)
            - [4.2.1. Validation Script Usage](#421-validation-script-usage)
            - [4.2.2 Running tests manually](#422-running-tests-manually)
        - [4.3. Config Modifications](#43-config-modifications)
            - [4.3.1 Set NCCL Topology](#431-set-nccl-topology)
            - [4.3.2 Environment Variables](#432-environment-variables)
                - [4.3.2.1 Azure Variables](#4321-azure-variables)
                - [4.3.2.2 AWS Variables](#4322-aws-variables)
    - [5. Quick Start Guide](#5-quick-start-guide)
        - [5.1. Getting Started with Multimodal NeMo Megatron](#51-getting-started-with-multimodal-nemo-megatron)
            - [5.1.1. Prepare Environment](#511-prepare-environment)
                - [5.1.1.1. Slurm](#5111-slurm)
                - [5.1.1.2. Base Command Platform](#5112-base-command-platform)
            - [5.1.2. Configure and Customize Pipeline](#512-configure-and-customize-pipeline)
                - [5.1.2.1. Cluster Configurations](#5121-cluster-configurations)
                - [5.1.2.2. Pipeline Configurations](#5122-pipeline-configurations)
                - [5.1.2.3. Environment Variables Configurations](#5123-environment-variables-configurations)
                - [5.1.2.4. NUMA Mapping Configurations](#5124-numa-mapping-configurations)
            - [5.1.3. Launch Pipeline](#513-launch-pipeline)
            - [5.1.4. Example: Pre-train Stable Diffusion 860M Model for 10 Epochs with Resolution 256](#514-example-pre-train-stable-diffusion-860m-model-for-10-epochs-with-resolution-256)
        - [5.2. Data Preparation](#52-data-preparation)
            - [5.2.1 ImageNet](#521-imagenet)
                - [5.2.1.1 ImageNet 1k](#5211-imagenet-1k)
                - [5.2.1.2 ImageNet 21k](#5212-imagenet-21k)
            - [5.2.2 Multimodal Datasets](#522-multimodal-datasets)
                - [5.2.2.1 Overview](#5221-overview)
                - [5.2.2.2 Running the Pipeline](#5222-running-the-pipeline)
                - [5.2.2.3 Configuration for Precaching](#5223-configuration-for-precaching)
                    - [5.2.2.3.1 General Format](#52231-general-format)
                    - [5.2.2.3.2 Precaching Config](#52232-precaching-config)
                    - [5.2.2.3.3 Resume Precaching (Advanced)](#52233-resume-precaching-advanced)
                    - [5.2.2.3.4 Known Issue](#52234-known-issue)
            - [5.2.3 Instruct Pix2Pix](#523-instruct-pix2pix)
            - [5.2.4 MSCOCO for FID Evaluation](#524-mscoco-for-fid-evaluation)
                - [5.2.4.1 Download and Setup](#5241-download-and-setup)
                - [5.2.4.2 Preprocess Images and Captions](#5242-preprocess-images-and-captions)
        - [5.3. Model Training](#53-model-training)
            - [5.3.1. Vision Transformer Training](#531-vision-transformer-training)
            - [5.3.2. CLIP Training](#532-clip-training)
            - [5.3.3. Stable Diffusion Training](#533-stable-diffusion-training)
            - [5.3.4. Instruct Pix2Pix Training](#534-instruct-pix2pix-training)
            - [5.3.5. DreamBooth Training](#535-dreambooth-training)
        - [5.4. Checkpoint Conversion](#54-checkpoint-conversion)
        - [5.5. Model Fine-tuning](#55-model-fine-tuning)
            - [5.5.1. Vision Transformer Fine-tuning](#551-vision-transformer-fine-tuning)
        - [5.6. Model Evaluation](#56-model-evaluation)
            - [5.6.1. Vision Transformer Evaluation](#561-vision-transformer-evaluation)
            - [5.6.2. CLIP Evaluation](#562-clip-evaluation)
            - [5.6.3. Stable Diffusion Evaluation](#563-stable-diffusion-evaluation)
        - [5.7. Model Inference (in NeMo Framework)](#57-model-inference-in-nemo-framework)
            - [5.7.1. Vision Transformer Inference (in NeMo Framework)](#571-vision-transformer-inference-in-nemo-framework)
            - [5.7.2. CLIP Inference (in NeMo Framework)](#572-clip-inference-in-nemo-framework)
            - [5.7.3. Stable Diffusion Inference (in NeMo Framework)](#573-stable-diffusion-inference-in-nemo-framework)
            - [5.7.4. Instruct Pix2Pix Inference (in NeMo Framework)](#574-instruct-pix2pix-inference-in-nemo-framework)
            - [5.7.5. DreamBooth Inference (in NeMo Framework)](#575-dreambooth-inference-in-nemo-framework)
        - [5.8. Model Export](#58-model-export)
            - [5.8.1. Vision Transformer Export](#581-vision-transformer-export)
            - [5.8.2. CLIP Export](#582-clip-export)
            - [5.8.3. Stable Diffusion Export](#583-stable-diffusion-export)
            - [5.8.4. Instruct Pix2pix Export](#584-instruct-pix2pix-export)
            - [5.8.5. DreamBooth Export](#585-dreambooth-export)
    - [6. Deploying the NeMo Megatron Model](#6-deploying-the-nemo-megatron-model)
        - [6.1 Setup](#61-setup)
        - [6.2 Start NVIDIA Triton Inference Server](#62-start-nvidia-triton-inference-server)
            - [6.2.1 Stable Diffusion, Dreambooth](#621-stable-diffusion-dreambooth)
            - [6.2.2 Instruct Pix2Pix](#622-instruct-pix2pix)
            - [6.2.3 Vision Transformer](#623-vision-transformer)
            - [6.2.4 CLIP](#624-clip)
        - [6.3 Query NVIDIA Triton Inference Server](#63-query-nvidia-triton-inference-server)
            - [6.3.1 Stable Diffusion and Dreambooth](#631-stable-diffusion-and-dreambooth)
            - [6.3.2 Instruct Pix2Pix](#632-instruct-pix2pix)
    - [7. Performance](#7-performance)
        - [7.1. Vision Transformer Results](#71-vision-transformer-results)
            - [7.1.1. Training Accuracy Results](#711-training-accuracy-results)
            - [7.1.2. Training Performance Results](#712-training-performance-results)
            - [7.1.3. Inference Performance Results](#713-inference-performance-results)
        - [7.2. CLIP Results](#72-clip-results)
            - [7.2.1. Training Accuracy Results](#721-training-accuracy-results)
            - [7.2.2. Training Performance Results](#722-training-performance-results)
            - [7.2.3. Inference Performance Results](#723-inference-performance-results)
        - [7.3. Stable Diffusion Results](#73-stable-diffusion-results)
            - [7.3.1. Training Accuracy Results](#731-training-accuracy-results)
            - [7.3.2. Training Performance Results](#732-training-performance-results)
            - [7.3.3. Inference Performance Results](#733-inference-performance-results)
        - [7.4. Instruct Pix2Pix Results](#74-instruct-pix2pix-results)
            - [7.4.1. Training Quality Results](#741-training-quality-results)
            - [7.4.2. Inference Performance Results](#742-inference-performance-results)
        - [7.5. DreamBooth Results](#75-dreambooth-results)
            - [7.5.1. Training Quality Results](#751-training-quality-results)
            - [7.5.2. Inference Performance Results](#752-inference-performance-results)
    - [8. Changelog](#8-changelog)
    - [9. Known Issues](#9-known-issues)

<!-- /TOC -->

## 1. Model Overview

The Multimodal NeMo Megatron is a powerful extension of the NeMo framework, specifically designed for developers who aim
to efficiently train and scale multimodal models. With Multimodal NeMo Megatron, you can effortlessly train various
variants of multimodal models, such as CLIP, Stable Diffusion and more. This powerful tool is capable of
scaling your models to multiple nodes on NVIDIA DGX SuperPOD deployments.

The deep learning (DL) software stack is meticulously optimized for DGX SuperPOD configurations, utilizing NVIDIA's
InfiniBand technology to deliver efficient on-premises computing for training and inference of complex workloads.

The Multimodal NeMo Megatron utilizes model parallelism techniques to efficiently train large models that cannot fit
within the memory of a single GPU. During the training process, both tensor (intra-layer) and pipeline (inter-layer)
model parallelism are employed. Tensor model parallelism distributes individual transformer layers across multiple
devices, while pipeline model parallelism allocates different layers of a model to separate devices. For a more in-depth
understanding, please refer to [this paper](https://arxiv.org/pdf/2104.04473.pdf). We are currently in the process of
incorporating this feature into all our models. As of now, Tensor Parallelism is
available in both **Vision Transformer** and **CLIP** models.

### 1.1. Vision Transformer (ViT)

The Vision Transformer, commonly referred to as ViT [[Paper]](https://arxiv.org/pdf/2010.11929v2.pdf), is a foundation
model for image classification tasks in Multimodal
NeMo Megatron. It
leverages a Transformer-like architecture to process image patches, rather than relying on traditional convolutional
neural networks. In the ViT, an image is divided into fixed-size patches (usually 14x14 or 16x16), which are then
linearly embedded and augmented
with position embeddings. The resulting sequence of vectors is fed into a standard Transformer encoder. To enable
classification, a learnable "classification token" is added to the sequence.

### 1.2. CLIP

Contrastive Language-Image Pre-training (CLIP) [[Paper]](https://arxiv.org/pdf/2103.00020.pdf) offers an efficient
method for learning image representations using natural language supervision. In essence, CLIP trains both an image
encoder and a text encoder from scratch. The goal is to predict the correct pairings of a batch of (image, text)
training examples by jointly training these encoders.

During pre-training, the model is designed to predict which images and texts form a semantically coherent pair by
maximizing the similarity between the correct (image, text) pairs while minimizing the similarity between incorrect
pairs. This contrastive learning approach ensures that CLIP learns meaningful and contextually rich representations of
both visual and textual data.

Upon completion of the pre-training phase, CLIP modules can be fine-tuned for specialized downstream tasks or directly
employed for zero-shot learning. For instance, the learned text encoder generates high-level representations by
embedding captions in **Stable Diffusion**. This robust approach facilitates seamless image and text representation
learning and has demonstrated exceptional effectiveness across a diverse range of applications.

### 1.3. Stable Diffusion

Stable Diffusion (SD) [[Paper]](https://arxiv.org/pdf/2112.10752v2.pdf) is a powerful generative model that can
produce high-quality images based on textual descriptions. By decomposing the image formation process into a sequential
application of denoising autoencoders, diffusion models (DMs) have achieved state-of-the-art synthesis results on image
data and beyond. However, due to their direct operation in pixel space, optimization of powerful DMs is computationally
expensive and can consume hundreds of GPU days. To address this challenge, the SD model is applied in the latent space
of powerful pretrained autoencoders. This enables DM training on limited computational resources while retaining their
quality and flexibility, greatly boosting visual fidelity.

The SD model also introduces cross-attention layers into the model architecture, allowing it to turn diffusion models
into powerful and flexible generators for general conditioning inputs such as text or bounding boxes. As a result, the
SD model achieves a new state of the art for image inpainting and highly competitive performance on various tasks,
including unconditional image generation, semantic scene synthesis, and super-resolution. Additionally, the SD model
significantly reduces computational requirements compared to pixel-based DMs, making it an attractive solution for a
wide range of applications.

### 1.4. Instruct Pix2Pix

[Instruct Pix2Pix](https://www.timothybrooks.com/instruct-pix2pix/) introduces a method for editing images based on
human-written instructions. Given an input image and a textual directive, the model follows these instructions to modify
the image accordingly.

Multimodal NeMo Megatron offers a training pipeline for conditional diffusion models using the edit dataset.
Additionally, we provide a tool that generates modified images based on user-written instructions during the inference
process.

### 1.5. DreamBooth

Dreambooth is a solution to personalize large diffusion models like Stable Diffusion, which are powerful but lack the
ability to mimic subjects of a given reference set. With Dreambooth, you only need a few images of a specific subject to
fine-tune a pretrained text-to-image model, so that it learns to bind a unique identifier with a special subject. This
unique identifier can then be used to synthesize fully-novel photorealistic images of the subject contextualized in
different scenes.

Dreambooth provides a new prior preservation loss, which enables synthesizing the subject in diverse scenes, poses,
views, and lighting conditions that do not appear in the reference images. With this new approach, Dreambooth achieves
several previously-unassailable tasks, including subject recontextualization, text-guided view synthesis, appearance
modification, and artistic rendering, while still preserving the subject's key features.

## 2. Feature Matrix

### 2.1. ViT Models

| Feature                  | Training                                                 | Inference                                                                                                                                     |
|--------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Data parallelism         | Yes                                                      | N/A                                                                                                                                           |
| Tensor parallelism       | Yes                                                      | Yes                                                                                                                                           |
| Pipeline parallelism     | No                                                       | No                                                                                                                                            |
| Sequence parallelism     | No                                                       | No                                                                                                                                            |
| Activation checkpointing | Yes (Uniform or Block)                                   | No                                                                                                                                            |
| FP32/TF32                | Yes                                                      | Yes (FP16 enabled by default)                                                                                                                 |
| AMP/FP16                 | No                                                       | Yes                                                                                                                                           |
| AMP/BF16                 | Yes                                                      | No                                                                                                                                            |
| BF16 O2                  | Yes                                                      | No                                                                                                                                            |
| TransformerEngine/FP8    | No                                                       | No                                                                                                                                            |
| Multi-GPU                | Yes                                                      | Yes                                                                                                                                           |
| Multi-Node               | Yes                                                      | Yes                                                                                                                                           |
| Inference deployment     | N/A                                                      | [NVIDIA Triton supported](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton) |
| SW stack support         | Slurm DeepOps/Base Command Manager/Base Command Platform | Slurm DeepOps/Base Command Manager/Base Command Platform                                                                                      |
| NVfuser                  | No                                                       | N/A                                                                                                                                           |
| Distributed Optimizer    | No                                                       | N/A                                                                                                                                           |
| TorchInductor            | No                                                       | N/A                                                                                                                                           |
| Flash Attention          | No                                                       | N/A                                                                                                                                           |

### 2.2 CLIP Models

| Feature                  | Training                                                 | Inference                                                                                                                                     |
|--------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Data parallelism         | Yes                                                      | N/A                                                                                                                                           |
| Tensor parallelism       | Yes                                                      | Yes                                                                                                                                           |
| Pipeline parallelism     | No                                                       | No                                                                                                                                            |
| Sequence parallelism     | No                                                       | No                                                                                                                                            |
| Activation checkpointing | Yes (Uniform or Block)                                   | No                                                                                                                                            |
| FP32/TF32                | Yes                                                      | Yes (FP16 enabled by default)                                                                                                                 |
| AMP/FP16                 | No                                                       | Yes                                                                                                                                           |
| AMP/BF16                 | Yes                                                      | No                                                                                                                                            |
| BF16 O2                  | Yes                                                      | No                                                                                                                                            |
| TransformerEngine/FP8    | No                                                       | No                                                                                                                                            |
| Multi-GPU                | Yes                                                      | Yes                                                                                                                                           |
| Multi-Node               | Yes                                                      | Yes                                                                                                                                           |
| Inference deployment     | N/A                                                      | [NVIDIA Triton supported](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton) |
| SW stack support         | Slurm DeepOps/Base Command Manager/Base Command Platform | Slurm DeepOps/Base Command Manager/Base Command Platform                                                                                      |
| NVfuser                  | No                                                       | N/A                                                                                                                                           |
| Distributed Optimizer    | No                                                       | N/A                                                                                                                                           |
| TorchInductor            | No                                                       | N/A                                                                                                                                           |
| Flash Attention          | No                                                       | N/A                                                                                                                                           |

### 2.3. Stable Diffusion

| Feature                  | Training                                                 | Inference                                                                                                                                     |
|--------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Data parallelism         | Yes                                                      | N/A                                                                                                                                           |
| Tensor parallelism       | No                                                       | No                                                                                                                                            |
| Pipeline parallelism     | No                                                       | No                                                                                                                                            |
| Sequence parallelism     | No                                                       | No                                                                                                                                            |
| Activation checkpointing | No                                                       | No                                                                                                                                            |
| FP32/TF32                | Yes                                                      | Yes (FP16 enabled by default)                                                                                                                 |
| AMP/FP16                 | Yes                                                      | Yes                                                                                                                                           |
| AMP/BF16                 | No                                                       | No                                                                                                                                            |
| BF16 O2                  | No                                                       | No                                                                                                                                            |
| TransformerEngine/FP8    | No                                                       | No                                                                                                                                            |
| Multi-GPU                | Yes                                                      | Yes                                                                                                                                           |
| Multi-Node               | Yes                                                      | Yes                                                                                                                                           |
| Inference deployment     | N/A                                                      | [NVIDIA Triton supported](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton) |
| SW stack support         | Slurm DeepOps/Base Command Manager/Base Command Platform | Slurm DeepOps/Base Command Manager/Base Command Platform                                                                                      |
| NVfuser                  | No                                                       | N/A                                                                                                                                           |
| Distributed Optimizer    | No                                                       | N/A                                                                                                                                           |
| TorchInductor            | Yes                                                      | N/A                                                                                                                                           |
| Flash Attention          | Yes                                                      | N/A                                                                                                                                           |

### 2.4. Instruct Pix2Pix / DreamBooth Models

| Feature                  | Training                                                 | Inference                                                                                                                                     |
|--------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Data parallelism         | Yes                                                      | N/A                                                                                                                                           |
| Tensor parallelism       | No                                                       | No                                                                                                                                            |
| Pipeline parallelism     | No                                                       | No                                                                                                                                            |
| Sequence parallelism     | No                                                       | No                                                                                                                                            |
| Activation checkpointing | No                                                       | No                                                                                                                                            |
| FP32/TF32                | Yes                                                      | Yes (FP16 enabled by default)                                                                                                                 |
| AMP/FP16                 | Yes                                                      | Yes                                                                                                                                           |
| AMP/BF16                 | Yes                                                      | No                                                                                                                                            |
| BF16 O2                  | No                                                       | No                                                                                                                                            |
| TransformerEngine/FP8    | No                                                       | No                                                                                                                                            |
| Multi-GPU                | Yes                                                      | Yes                                                                                                                                           |
| Multi-Node               | Yes                                                      | Yes                                                                                                                                           |
| Inference deployment     | N/A                                                      | [NVIDIA Triton supported](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton) |
| SW stack support         | Slurm DeepOps/Base Command Manager/Base Command Platform | Slurm DeepOps/Base Command Manager/Base Command Platform                                                                                      |
| NVfuser                  | No                                                       | N/A                                                                                                                                           |
| Distributed Optimizer    | No                                                       | N/A                                                                                                                                           |
| TorchInductor            | Yes                                                      | N/A                                                                                                                                           |
| Flash Attention          | Yes                                                      | N/A                                                                                                                                           |

## 3. Setup

### 3.1. Support Matrix

| Software             | EA                 |
|----------------------|--------------------|
| NVIDIA Triton        | 2.31.0             |
| FasterTransformer    | v5.3+4402759e      |
| PyTorch              | 1.14.0a0+44dac51   |
| NeMo                 | 1.17.0+<FINAL-SHA> |
| PyTorch Lightning    | 1.9.4              |
| Hydra                | 1.2.0              |
| CUDA                 | NVIDIA CUDA 12.0   |
| cuBLAS               | 12.0.2.224         |
| cuDNN                | 8.7.0.84           |
| NCCL                 | 2.16.5             |
| Container OS         | Ubuntu 20.04       |
| rdma-core            | 36.0               |
| GDRcopy              | 2.3                |
| HPC-X                | 2.13               |
| Base Command Manager | 1.0.0              |
| DeepOps              | 21.06              |

## 4. Cloud Service Providers

### 4.1. Cluster Bring-Up

#### 4.1.1. Common

To set up a Slurm cluster for NeMo Megatron, we recommend using [Nephele](https://github.com/nvidia/nephele). This
cluster deployment tool has been tested on Azure, AWS, and Oracle Cloud.
We recommend hosting Nephele on a new VM instance in the CSP of your choice. To get started:

- Clone the Nephele repo
- Install the dependencies
- Modify `nephele.conf`
    - Add your CSP credentials
    - Change `REPLICAS_x8a100` to the number of nodes in your desired cluster

You can then run `./nephele init` and `./nephele create`.

We also recommend mounting an external persistent NFS once the cluster is up and running (ensure it is mounted on all
nodes) and using this to configure and run NeMo Megatron.

The above steps apply to all CSPs, including Azure, AWS, and OCI.
Some modifications are necessary for OCI and AWS and are detailed below.
Note that for OCI, a custom image must be imported, which should be done before running `./nephele create`.

#### 4.1.2. OCI

NeMo Megatron supports running training and inference containers on OCI. For detail orchestration scripts, reach out
to [oci_nm@nvidia.com](mailto:oci_nm@nvidia.com)

#### 4.1.3. AWS

To launch jobs on AWS, the EFA driver and NCCL plugin first need to be installed on top of the training container.
We recommend building a new container image with Docker, then creating an Enroot image.

On the scheduler node:

- Install Docker
- Build the image with EFA drivers and NCCL plugin from `csp_tools/aws/Dockerfile`
- Run this command on the Docker image to create an Enroot image:

```
    enroot import --output nemo_megatron_training.sqsh dockerd://<image name>:<tag>
```

- Move the `.sqsh` file to the root of NeMo-Megatron-Launcher
- Set the container path in `launcher_scripts/conf/config.yaml` to the new Enroot image:

```
container: /path/to/nemo_megatron_launcher/nemo_megatron_training.sqsh
```

### 4.2. Cluster Validation

Before running the cluster validation script, ensure your NGC credentials have been added
to `~/.config/enroot/.credentials` on all nodes.

The cluster validation script at `csp_tools/<csp>/cluster_validation.sh` will run GPU diagnostics and test NCCL
node-to-node bus bandwidth.
The logs from these tests will be stored at `results/cluster_validation`. The script will list any nodes that fail these
tests.
These nodes should be replaced or restarted through the CSP UI.

#### 4.2.1. Validation Script Usage

The script has 3 required parameters:

- `--nodes`: the number of nodes
- `--nodelist`: the list of node names
- `--partition`: the Slurm partition the nodes are assigned to

The values for these parameters should be in the same format that is found in `sinfo`.
With the following example:

```
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
x8a100       up   infinite      8   idle x8a100-[0000-0007]
```

To test all 8 idle nodes, the script would be run as:

```
bash cluster_validation.sh --nodes=8 --nodelist=x8a100-[0000-0007] --partition=x8a100
```

By default, the script will run both the GPU diagnostics and the NCCL test. You can choose to run only one or the other
by specifying:

- `--dcgm`: run GPU diagnostics only
- `--nccl`: run NCCL test only

See `bash cluster_validation.sh -h` for more information.

#### 4.2.2 Running tests manually

The `cluster_validation.sh` script is essentially a wrapper of the 2 Slurm job scripts in the CSP directories. If you
prefer, you can run these jobs manually.
Make sure to use the Slurm job script in your corresponding CSP's path (`csp_tools/<csp>/dcgmi_diag.sh`
and `csp_tools/<csp>/nccl.sh`)

For the GPU diagnostics job, provide these arguments when submitting the job to Slurm:

```
sbatch -p <partition> -w <node list> -o <job log file> dcgmi_diag.sh
```

For the NCCL test job, `cluster_validation.sh` performs a pair-wise sweep of the nodes, as this is a sufficient test,
but you can test with a different number of nodes if desired.

First build the test binaries:

```
sbatch -N 1 build-nccl-tests.sh
```

Then, to run a 2-node `all_reduce_perf` job:

```
sbatch -w <node 1>,<node 2> -o <job log file> nccl.sh
```

To run the job with more nodes, simply add the node names to the `-w` flag in the same comma-separated list format.

### 4.3. Config Modifications

Before launching jobs some changes to the config must be made.

#### 4.3.1 Set NCCL Topology

The NCCL topology file is unique for each CSP, and can be found in their corresponding
folders (`csp_tools/<csp>/topo.xml`)

In `launcher_scripts/conf/config.yaml`, mount the directory containing the topology file:

```
container_mounts:
  - /path/to/nemo_megatron_laujncher/csp_tools/<csp>/:/nccl
```

Then set the path of the file in the container:

```
env_vars:
    NCCL_TOPO_FILE: /nccl/topo.xml
```

#### 4.3.2 Environment Variables

##### 4.3.2.1 Azure Variables

Set these environment variables in `config.yaml` (these are only needed for Azure):

```
env_vars:
  UCX_IB_PCI_RELAXED_ORDERING: auto
  NCCL_IB_PCI_RELAXED_ORDERING: 2
  NCCL_IB_TIMEOUT: 22
  NCCL_DEBUG: INFO
```

##### 4.3.2.2 AWS Variables

AWS recommends setting the following flag to avoid data corruption:

```
env_vars:
  NCCL_PROTO: simple
```

Setting this flag reduces training throughput by roughly 2%.

## 5. Quick Start Guide

### 5.1. Getting Started with Multimodal NeMo Megatron

#### 5.1.1. Prepare Environment

<!--
The whole solution uses a set of Docker containers executed at the Slurm
cluster using the pyxis plug-in Base Command Platform cluster. The training
container also includes conversion scripts and NVIDIA Triton Model Navigator.
The inference container is just the NVIDIA Triton Inference Server with the
FasterTransformer backend installed.    For Base Command Platform, the NeMo Megatron
scripts repository (bcp branch) will be part of the container image. It is
recommended to create a nemo_megatron_ws_scripts_<username> workspace in your ace and
copy the nemo_megatron_launcher directory there    either from the container image or
from git clone of the above repository if you have access.    Install the NeMo Megatron
scripts dependencies on the head node of your cluster. Base Command Platform
clusters do not have a head login node. We're currently running these scripts
on a DGX node in the Base Command Platform cluster. Once the cluster has
cpu-only nodes then we can use those. Till then we can run on DGX node or in a
local conda environment.    To be able to call the necessary scripts from the
login node on a cluster, some packages must be installed there using the
requirements.txt file:
```
pip install -r requirements.txt
```
You can use virtualenv to prevent polluting your head node environment for
other Python projects. If your Slurm configuration environment lacks pip, then
you can use get_pip.py with just python3.
 -->
**NOTE:** Ensure the high-speed filesystem is mounted on the job submission
node(s) at the same path as on the compute nodes.

The whole solution uses a set of Docker containers executed on a Slurm
cluster (using the [pyxis](https://github.com/NVIDIA/pyxis) plug-in) or
a Base Command Platform cluster. The training container also includes
conversion scripts. The inference container
comprises the NVIDIA Triton Inference Server with the FasterTransformer
backend installed.

##### 5.1.1.1. Slurm

The NeMo Megatron codebase is included as part of the training container. To
copy it to a local directory in the cluster, it needs to be extracted from the
container. To copy the code to a directory named /path/to/local/dir the
following command can be executed. The NeMo Megatron repository for
Slurm has been verified on both Slurm-based DeepOps clusters as well as Base
Command Manager.

```
srun -p [partition] -N 1 --container-mounts=/path/to/local/dir:/workspace/mount_dir --container-image=[container_tag] bash -c "cp -r /opt/NeMo-Megatron-Launcher/launcher_scripts /workspace/mount_dir/"
```

Install the NeMo Megatron scripts dependencies on the head node of the cluster:

```
pip install -r requirements.txt
```

You can use virtualenv to prevent polluting your head node environment for
other Python projects. If your configuration lacks pip, then you can
install pip using use [get_pip.py](https://github.com/pypa/get-pip) with just `python3`.

##### 5.1.1.2. Base Command Platform

The NeMo Megatron Launcher codebase is included as part of the training
container. Before starting, set up the ngc cli and configuration as described
in the Base Command Platform User Guide. In this guide, we will mainly
use two Base Command Platform workspaces, one for storing the training dataset,
and another for storing the results, checkpoints and logs. Therefore, start by
creating these workspaces (e.g. `nemo_megatron_data_ws` and `nemo_megatron_results_ws`). See
the Base Command Platform User Guide for how to create and work with Base
Command Platform workspaces.

#### 5.1.2. Configure and Customize Pipeline

This section provides instructions for configuring and customizing the pipeline in NeMo-Megatron-Launcher. It covers
four areas: cluster configurations, pipeline configurations, environment variables configurations, and NUMA mapping
configurations.

##### 5.1.2.1. Cluster Configurations

The first parameter that must be set is the `launcher_scripts_path` parameter inside the
`conf/config.yaml` file. This parameter must point to the absolute path where
the `launcher_scripts` folder (pulled from the container) is stored in the file system.    
Additionally, if using a Slurm based
cluster, the config file in the subfolder of `conf/cluster/bcm.yaml` has the
parameters to set the generic cluster related information, such as the
`partition` or `account` parameters. Tailor the cluster configuration below to match your cluster setup.

```yaml
partition: null
account: null
exclusive: True
gpus_per_task: null
gpus_per_node: 8
mem: 0
overcommit: False
job_name_prefix: "nemo-multimodal-"
```

**Slurm**: The `launcher_scripts_path` parameter will automatically be mounted to the
container at the same path as in the local file system. Any additional
directories that should be mounted must be specified using the
`container_mounts` parameter. If the paths contain the colon character (`:`),
the code will assume both the source and destination paths are provided.
Otherwise, the given paths will be mounted to the same path inside the container.
The `data_dir` parameter can also be
modified to point to where the dataset will be loaded from or saved (an existing folder). The
`base_results_dir` can also be modified to point to where the results,
checkpoints and logs will be stored. These last two parameters will be
automatically mounted into the container. The parameters `cluster` and `cluster_type`
must be set to `bcm` for all the tasks.

**Base Command Platform**: The `launcher_scripts_path` should be set to
`/opt/NeMo-Megatron-Launcher/launcher_scripts` , which is the default location where the scripts
are located inside the container. The `data_dir` parameter can also be
modified to point to where the dataset will be loaded from or saved. The
`base_results_dir` can also be modified to point to where the results,
checkpoints and logs will be stored. In the case of Base Command Platform, we recommend
that `data_dir` points to one of the workspaces, and `base_results_dir`
points to the other. They should both be mounted in read and write (RW)
mode. The parameter `cluster_type` must be set to `bcp` for all the tasks.

##### 5.1.2.2. Pipeline Configurations

The `conf/config.yaml` file contains default configuration settings for various stages of your pipeline, including data
preparation, training, fine-tuning, evaluation, and more. The `stages` field specifies the stages that will be executed
during the pipeline run.

```yaml
defaults:
  - _self_
  - cluster: bcm  # Leave it as bcm even if using bcp. It will be ignored for bcp.
  - data_preparation: multimodal/download_multimodal
  - training: clip/vit_B_32
  - conversion: null
  - fine_tuning: null
  - evaluation: clip/imagenet_zeroshot
  - fw_inference: null
  - export: clip/export_clip
  - override hydra/job_logging: stdout

stages:
  - data_preparation
  - training
  - fw_inference
  - export
```

All configuration options for each stage can be found in the `conf` folder, organized by stage. The configuration files
are structured as `conf/(stage_name)/(model_type)/(model_name).yaml`. To use your desired
configuration, simply edit the fields accordingly. For example, you can find training configuration options in
the `conf/training` folder. The `training` field in the `defaults` section, such as `clip/vit_B_32`, indicates that the
training configuration for the CLIP model is sourced from the `conf/clip/vit_B_32.yaml` file. To view or modify the
training configuration, you can check this specific file.

**Customize the Pipeline for Your Needs:**

1. **Include or exclude a stage**: To include or exclude a stage in the pipeline, add or remove the stage name from
   the `stages` list.
2. **Modify configuration settings**: To modify the configuration settings for a specific stage, navigate to the
   appropriate folder in the `conf` directory (e.g., `conf/training` for training options) and edit the relevant fields.
3. **Use a different configuration file**: To use a different configuration file for a stage, update the corresponding
   field in the `defaults` section (e.g., change `training: clip/vit_B_32` to `training: (model_type)/(model_name)`).
4. **Update specific stage configurations**: Modify the YAML files in `conf/(stage_name)/(model_type)/(model_name).yaml`
   to update specific stage configurations, such as the number of nodes, precision, and model configurations.

##### 5.1.2.3. Environment Variables Configurations

To configure or add additional environment variables when running pipelines, you can modify or include new fields under
the env_vars section in the conf/config.yaml file. If a variable is set to null, it will be ignored.

```yaml
env_vars:
  NCCL_TOPO_FILE: null # Should be a path to an XML file describing the topology
  UCX_IB_PCI_RELAXED_ORDERING: null # Needed to improve Azure performance
  ...
  TRANSFORMER_OFFLINE: 1
```

By adjusting these settings, you can customize the environment variables to better suit your specific needs and
requirements during pipeline execution.

##### 5.1.2.4. NUMA Mapping Configurations

NUMA mapping is a technique used with multiple processors, where memory access times can vary depending on which
processor is accessing the memory. The goal of NUMA mapping is to assign memory to processors in a way that minimizes
non-uniform memory access times and ensures that each processor has access to the memory it needs with minimal delay.
This technique is important for maximizing system performance in high-performance computing environments.

The NUMA mapping can also be configured from the `conf/config.yaml` file. The
mapping should be automatic; the code will read the number of CPU cores available
in your cluster, and provide the best possible mapping, to maximize performance.
The mapping is enabled by default, but it can be disabled by setting
`enable: False` in the `numa_mapping` section of the `conf/config.yaml` file.
The type of mapping can also be configured using the same file. See the full
config parameters below:

```yaml
numa_mapping:
  enable: True  # Set to False to disable all mapping (performance will suffer).
  mode: unique_contiguous  # One of: all, single, single_unique, unique_interleaved or unique_contiguous.
  scope: node  # Either node or socket.
  cores: all_logical  # Either all_logical or single_logical.
  balanced: True  # Whether to assing an equal number of physical cores to each process.
  min_cores: 1  # Minimum number of physical cores per process.
  max_cores: 8  # Maximum number of physical cores per process. Can be null to use all available cores.
```

#### 5.1.3. Launch Pipeline

`main.py` is the primary file to execute for running various stages in your pipeline, including data preparation,
training, conversion, fine-tuning, and evaluation.

To run the specified pipelines with the provided configurations, simply execute:

```
python3 main.py
```

**Modifying Configurations**: NeMo launcher uses Hydra and OmegaConf to manage job configurations with YAML files. There
are **two ways** to modify the configurations:

1. Edit the configuration file directly: As mentioned in Section 5.1.2., you can directly edit the corresponding
   configuration file to make the necessary changes.
2. Override configurations through the command line: You can also override some configurations directly through the
   command line when calling main.py. For example, to override the stages list inside `conf/config.yaml`, use:
   ```
   python3 main.py stages=[training]
   ```

**Pipeline Execution Details**: After calling `main.py`, the NeMo launcher scripts perform several tasks to
set up and run your customized pipeline. Here's an overview of these tasks:

1. **Interpolate the configs**: The NeMo launcher scripts will first interpolate the configuration files based on Hydra
   overwriting. This process ensures that any modifications or auto-calculated fields are incorporated into the
   configuration.
2. **Save the YAML file**: After interpolating the configs, the NeMo launcher scripts will save the updated YAML file.
   This file will contain all the customizations specified through the command line or by editing the configuration
   files directly.
3. **Generate submission scripts**: Next, the NeMo launcher scripts will generate submission scripts. This script
   includes the necessary calls to NeMo, as well as other collections needed for the pipeline. If there are multiple
   scripts, they will be streamlined with dependency.
4. **Submit the scripts**: Finally, the NeMo launcher scripts will submit the generated scripts, which will execute your
   customized pipeline with the specified configurations.

**Dry Run**: To perform a dry run of your pipeline without actually executing it, you can set the environment
variable `NEMO_LAUNCHER_DEBUG=1`. With this environment variable set, running `main.py` will generate the interpolated
configuration files, save the YAML file, and create the launch script as usual, but it will not submit or launch the
scripts. This allows you to inspect and verify the generated files and scripts before proceeding with the actual
execution of the pipeline.

By automating these tasks, the NeMo launcher scripts streamline the process of setting up and running the pipeline,
making it easier for you to focus on your experiments and analyses.

**Note for Base Command Platform Users**: When using the Base Command Platform, directly modifying the configuration
file is not recommended. This is because different nodes on the Base Command Platform do not share the same file system,
so changes made on one node will not be reflected on other nodes. To ensure consistency across nodes, always use command
line overrides for configuration changes on Base Command Platform.

#### 5.1.4. Example: Pre-train Stable Diffusion 860M Model for 10 Epochs with Resolution 256

In this example, we will demonstrate how to customize the pipeline according to the following instructions:

1. Run only the training stage.
2. Train the `stable_diffusion` model with the `860m_res_256` configuration.
3. Change the training epoch from `1` to `10`.

**Step-by-Step Guide**

1. **Include only the training stage**: To run only the training stage, update the `stages` list in `conf/config.yaml`:
   ```yaml
   stages:
     - training
   ```
2. **Select the `stable_diffusion` model with `860m_res_256` configuration**: Update the `training` field in
   the `defaults` section of `conf/config.yaml`:
   ```yaml
   training: stable_diffusion/860m_res_256
   ```
3. **Change the training epochs**: Navigate to the `conf/training/stable_diffusion/860m_res_256.yaml` file and update
   the `max_epochs` field under the `trainer` section:
   ```yaml
    trainer:
      max_epochs: 10
   ```
4. **Pipeline Execution**: With these customizations in place, the pipeline will now execute only the `training` stage,
   using
   the `stable_diffusion` model with the `860m_res_256` configuration, and train for a total of `10` epochs.
   To run the customized pipeline, simply execute:
   ```
   python3 main.py
   ```

Instead of manually editing the configuration files, you can also use **Hydra's override feature** to achieve the same
customizations in a single command. This allows you to quickly test different configurations without modifying the
original files. To run the customized pipeline according to the instructions provided earlier, use the following
command:

```
python3 main.py stages=[training] training=stable_diffusion/860m_res_256 training.trainer.max_epochs=10
```

**Note**: When using Hydra's override feature, make sure to include the stage name (training in this example) for
overriding a stage configuration found in conf/(stage_name)/(model_type)/(model_name).yaml. This ensures that the
correct stage and configuration file are targeted for the override.

### 5.2. Data Preparation

#### 5.2.1 ImageNet

_Note: It is the responsibility of each user to check the content
of the dataset, review the applicable licenses, and determine if it is suitable for their intended use.
Users should review any applicable links associated with the dataset before placing the data on their machine._

Please note that according to the ImageNet terms and conditions, automated scripts for downloading the dataset are not
provided. Instead, kindly follow the steps outlined below to download and extract the data.

##### 5.2.1.1 ImageNet 1k

1. Create an account on [ImageNet](http://image-net.org/download-images) and navigate to ILSVRC 2012.
   Download "Training images (Task 1 & 2)" and "Validation images (all tasks)" to `data/imagenet_1k`.
2. Extract the training data:

  ```bash
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..
  ```

3. Extract the validation data and move the images to subfolders:

  ```bash
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```

##### 5.2.1.2 ImageNet 21k

1. Create an account on [ImageNet](http://image-net.org/download-images) and download "ImageNet21k" to
   `data/imagenet_21k`.
2. Extract the data:

  ```bash
  tar -xvf winter21_whole.tar.gz && rm -f winter21_whole.tar.gz
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  ```

#### 5.2.2 Multimodal Datasets

_Note: It is the responsibility of each user to check the content
of the dataset, review the applicable licenses, and determine if it is suitable for their intended use.
Users should review any applicable links associated with the dataset before placing the data on their machine._

##### 5.2.2.1 Overview

For all multimodal models (except Instruct-Pix2Pix; see Section 5.2.3), we provide a generic pipeline as detailed below
to download and prepare the dataset. The pipeline is suitable for any multimodal datasets hosted on the
[Hugging Face data repository](https://huggingface.co/datasets?task_categories=task_categories:text-to-image)
where the data is stored as one or more parquet files. The pipeline processes the dataset into the
[WebDataset](https://github.com/webdataset/webdataset) format, consisting of tar files of equal sizes for
efficient training.

The 5 sub-stages are as follows.

1. `download_parquet`: Parquet files consisting of text (captions) and image URLs are downloaded from a Hugging Face
   repository.
2. `download_images`: The images are downloaded from their respective URLs and, along with the captions, are
   packed into tar files following the Webdataset format.
3. `reorganize_tar`: (Optional) Due to a variety of reasons (such as unstable network or removal of images),
   some images may fail to download, resulting in uneven tar files with varying number of examples each.
   If you are using a training sampler that does not support uneven tar files, you need to re-organize the contents of
   the
   tar files so that each one contains an equal number of image-text pairs.
4. `precache_encodings`: (Optional) If you are training a model with frozen encoders (e.g. Stable Diffusion),
   you have the option to precache (precompute) image and/or text encodings (embeddings) in this sub-stage.
   Precaching these encodings can significantly enhance training throughput.
5. `generate_wdinfo`: (Optional) The `wdinfo.pkl` file, which stores information on dataset shards, is generated.

Depending on your specific circumstance, not all sub-stages need to be run all at once.
For example, for parquet datasets not hosted on HuggingFace or whose format is not parquet,
sub-stages 2-5 can be used to process locally downloaded datasets.
For webdatasets already downloaded locally, sub-stages 4-5 can be used to precache the encoding to reduce training time.
For models that encode image and text on-the-fly, only sub-stages 1-3 need to be run.

Instruction for configuring each sub-stage is provided as a comment next to each field in
`conf/data_preparation/multimodal/download_multimodal.yaml`

##### 5.2.2.2 Running the Pipeline

Follow Section 5.1.1 to set up the environment.
To run the data preparation pipeline for multimodal, set the `conf/config.yaml` file to:

```yaml
defaults:
  - data_preparation: multimodal/download_multimodal

stages:
  - data_preparation
```

In `multimodal/download_multimodal.yaml`, set the `dataset_repo_id` and the `dataset_output_root` path.
Enable the desired sub-stages by setting the `enable` flag, and modify other parameters as needed.

Then run:

```
python3 main.py
```

##### 5.2.2.3 Configuration for Precaching

###### 5.2.2.3.1 General Format

Precaching refers to the offline computation of image and text encodings prior to training a model. This technique
is suitable for any model that uses pretrained, frozen encoders during training.
By using precached encodings, embeddings for image and textx do not need to be recomputed in each epoch,
thereby significantly improving training throughput (up to 60% higher).

Precached encodings are saved in the format of WebDataset.
Each tar file contains one pickle file to store all the modality embeddings for each training example. Optionally,
the tar file may also include the original image or text files

```text
t0_r0_0.tar
|---- 00000.pickle
|---- 00000.jpg (optional)
|---- 00000.txt (optional)
|---- 00001.pickle
|---- 00001.jpg (optional)
|---- 00001.txt (optional)
...
```

Each pickle file stores one python dictionary, with key value pairs storing the embedding name and the embedding as a
numpy array.

###### 5.2.2.3.2 Precaching Config

Configuration for precaching can be extensive and intricate for some models. To maintain clarity and ensure an
organized workflow, we utilize a separate YAML file for these configurations.
An example can be found here: `mulimodal/precache_sd.yaml`.

```yaml
encodings:
  - modality: image
    extension: jpg
    key: autoencoderkl_image
    precision: 16
    encoder_config:
      cls: nemo.collections.multimodal.models.stable_diffusion.ldm.autoencoder.AutoencoderKL
      ... (kwargs to initialize the encoder)
  - modality: text
    extension: txt
    key: clip-vit-large-patch14_text
    precision: 32
    store_pad_tokens: True
    encoder_config:
      cls: nemo.collections.multimodal.modules.stable_diffusion.encoders.modules.FrozenCLIPEmbedder
      ... (kwargs to initialize the encoder)
```

In this YAML file, the `encodings` field specifies a list of embeddings to be saved in the pickle file.
Each entry can have the following attributes:

- `modality`: either image or text
- `extension`: file extension for this modality in the tar file (e.g. 'jpg', 'txt')
- `key`: dictionary key for the encoding.
  It is recommended to follow the format `{model_name}-{model_variant}_{modality}`, if applicable.
  e.g. `clip-vit-large-patch14_text`
- `precision`: precision of the stored tensors (32 or 16)
- `store_pad_tokens`: Whether to store the PAD tokens. Not storing PAD tokens can significantly reduce disk usage,
  but the training script must account for this. Ignored for image modality.
- `encoder_config`: This dictionary must contain `cls` which points to the location of the encoder class.
  The rest of the parameters are treated as kwargs to initiate the encoder class.
    - Note: the encoder class must implement an `encode` or `__call__` function. If `store_pad_tokens`, this function
      must
      return the encoded tensor. Otherwise, this function must return a tuple of (encoded_tensor, text_mask).

Note that it is not required to have only one encoding per modality, if there are multiple encoders.
The `encodings` field is designed as a list to account for this. For example, it's possible to have one image embedding
from CLIP, one text embedding from CLIP, and a second text embedding from T5.

###### 5.2.2.3.3 Resume Precaching (Advanced)

The precaching module is able to launch multiple tasks (as specified by `precache_encodings.node_array_size`)
in parallel in order to reduce the time required for each task. In the event of failed or interrupted run, we provide
the option
to resume precaching by specifying the exact `task_id` or range of `task_id`s to re-run. This option eliminates the need
to rerun the entire precaching process which can be lengthy.

Consider the following two scenarios as examples.

1. Interrupted runs: suppose 100 tasks (0-99) were launched, but tasks 50-99 did not complete before the cluster went
   down. To resume the runs, specify a string in `node_array_size` in
   `conf/data_preparation/multimodal/download_multimodal.yaml`

```yaml
precache_encodings:
  node_array_size: 50-99
```

In addition, in `nemo_launcher/collections/dataprep_scripts/multimodal_dataprep/conf/config.yaml`, specify

```yaml
override_task_count: 100
```

2. Failed run: suppose 100 tasks (0-99) were launched, but task 67 experienced node failure.
   To re-run task 67, specify in `conf/data_preparation/multimodal/download_multimodal.yaml`

```yaml
precache_encodings:
  node_array_size: 1
```

In addition, in `nemo_launcher/collections/dataprep_scripts/multimodal_dataprep/conf/config.yaml`, specify

```yaml
override_task_id: 67
override_task_count: 100
```

###### 5.2.2.3.4 Known Issue

Due to a [Lightning DDP limitation](https://github.com/Lightning-AI/lightning/issues/3325), the precaching module may
drop about 0.01% to 0.1% of input data. The specific ratio will depend on the cluster configuration, tarfile chunk_size,
precaching batch_size and dataset size, but will be consistent across runs.
We anticipate that dropping a small percentage of data will not have a significant impact on model training.

#### 5.2.3 Instruct Pix2Pix

_Note: It is the responsibility of each user to check the content
of the dataset, review the applicable licenses, and determine if it is suitable for their intended use.
Users should review any applicable links associated with the dataset before placing the data on their machine._

To download and prepare the custom dataset used for training Instruct-Pix2Pix, please follow the instruction from
the official [Instruct-Pix2Pix](https://github.com/timothybrooks/instruct-pix2pix#generated-dataset)
repository.

Move the downloaded data to `${data_dir}/instruct_pix2pix/clip-filtered-dataset`

#### 5.2.4 MSCOCO for FID Evaluation

_Note: It is the responsibility of each user to check the content
of the dataset, review the applicable licenses, and determine if it is suitable for their intended use.
Users should review any applicable links associated with the dataset before placing the data on their machine._

For more details on the evaluation workflow, please see Section 5.6.3.

##### 5.2.4.1 Download and Setup

1. Review the terms of use from the official [COCO](https://cocodataset.org/#download) website.
2. Download the 2014 validation images, and extract the images to `${data_dir}/fid_evaluation/coco2014/val2014`
3. Download the 2014 train/val annotations, and extract `captions_val2014.json` to
   `${data_dir}/fid_evaluation/coco2014/captions_val2014.json`
4. Review the terms of use of [COCO API](https://github.com/cocodataset/cocoapi), then install the Python API following
   the instructions.
5. Install the dependencies for the preprocessing script: `pip install matplotlib cython Pillow`

##### 5.2.4.2 Preprocess Images and Captions

Follow Section 5.1.1 to set up the environment.
To run the data preparation pipeline for FID evaluation, set the `conf/config.yaml` file to:

```yaml
defaults:
  - data_preparation: fid_evaluation/download_coco2014

stages:
  - data_preparation
```

In `fid_evaluation/download_coco2014`, set the `dataset_output_root` path to a desired location, and specify
whether to preprocess images and captions.

Then run:

```
python3 main.py
```

### 5.3. Model Training

We provide predefined training configurations for all released model types, which can be found in the `conf/training/`
directory. These configurations include carefully selected hyper parameters that serve as a guideline for creating
custom model configurations. To choose the desired configuration, simply update the training parameter in
the `conf/config.yaml` file. For additional guidance on customizing configurations, please refer
to [Section 5.1](#51-getting-started-with-multimodal-nemo-megatron) in the
documentation.

For the Base Command Platform (BCP), it is important to note that all jobs must be launched in multi-node mode. This
requirement ensures proper setup of BCP pytorch environment.

#### 5.3.1. Vision Transformer Training

We have curated 5 configurations with suggested hyperparameters specifically for the NVIDIA DGX SuperPOD, which is
equipped with 8 NVIDIA A100 80GB GPUs. The configurations for the curated models can be found in the `conf/training/vit`
directory. You can access and modify the parameters to adjust the hyperparameters for your specific training runs. By
customizing these settings, you can tailor the model's performance and training efficiency to better suit your needs and
requirements.

| Model | Model size (M) | Hidden size | FFN_dim | Attention heads | Number of layers | Batch Size per GPU | Accumulated Global Batch Size | Precision | AMP Level | Total Training Samples |
|-------|----------------|-------------|---------|-----------------|------------------|--------------------|-------------------------------|-----------|-----------|------------------------|
| B/16  | 86             | 768         | 3072    | 12              | 12               | 512                | 4096                          | BF16      | O2        | 400M                   |
| L/16  | 303            | 1024        | 4096    | 16              | 24               | 256                | 4096                          | BF16      | O2        | 400M                   |
| H/14  | 632            | 1280        | 5120    | 16              | 32               | 128                | 4096                          | BF16      | O2        | 400M                   |
| g/14  | 1011           | 1408        | 6144    | 16              | 40               | 64                 | 4096                          | BF16      | O2        | 400M                   |
| G/14  | 1843           | 1664        | 8192    | 16              | 48               | 32                 | 4096                          | BF16      | O2        | 400M                   |

To enable the training stage with a Vision Transformer (ViT) model, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `training` field to point to the desired ViT
   configuration file. For example,
   if you want to use the `B/16`(i.e. `B_16`) configuration, change the `training` field to `vit/B_16`.
   ```yaml
    defaults:
      - _self_
      - cluster: bcm
      - data_preparation: null
      - training: vit/vit_B_16
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the training stage is included. For example,
   ```yaml
    stages:
      - training
      ...
   ```

**Remarks**: The correctness of our Vision Transformer implementation has been verified by pretraining `ViT B/16` for
300 epochs on the ImageNet 1K dataset. This demonstrates that our implementation is consistent with the expected
performance and results of Vision Transformers in general.

#### 5.3.2. CLIP Training

We have curated 3 configurations with suggested hyperparameters specifically for the NVIDIA DGX SuperPOD, which is
equipped with 8 NVIDIA A100 80GB GPUs. The configurations for the curated models can be found in
the `conf/training/clip` directory. You can access and modify the parameters to adjust the hyperparameters for your
specific training runs. By customizing these settings, you can tailor the model's performance and training efficiency to
better suit your needs and requirements.

| Model    | Image size | Text Model size (M) | Image Model size (M) | Output dim | Batch Size per GPU | Accumulated Global Batch Size | Precision | AMP Level | Total Training Samples |
|----------|------------|---------------------|----------------------|------------|--------------------|-------------------------------|-----------|-----------|------------------------|
| ViT B/32 | 224        | 63                  | 87                   | 512        | 500                | 32000                         | BF16      | O2        | 12B                    |
| ViT L/14 | 224        | 123                 | 303                  | 768        | 112                | 32256                         | BF16      | O2        | 12B                    |
| ViT H/14 | 224        | 354                 | 638                  | 1024       | 80                 | 32000                         | BF16      | O2        | 12B                    |

To enable the training stage with a CLIP model, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `training` field to point to the desired CLIP
   configuration file. For example,
   if you want to use the `ViT B/32` (i.e. `vit_B_32`), change the `training` field to `clip/vit_B_32`.
   ```yaml
    defaults:
      - _self_
      - cluster: bcm
      - data_preparation: multimodal/download_multimodal
      - training: clip/vit_B_32
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the training stage is included. For example,
   ```yaml
    stages:
      - data_preparation
      - training
      ...
   ```

**Remarks**:

1. NeMo CLIP does not yet support gradient accumulation. Therefore, please
   ensure `micro_batch_size * num_gpus = global_batch_size` (i.e. gradient accumulation step is 1).
2. For CLIP models, you can enable Exponential Moving Average (EMA) by setting `training.exp_manager.ema.enable=True`.
   However, EMA is currently not compatible with AMP O2. To use EMA, you must disable AMP O2 by
   setting `training.model.megatron_amp_O2=False`. Enabling EMA can help your model converge faster, but be aware that
   it may result in a slight performance penalty.

#### 5.3.3. Stable Diffusion Training

We have curated configurations with suggested hyperparameters specifically for the NVIDIA DGX SuperPOD, which is
equipped with 8 NVIDIA A100 80GB GPUs. The configurations for the curated models can be found in
the `conf/training/stable_diffusion` directory. You can access and modify the parameters to adjust the hyperparameters
for your
specific training runs. By customizing these settings, you can tailor the model's performance and training efficiency to
better suit your needs and requirements.

The training process for Stable Diffusion typically involves multiple stages in which different resolutions and datasets
are deliberately alternated to achieve superior image quality. We provide two training configurations here: one for
pretraining at a resolution of 256x256 and another for resuming from the pretraining weights and continuing to improve
the model's performance. It is important to note that to maintain image quality improvement, each stage requires loading
the unet weights from the previous stage and ideally switching to another dataset to improve diversity. We have verified
convergence up to SD v1.5 by switching between multiple subsets of our multimodal blend*. Reproducing SD v1.5 using the
datasets recommended in the Huggingface model cards is straightforward with our implementation.

\**Our multimodal dataset is originated from Common Crawl with custom filtering.*

| Stage       | Resolution | Unet model size (M) | Text conditioning model       | Batch Size per GPU | Accumulated Global Batch Size | Precision | AMP Level | Dataset              | Dataset Filtering       | Total Training Samples |
|-------------|------------|---------------------|-------------------------------|--------------------|-------------------------------|-----------|-----------|----------------------|-------------------------|------------------------|
| Pretraining | 256        | 859                 | openai/clip-vit-large-patch14 | 128                | 8192                          | FP16      | O1        | Our Multimodal Blend | None                    | 680M                   |
| SD v1.1     | 512        | 859                 | openai/clip-vit-large-patch14 | 32                 | 8192                          | FP16      | O1        | Our Multimodal Blend | Resolution >= 1024x1024 | 409M                   |
| SD v1.2     | 512        | 859                 | openai/clip-vit-large-patch14 | 32                 | 8192                          | FP16      | O1        | Our Multimodal Blend | Resolution >= 512x512   | 1.23B                  |
| SD v1.5     | 512        | 859                 | openai/clip-vit-large-patch14 | 32                 | 8192                          | FP16      | O1        | Our Multimodal Blend | Resolution >= 512x512   | 1.32B                  |

To enable the training stage with Stable Diffusion, make sure:

1. In the `defaults` section, update the `training` field to point to the desired Stable Diffusion configuration file.
   For example,
   if you want to start the pretraining from scratch, change the training field to `stable_diffusion/860m_res_256.yaml`.
   ```yaml
    defaults:
      - _self_
      - cluster: bcm
      - data_preparation: multimodal/download_multimodal
      - training: stable_diffusion/860m_res_256.yaml
      ...
   ```
2. In the stages field, make sure the training stage is included. For example,
   ```yaml
    stages:
      - data_preparation
      - training
      ...
   ```

**Remark**:

1.To continue training the Stable Diffusion model from the pretraining results, we reset the trainig process
by only loading the UNet weights. You can do this by using the last checkpoint from the previous training and passing it
to `training.model.unet_config.from_pretrained`. Due to different naming in model parameters, indicating you are loading
from checkpoint trained by NeMo , set `training.model.unet_config.from_NeMo=True`. If you are resuming training from a
Huggingface checkpoint, you can also load the Unet weights from that source. In this case, you need to
set `training.model.unet_config.from_NeMo=False`.

2.For the training process up to SD-v1.5, we have enabled 10% dropping of text conditioning after SD v1.1 training phase
completed, to improve the performance on classifier-free
guidance.

#### 5.3.4. Instruct Pix2Pix Training

Instruct Pix2Pix essentially performs tuning on top of an existing Stable Diffusion checkpoint. The recommended
configuration can be found in the `conf/training/instruct_pix2pix` directory. You can access and modify the parameters
to customize the hyperparameters according to your specific training requirements.

To enable the training stage with an Instruct Pix2Pix model, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `training` field to point to the desired Instruct Pix2Pix
   configuration file. For example,
   if you want to use the `860m_sd_edit`, change the `training` field to `instruct_pix2pix/860m_sd_edit`.
   ```yaml
    defaults:
      - _self_
      - cluster: bcm
      - data_preparation: null
      - training: instruct_pix2pix/860m_sd_edit
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the training stage is included. For example,
   ```yaml
    stages:
      - data_preparation
      - training
      ...
   ```

**Remarks**: You can feed the trained Stable Diffusion checkpoint into Instruct Pix2Pix training by
specifying `training.model.ckpt_path` (or set `ckpt_path` field in the `model` section of `860m_sd_edit.yaml`). The
checkpoint can be sourced from either NeMo or Hugging Face in the form of a `.ckpt` file.

#### 5.3.5. DreamBooth Training

Dreambooth is also fine-tuning on top of an existing Stable Diffusion checkpoint. The recommended configuration can be
found in the `conf/training/dreambooth` directory. You can access and modify the parameters to customize the
hyperparameters according to your specific training requirements. The instance dataset should contain several pictures
of
object you want to inject to the model. To achieve better quality, 3-5 pictures from different angles is preferred.
To enable the training stage with a dreambooth model, make sure:

1. In the defaults section, update the training field to point to the desired configuration file. For
   example, `dreambooth/860m.yaml`.

   ```yaml
    defaults:
       - _self_
       - cluster: bcm
       - data_preparation: null
       - training: dreambooth/860m.yaml
       ...
   ```


2. In the stages field, make sure the training stage is included. For example,

   ```yaml
    stages:
      ...
      - training
      ...
   ```

**Remarks**:

1.To train DreamBooth with a prior preservation loss, you need to prepare a regularization dataset. The regularization
dataset is usually populated by images generated from a similar prompt without a special token, using the original
Stable Diffusion checkpoint that we fine-tuned on. For example, if the instance prompt you are training on is "a photo
of a sks dog", then the regularization data could be generated by a prompt like "a photo of a dog".

2.To generate regularization images, pass the Stable Diffusion checkpoint you want to use to
training.model.restore_from_path. Note that the .nemo checkpoint is required here. The U-Net weights you want to
fine-tune on should be set in training.model.unet_config.from_pretrained. You can follow the same procedure as described
above in section [5.3.3. Stable Diffusion Training].

### 5.4. Checkpoint Conversion

We provide a convenient tool for converting checkpoints from the `.ckpt` format to the `.nemo` format. The `.nemo`
format checkpoints can be used later in evaluation and inference stages. Users don't need to run the checkpoint
conversion explicitly, as a `.nemo` checkpoint will be automatically generated and saved in the checkpoints folder at
the end of training or fine-tuning. However, if you want to perform inference with an intermediate checkpoint, you will
need to use the conversion script to convert the checkpoint from the `.ckpt` format to the `.nemo` format.

The usage of the conversion script is consistent across different model types. All conversion configuration files can be
found in the `conf/conversion` folder. For additional guidance on customizing configurations, please refer
to [Section 5.1](#51-getting-started-with-multimodal-nemo-megatron) in the
documentation.

To enable the `conversion` stage and configure conversion settings, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `conversion` field to point to the desired model type's
   configuration file. For example, if you want to convert a CLIP model, change the `conversion` field
   to `clip/convert_clip`.
   ```yaml
    defaults:
      - conversion: clip/convert_clip
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `conversion` stage is included. For example,
   ```yaml
    stages:
      ...
      - conversion
      ...
   ```
3. In the target conversion YAML file, modify required fields like `checkpoint_folder` and `checkpoint_name`. For
   example, if you want to convert a CLIP model, modify or override the following fields
   inside `conf/conversion/clip/convert_clip.yaml`.
   ```yaml
   run:
     model_train_name: clip_vit_B_32
     train_dir: ${base_results_dir}/${.model_train_name}
   model:
     model_type: megatron_clip
     checkpoint_folder: ${conversion.run.train_dir}/results/checkpoints
     checkpoint_name: latest # latest OR name pattern of a checkpoint (e.g. megatron_gpt_*last.ckpt)
     hparams_file: ${conversion.run.train_dir}/results/hparams.yaml # Optional
   ```

**Remark**:

1. The `checkpoint_name` can be set to `latest`, which means it will get the latest checkpoint in the folder, or to a
   regex pattern such as `megatron_clip_*last.ckpt`.
2. By default, the checkpoint folder will link to the training or fine-tuning checkpoints folder and find the latest
   checkpoint.
3. **Advanced**: The hparams_file field is optional. If you want to change any hyperparameters for model initialization,
   you can override them in the hparams.yaml file. However, be cautious when making changes, as altering the model
   architecture may prevent the model weights from loading correctly.

### 5.5. Model Fine-tuning

We provide predefined fine-tuning configurations for Vision Transformer models, which can be found in
the `conf/fine_tuning/`
directory. These configurations include carefully selected hyper parameters that serve as a guideline for creating
custom model configurations. For additional guidance on customizing configurations, please refer
to [Section 5.1](#51-getting-started-with-multimodal-nemo-megatron) in the
documentation.

#### 5.5.1. Vision Transformer Fine-tuning

We provide a predefined fine-tuning configuration for the `ViT B/16` model on ImageNet-1K, which can be found in
the `conf/fine_tuning/imagenet1k.yaml` file. The following table highlights the key differences between ViT pretraining
and fine-tuning:

| Aspect               | ViT Pretraining           | ViT Fine-tuning              |
|----------------------|---------------------------|------------------------------|
| Configuration Folder | `conf/training/vit`       | `conf/fine_tuning/vit`       |
| Training Samples     | 400M                      | 10M                          |
| Optimizer            | Fused AdamW               | SGD                          |
| Resolution           | 224x224                   | 384x384                      |
| Classification Head  | MLP with one hidden layer | MLP with single linear layer |

To enable the fine-tuning stage with a ViT model, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `fine_tuning` field to point to the desired ViT
   configuration file. For example,
   if you want to use the `vit/imagenet1k` configuration, change the `fine_tuning` field to `vit/imagenet1k`.
   ```yaml
    defaults:
      - fine_tuning: vit/imagenet1k
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `fine_tuning` stage is included. For example,
   ```yaml
    stages:
      - fine_tuning
      ...
   ```

**Remarks**: To load a pretrained checkpoint for fine-tuning, set the `restore_from_path` field in the `model` section
to the path of the pretrained checkpoint in `.nemo` format. By default, this field links to the `.nemo` format
checkpoint located in the training checkpoints folder.

### 5.6. Model Evaluation

In Multimodal NeMo Megatron, we also provide simple scripts for users to benchmark their trained models, including ViT,
CLIP and Stable Diffusion. The configuration files for these evaluations can be found in the `conf/evaluation`
directory. These scripts allow you to assess the performance of your trained models on various metrics. For additional
guidance on customizing configurations, please refer to [Section 5.1](#51-getting-started-with-multimodal-nemo-megatron)
in the
documentation.

#### 5.6.1. Vision Transformer Evaluation

For the Vision Transformer, our evaluation script processes the ImageNet 1K validation folder and computes the final
validation accuracy.

To enable the evaluation stage with a ViT model, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `evaluation` field to point to the desired ViT
   configuration file. For example,
   if you want to use the `vit/imagenet_val` configuration, change the `evaluation` field to `vit/imagenet_val`.
   ```yaml
    defaults:
      - evaluation: vit/imagenet_val
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `evaluation` stage is included. For example,
   ```yaml
    stages:
      - evaluation
      ...
   ```
3. Configure `imagenet_val` field of `conf/evaluation/vit/imagenet_val.yaml` to be the ImageNet 1K validation folder.

**Remarks**:

1. To load a pretrained checkpoint for inference, set the `restore_from_path` field in the `model` section to the path
   of the pretrained checkpoint in `.nemo` format in `conf/evaluation/vit/imagenet_val.yaml`. By default, this field
   links to the `.nemo` format checkpoint located in the ImageNet 1K fine-tuning checkpoints folder.
2. We highly recommend users to use the same precision (i.e. `trainer.precision`) for evaluation as was used during
   training.

#### 5.6.2. CLIP Evaluation

For CLIP models, our evaluation script calculates zero-shot ImageNet 1K validation accuracy.

To enable the evaluation stage with a CLIP model, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `evaluation` field to point to the desired CLIP
   configuration file. For example,
   if you want to use the `clip/imagenet_zeroshot` configuration, change the `evaluation` field
   to `clip/imagenet_zeroshot`.
   ```yaml
    defaults:
      - evaluation: clip/imagenet_zeroshot
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `evaluation` stage is included. For example,
   ```yaml
    stages:
      - evaluation
      ...
   ```
3. Configure `imagenet_val` field of `conf/evaluation/clip/imagenet_zeroshot.yaml` to be the ImageNet 1K validation
   folder.

**Remarks**:

1. To load a pretrained checkpoint for inference, set the `restore_from_path` field in the `model` section to the path
   of the pretrained checkpoint in `.nemo` format in `conf/evaluation/clip/imagenet_zeroshot.yaml`. By default, this
   field links to the `.nemo` format checkpoint located in the CLIP trainning checkpoints folder.
2. **Knonw issue**: In CLIP model evaluation, using `fp32` for inference with a trained model in `bf16` or `fp16` does
   not produce expected results. We highly recommend users to use the same precision (i.e. `trainer.precision`) for
   inference as was used during training.

#### 5.6.3. Stable Diffusion Evaluation

Our evaluation script performs image generation for the captions provided in the validation subset of the MS COCO
dataset, computes the FID score between real and generated images, computes the CLIP score betweel generated images and
teh corresponding captions, and plots the FID-CLIP graph. This is a multi-stage evaluation, and our scripts will
automatically generate SLURM jobs with dependencies.

To configure the configuration files and enable the evaluation stage for Stable Diffusion, follow the steps outlined
below:

1. In the `defaults` section of `conf/config.yaml`, update the `evaluation` field to point to the desired Stable
   Diffusion
   configuration file. For example,
   if you want to use the `stable_diffusion/fid_clip` configuration, change the `evaluation` field
   to `stable_diffusion/fid_clip`.
   ```yaml
    defaults:
      - evaluation: stable_diffusion/fid_clip
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `evaluation` stage is included. For example,
   ```yaml
    stages:
      - evaluation
      ...
   ```
3. Configure `conf/evaluation/stable_diffusion/fid_clip.yaml` to specify `node_array_size` and `ntasks_per_node`, as
   well as which sub-stages to run.
   ```yaml
    generate_images: True
    compute_fid_scores: True
    compute_clip_scores: True
    plot_fid_clip: True
   ```

**Remarks**:

1. To load a pretrained checkpoint for inference, set the `restore_from_path` field in the `model` section to the path
   of the pretrained checkpoint in `.nemo` format in `conf/evaluation/stable_diffusion/fid_clip.yaml`. By default, this
   field links to the `.nemo` format checkpoint located in the Stable Diffusion training checkpoints folder.
2. We highly recommend users to use the same precision (i.e. `trainer.precision`) for evaluation as was used during
   training.
3. The `generate_images` sub-stage involves a multi-node run, whereas the other stages utilize only a single GPU.

### 5.7. Model Inference (in NeMo Framework)

In Multimodal NeMo Megatron, we provide scripts to perform inference directly via NeMo framework, rather than using
NVIDIA Triton Inference Server. This allows you to infer with your pretrained models directly without the need for a
separate deployment or inference server. It is particularly useful when you want to experiment with different model
configurations, perform quick evaluations, or prototype a solution before deploying it at scale with Triton Inference
Server or another deployment option.

Our framework inference configurations are provided in the folder `conf/fw_inference`. For additional guidance on
customizing configurations, please refer to [Section 5.1](#51-getting-started-with-multimodal-nemo-megatron) in the
documentation.

#### 5.7.1. Vision Transformer Inference (in NeMo Framework)

For Vision Transformer, our inference script processes a folder of images. For each image in the folder, the script
classifies it into one of the ImageNet 1K classes.

To enable the inference stage with a ViT model, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `fw_inference` field to point to the desired ViT
   configuration file. For example,
   if you want to use the `vit/imagenet1k` configuration, change the `fw_inference` field to `vit/imagenet1k`.
   ```yaml
    defaults:
      - fw_inference: vit/imagenet1k
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `fw_inference` stage is included. For example,
   ```yaml
    stages:
      - fw_inference
      ...
   ```
3. Configure `data_path` of `conf/fw_inference/vit/imagenet1k.yaml` to be the folder containing images for inference.

**Remarks**:

1. To load a pretrained checkpoint for inference, set the `restore_from_path` field in the `model` section to the path
   of the pretrained checkpoint in `.nemo` format in `conf/fw_inference/vit/imagenet1k.yaml`. By default, this field
   links to the `.nemo` format checkpoint located in the ImageNet 1K fine-tuning checkpoints folder.
2. We highly recommend users to use the same precision (i.e. `trainer.precision`) for inference as was used during
   training.

#### 5.7.2. CLIP Inference (in NeMo Framework)

For CLIP models, our inference script calculates CLIP similarity scores between a given image and a list of provided
texts.

To enable the inference stage with a CLIP model, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `fw_inference` field to point to the desired CLIP
   configuration file. For example,
   if you want to use the `clip/clip_similarity` configuration, change the `fw_inference` field
   to `clip/clip_similarity`.
   ```yaml
    defaults:
      - fw_inference: clip/clip_similarity
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `fw_inference` stage is included. For example,
   ```yaml
    stages:
      - fw_inference
      ...
   ```
3. Configure `image_path` and `texts` fields of `conf/fw_inference/clip/clip_similarity.yaml`. Set `image_path` to the
   path of
   the image for inference, and provide a list of texts for the `texts` field.

**Remarks**:

1. To load a pretrained checkpoint for inference, set the `restore_from_path` field in the `model` section to the path
   of the pretrained checkpoint in `.nemo` format in `conf/fw_inference/clip/clip_similarity.yaml`. By default, this
   field links to the `.nemo` format checkpoint located in the CLIP training checkpoints folder.
2. **Knonw issue**: In CLIP model inference, using `fp32` for inference with a trained model in `bf16` or `fp16` does
   not produce expected results. We highly recommend users to use the same precision (i.e. `trainer.precision`) for
   inference as was used during training.

#### 5.7.3. Stable Diffusion Inference (in NeMo Framework)

For text-to-image models, the inference script generates images from text prompts defined in the config file.

To enable the inference stage with Stable Diffusion, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `fw_inference` field to point to the desired Stable
   Diffusion inference configuration file. For example,
   if you want to use the `stable_diffusion/text2img.yaml` configuration, change the `fw_inference` field
   to `stable_diffusion/text2img`.
   ```yaml
    defaults:
      - fw_inference: stable_diffusion/text2img
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `fw_inference` stage is included. For example,
   ```yaml
    stages:
      - fw_inference
      ...
   ```
3. Configure `prompts` and `num_images_per_prompt` fields of `conf/fw_inference/stable_diffusion/text2img.yaml`.
   Set `model.restore_from_path` to the `.nemo` ckpt you want generate images with.

#### 5.7.4. Instruct Pix2Pix Inference (in NeMo Framework)

For Instruct Pix2Pix models, our inference script processes an original image based on a provided edit prompt, modifies
the image accordingly, and saves the edited image as a new file.

To enable the inference stage with a Instruct Pix2Pix model, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `fw_inference` field to point to the desired Instruct
   Pix2Pix configuration file. For example,
   if you want to use the `instruct_pix2pix/edit_cli` configuration, change the `fw_inference` field
   to `instruct_pix2pix/edit_cli`.
   ```yaml
    defaults:
      - fw_inference: instruct_pix2pix/edit_cli
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `fw_inference` stage is included. For example,
   ```yaml
    stages:
      - fw_inference
      ...
   ```
3. Configure the `edit` section in `conf/fw_inference/instruct_pix2pix/edit_cli.yaml`. Most importantly, set the `input`
   field to the path of the original image for inference, and provide an edit prompt in the `prompt` field. The script
   will generate `num_images_per_prompt` images at once based on the provided prompt.
   ```yaml
   edit:
     resolution: 512
     steps: 100
     input: ??? # path/to/input/picture
     outpath: ${fw_inference.run.results_dir}
     prompt: ""
     cfg_text: 7.5
     cfg_image: 1.2
     num_images_per_prompt: 8
     combine_images: [2, 4] # [row, column], set to null if don't want to combine
     seed: 1234
   ```

**Remarks**:

1. To load a pretrained checkpoint for inference, set the `restore_from_path` field in the `model` section to the path
   of the pretrained checkpoint in `.nemo` format in `conf/fw_inference/vit/imagenet1k.yaml`. By default, this field
   links to the `.nemo` format checkpoint located in the ImageNet 1K fine-tuning checkpoints folder.
2. We highly recommend users to use the same precision (i.e. `trainer.precision`) for inference as was used during
   training.
3. Tips for getting better quality results: https://github.com/timothybrooks/instruct-pix2pix#tips

#### 5.7.5. DreamBooth Inference (in NeMo Framework)

For Dreambooth, the inference script generates images from text prompts defined in the config file, similar to section
5.7.3. Note that, dreambooth is a fine-tuning model based on diffusion models to link a special token with certain
subject, so make sure the special token you trained on is included in the text prompt. For
example, `a photo of sks dog sleeping`.

To enable the inference stage with dreambooth, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `fw_inference` field to point to the desired DreamBooth
   inference configuration file. For example,
   if you want to use the `dreambooth/text2img.yaml` configuration, change the `fw_inference` field
   to `dreambooth/text2img`.
   ```yaml
    defaults:
      - fw_inference: dreambooth/text2img
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `fw_inference` stage is included. For example,
   ```yaml
    stages:
      - fw_inference
      ...
   ```
3. Configure `prompts` and `num_images_per_prompt` fields of `conf/fw_inference/dreambooth/text2img.yaml`.
   Set `model.restore_from_path` to the ckpt generated from dreambooth training.

### 5.8. Model Export

In Multimodal NeMo Megatron, we provide scripts to perform export directly via NeMo framework to ONNX and NVIDIA
TensorRT. This allows us to run accelerated inference on the NVIDIA Triton Inference Server detailed in the next
section, section 6.
For the CLIP and ViT models, setting `infer.max_batch_size`, will create ONNX and NVIDIA TensorRT models that accept
batch_sizes
from `1` to `infer.max_batch_size`. For the Stable Diffusion, Instruct Pix2Pix, and Dreambooth pipelines,
the `infer.num_images_per_prompt` (`edit.num_images_per_prompt` in Instruct Pix2Pix) will
act as the `batch_size`, but the NVIDIA TensorRT engines will only work for that size.

The `trainer.precision` config can be set to 16 or 32. Setting to 16 will build the NVIDIA TensorRT engines with fp16
acceleration enabled, expect
longer build times.

Please set `model.restore_from_path` before running export to the correct `.nemo` file.

All relevant inference config fields will be saved for deployment to be automatically read in as defaults. In the output
directory expect
`onnx` and `plan`directories. The `onnx` will contain the ONNX converted models, while the `plan` directory will contain
the NVIDIA TensorRT
Engines created from the ONNX models in addition to the config options.

#### 5.8.1. Vision Transformer Export

To enable the export stage with a ViT model, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `export` field to point to the desired ViT
   configuration file. For example,
   if you want to use the `vit/export_vit` configuration, change the `export` field to `vit/export_vit`.
   ```yaml
    defaults:
      - export: vit/export_vit
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `export` stage is included. For example,
   ```yaml
    stages:
      - export
      ...
   ```
3. Configure `infer.max_batch_size` of the `conf/export/vit/export_vit.yaml` file to set the max_batch_size to use for
   the ONNX and
   NVIDIA TensorRT model.
4. Set the resolution of the model with `max_dim` in the `infer` field. This will be used to generate the ONNX and
   NVIDIA TensorRT formats.

**Remarks**:

1. To load a pretrained checkpoint for inference, set the `restore_from_path` field in the `model` section to the path
   of the pretrained checkpoint in `.nemo` format in `conf/export/vit/export_vit.yaml`. By default, this field
   links to the `.nemo` format checkpoint located in the ImageNet 1K fine-tuning checkpoints folder.

#### 5.8.2. CLIP Export

To enable the export stage with a CLIP model, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `export` field to point to the desired CLIP
   configuration file. For example,
   if you want to use the `clip/export_clip` configuration, change the `export` field
   to `clip/export_clip`.
   ```yaml
    defaults:
      - export: clip/export_clip
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `export` stage is included. For example,
   ```yaml
    stages:
      - export
      ...
   ```
3. Configure `infer.max_batch_size` of the `conf/export/clip/export_clip.yaml` file to set the max_batch_size to use for
   the ONNX and
   NVIDIA TensorRT model.
4. Set the resolution of the model with `max_dim` in the `infer` field. One can also set the `infer.max_text` to be the
   maximum text size for the text_encoder.
   This will be used to generate the ONNX and NVIDIA TensorRT formats.

**Remarks**:

1. To load a pretrained checkpoint for inference, set the `restore_from_path` field in the `model` section to the path
   of the pretrained checkpoint in `.nemo` format in `conf/export/clip/export_clip.yaml`.

#### 5.8.3. Stable Diffusion Export

For text-to-image models, the export script generates three different optimized inference models.
The first model is the VAE Decoder, the second model is the UNet, and the third model is the CLIP Encoder.

1. In the `defaults` section of `conf/config.yaml`, update the `export` field to point to the desired Stable Diffusion
   inference configuration file. For example,
   if you want to use the `stable_diffusion/export_stable_diffusion.yaml` configuration, change the `export` field
   to `stable_diffusion/export_stable_diffusion`.
   ```yaml
    defaults:
      - export: stable_diffusion/export_stable_diffusion
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `export` stage is included. For example,
   ```yaml
    stages:
      - export
      ...
   ```
3. Configure `infer.num_images_per_prompt` of the `conf/export/stable_diffusion/export_stable_diffusion.yaml` file to
   set the batch_size to use for the ONNX and
   NVIDIA TensorRT models.

**Remarks**:

1. To load a pretrained checkpoint for inference, set the `restore_from_path` field in the `model` section to the path
   of the pretrained checkpoint in `.nemo` format in `conf/export/stable_diffusion/export_stable_diffusion.yaml`.

#### 5.8.4. Instruct Pix2pix Export

For Instruct Pix2Pix models, the export script generates four different optimized inference models.
The first model is the VAE Decoder, the second model is the UNet, the third model is the CLIP Encoder, and the fourth
model
is the VAE Encoder.

1. In the `defaults` section of `conf/config.yaml`, update the `export` field to point to the desired Stable Diffusion
   inference configuration file. For example,
   if you want to use the `instruct_pix2pix/export_instruct_pix2pix.yaml` configuration, change the `export` field
   to `instruct_pix2pix/export_instruct_pix2pix`.
   ```yaml
    defaults:
      - export: instruct_pix2pix/export_instruct_pix2pix
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `export` stage is included. For example,
   ```yaml
    stages:
      - export
      ...
   ```
3. Configure `edit.num_images_per_prompt` of the `conf/export/instruct_pix2pix/export_instruct_pix2pix.yaml` file to set
   the batch_size to use for the ONNX and
   NVIDIA TensorRT models.
4. Set a path to an example image to use in `edit.input`.

**Remarks**:

1. To load a pretrained checkpoint for inference, set the `restore_from_path` field in the `model` section to the path
   of the pretrained checkpoint in `.nemo` format in the `conf/export/instruct_pix2pix/export_instruct_pix2pix.yaml`
   file.

#### 5.8.5. DreamBooth Export

For Dreambooth, the export script generates three different optimized inference models.
The first model is the VAE Decoder, the second model is the UNet, and the third model is the CLIP Encoder.

1. In the `defaults` section of `conf/config.yaml`, update the `export` field to point to the desired Stable Diffusion
   inference configuration file. For example,
   if you want to use the `dreambooth/export_dreambooth.yaml` configuration, change the `export` field
   to `dreambooth/export_dreambooth`.
   ```yaml
    defaults:
      - export: dreambooth/export_dreambooth
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `export` stage is included. For example,
   ```yaml
    stages:
      - export
      ...
   ```
3. Configure `infer.num_images_per_prompt` of the `conf/export/dreambooth/export_dreambooth.yaml` file to set the
   batch_size to use for the ONNX and
   NVIDIA TensorRT models.

**Remarks**:

1. To load a pretrained checkpoint for inference, set the `restore_from_path` field in the `model` section to the path
   of the pretrained checkpoint in `.nemo` format in `conf/export/dreambooth/export_dreambooth.yaml`.

## 6. Deploying the NeMo Megatron Model

### 6.1 Setup

Prior to deploying a model or pipeline, the model or pipeline must be exported following the steps in section 5.8.
No other additional setup is required as the NeMo container comes with the relevant NVIDIA Triton Inference Server
libraries
preinstalled and ready to go.

### 6.2 Start NVIDIA Triton Inference Server

Starting the NVIDIA Triton Inference Server is a simple command. First, however, please read the model specific section
below
to make sure everything is in the correct place.
To start the NVIDIA Triton Inference Server

```
/opt/tritonserver/bin/tritonserver --log-verbose 2 --model-repository /opt/NeMo-Megatron-Launcher/deployment/server --model-control-mode=explicit --load-model <model>
```

`<model>` can be substitued for the `stable_diffusion`, `instruct_pix2pix`, `clip_trt`, `clip_vision_trt`, `vit_trt`.

#### 6.2.1 Stable Diffusion, Dreambooth

For Stable Diffusion and Dreambooth, copy the generated `plan` directory to the `deployment/server/stable_diffusion/1/`
directory.

#### 6.2.2 Instruct Pix2Pix

For Instruct Pix2Pix, copy the generated `plan` directory to the `deployment/server/instruct_pix2pix/1/` directory.

#### 6.2.3 Vision Transformer

Move the generated `.plan` file to `deployment/server/vit_trt/1/model.plan`.

#### 6.2.4 CLIP

Move the generated `.plan` file to `deployment/server/clip_vision_trt/1/model.plan`. For this model, two separate Triton
models need to be loaded
`--load-model clip_vision_trt --load-model clip_trt`. Querying `clip_trt` will provide tokenization and automatically
call `clip_vision_trt` using BLS.

### 6.3 Query NVIDIA Triton Inference Server

In a separate instance of the NeMo container, we can setup a client to query the server. In `deployment/client`, there
are a few examples of the clients.

#### 6.3.1 Stable Diffusion and Dreambooth

At query time, the values, `seed`, `unconditional_guidance_scale`, `inference_steps`, `eta` can be used as optional
inputs. If these are not set, the defaults are the values set during export.
The return is a single numpy array containing `num_images_per_prompt` images.

#### 6.3.2 Instruct Pix2Pix

At query time, the values, `seed`, `text_cfg_scale`, `steps`, `image_cfg_scale` can be used as optional inputs. If these
are not set, the defaults are the values set during export.
The return is a single numpy array containing `num_images_per_prompt` images. In the client example, make sure to set
the path to the input image.

## 7. Performance

### 7.1. Vision Transformer Results

#### 7.1.1. Training Accuracy Results

Training Accuracy: NVIDIA DGX SuperPOD (4 x 8 x A100 80GB for ViT B/16 Model)

We pretrained a ViT B/16 model on the ImageNet 1K dataset and fine-tuned it on the same dataset at a higher resolution,
following the recipe outlined in the [ViT paper](https://arxiv.org/abs/2010.11929). As a result, we achieved a Top-1
accuracy of **79.47%**, which is **1.56%** higher than the reported accuracy of 77.91% in the paper. Below are the
highlights of the training and fine-tuning recipe we used:

- Model: ViT B/16
- Dataset: ImageNet 1K
- Pretraining:
    - Epochs: 300
    - Batch Size: 4096
    - Training Resolution: 224
    - Optimizer: Adam (0.9, 0.999)
    - Base Learning Rate: 3.00E-03
    - Learning Rate Decay: Cosine
    - Weight Decay: 0.3
    - Dropout: 0.1
- Fine-tuning:
    - Steps: 20,000
    - Batch Size: 512
    - Fine-tuning Resolution: 512
    - Optimizer: SGD (0.9)
    - Base Learning Rate: 0.003 - 0.06
    - Learning Rate Decay: Cosine
    - Weight Decay: 0

#### 7.1.2. Training Performance Results

We measured the throughput of training Vision Transformer models on
different numbers of DGX A100 nodes and DGX H100 nodes, and we achieved near-linear
scaling on both platforms.

We are comparing the out-of-box performance on DGX H100 machines with the same configuration from DGX A100 machines.
This comparison is an apple-to-apple assessment, ensuring that we evaluate the relative performance of the two machine
types under equivalent conditions and configurations.

The tables and charts below show the performance results.

- NVIDIA DGX SuperPODs (16 x 8 x A100 80GB for ViT g/14 model)

|          |                                  |        |         |         | Nodes   |          |
|----------|----------------------------------|--------|---------|---------|---------|----------|
|          |                                  | 1      | 2       | 4       | 8       | 16       |
|          | Samples per Second               | 708.06 | 1369.35 | 2729.57 | 5397.29 | 10837.41 |
| ViT g/14 | Perfect Linear Scaling (Samples) | 708.06 | 1416.13 | 2832.25 | 5664.50 | 11329.00 |
|          | Speedup                          | 1x     | 1.93x   | 3.85x   | 7.62x   | 15.31x   |

<img src="img/ViT g_14 NeMo Megatron Throughput (A100).svg"/>

- NVIDIA DGX SuperPODs (16 x 8 x H100 80GB for ViT g/14 model)

|          |                                  |      |       |       | Nodes |        |
|----------|----------------------------------|------|-------|-------|-------|--------|
|          |                                  | 1    | 2     | 4     | 8     | 16     |
|          | Samples per Second               | 1527 | 3006  | 5900  | 11743 | 24002  |
| ViT g/14 | Perfect Linear Scaling (Samples) | 1527 | 3054  | 6109  | 12219 | 24439  |
|          | Speedup                          | 1x   | 1.97x | 3.86x | 7.69x | 15.71x |

<img src="img/ViT g_14 NeMo Megatron Throughput (H100).svg"/>

- DGX A100 vs. DGX H100: A Comparative Analysis of Vision Transformer Training

| Model       | Nodes | Global Batch Size | Micro Batch Size | Precision | Global Batch / Sec (A100) | Global Batch / Sec (H100) | Speedup (x) |
|-------------|-------|-------------------|------------------|-----------|---------------------------|---------------------------|-------------|
| ViT B/16    | 2     | 4096              | 256              | bf16 (O2) | 2.65                      | 5.88                      | 2.2         |
| ViT L/16    | 2     | 4096              | 256              | bf16 (O2) | 1.34                      | 2.84                      | 2.1         |
| ViT H/14    | 4     | 4096              | 128              | bf16 (O2) | 1.02                      | 2.17                      | 2.1         |
| ViT g/14    | 4     | 4096              | 64               | bf16 (O2) | 0.70                      | 1.52                      | 2.2         |
| ViT bigG/14 | 4     | 4096              | 32               | bf16 (O2) | 0.42                      | 0.86                      | 2.1         |

<img src="img/Vision Transformer Training Throughput Comparison.svg"/>

#### 7.1.3. Inference Performance Results

Latency times are taken as starting with an image on CPU and stopped on output.
For framework we use the Torch Automated Mixed Precision (AMP) for FP16 computation. For TRT, we export the various
models
with the FP16 acceleration. We use the optimized TRT engine setup present in the deployment directory to get the numbers
in the same environment as the framework.

GPU: NVIDIA DGX A100 (1x A100 80 GB)
Batch Size: Number of Images in a Batch

| Model    | Batch Size | TRT FP16 Latency (s) | FW FP 16 (AMP) Latency (s) | TRT vs FW Speedup (x) |
|----------|------------|----------------------|----------------------------|-----------------------|
|          | 1          | 0.006                | 0.014                      | 2.3                   |
|          | 2          | 0.008                | 0.015                      | 1.9                   |
| ViT b/16 | 4          | 0.011                | 0.015                      | 1.4                   |
|          | 8          | 0.018                | 0.017                      | 1.0                   |

### 7.2. CLIP Results

#### 7.2.1. Training Accuracy Results

Training Accuracy: NVIDIA DGX SuperPOD (8 x 8 x A100 80GB for CLIP B/32 Model)

We followed the training recipe from [Open CLIP blog](https://laion.ai/blog/large-openclip/#12b-samples-seen) to verify
our training pipeline. Our results are displayed in the table below:

| Framework | Dataset               | Model Name | Batch Size | Samples Seen | ImageNet Top-1 |
|-----------|-----------------------|------------|------------|--------------|----------------|
| OpenCLIP  | LAION 400M            | B/32       | 32k        | 12B          | 62.90%         |
| NeMo      | Our Multimodal Blend* | B/32       | 32k        | 12B          | 60.13%         |

\**Our multimodal dataset is originated from Common Crawl with custom filtering.*

We believe the final accuracy difference is due to the dataset, as LAION 400M is filtered with CLIP scores. To ensure
our implementation is consistent with OpenCLIP, we trained OpenCLIP with our dataset and found out that the loss curve
and validation accuracy were nearly identical to NeMo's CLIP.

#### 7.2.2. Training Performance Results

We measured the throughput of training CLIP models on
different numbers of DGX A100 nodes and DGX H100 nodes, and we achieved near-linear
scaling on both platforms.

We are comparing the out-of-box performance on DGX H100 machines with the same configuration from DGX A100 machines.
This comparison is an apple-to-apple assessment, ensuring that we evaluate the relative performance of the two machine
types under equivalent conditions and configurations.

The tables and charts below show the performance results.

- NVIDIA DGX SuperPODs (16 x 8 x A100 80GB for CLIP g/14 model)

|           |                                  |        |         |         | Nodes   |         |
|-----------|----------------------------------|--------|---------|---------|---------|---------|
|           |                                  | 1      | 2       | 4       | 8       | 16      |
|           | Samples per Second               | 621.90 | 1230.88 | 2446.72 | 4863.68 | 9650.36 |
| CLIP g/14 | Perfect Linear Scaling (Samples) | 621.90 | 1243.81 | 2487.61 | 4975.22 | 9950.44 |
|           | Speedup                          | 1x     | 1.98x   | 3.93x   | 7.82x   | 15.52x  |

<img src="img/CLIP g_14 NeMo Megatron Throughput (A100).svg"/>

- NVIDIA DGX SuperPODs (16 x 8 x H100 80GB for CLIP g/14 model)

|           |                                  |         |         |         | Nodes   |          |
|-----------|----------------------------------|---------|---------|---------|---------|----------|
|           |                                  | 1       | 2       | 4       | 8       | 16       |
|           | Samples per Second               | 1039.81 | 2005.48 | 4013.33 | 7627.56 | 14913.53 |
| CLIP g/14 | Perfect Linear Scaling (Samples) | 1039.81 | 2079.61 | 4159.22 | 8318.44 | 16636.88 |
|           | Speedup                          | 1x      | 1.93x   | 3.86x   | 7.34x   | 14.34x   |

<img src="img/CLIP g_14 NeMo Megatron Throughput (H100).svg"/>

- DGX A100 vs. DGX H100: A Comparative Analysis of CLIP Training

| Model     | Nodes | Global Batch Size | Micro Batch Size | Precision | Global Batch / Sec (A100) | Global Batch / Sec (H100) | Speedup (x) |
|-----------|-------|-------------------|------------------|-----------|---------------------------|---------------------------|-------------|
| CLIP B/32 | 4     | 16000             | 500              | bf16 (O2) | 1.49                      | 4.83                      | 3.2         |
| CLIP H/14 | 4     | 3584              | 112              | bf16 (O2) | 0.96                      | 1.92                      | 2.0         |
| CLIP g/14 | 4     | 2560              | 80               | bf16 (O2) | 1.08                      | 2.25                      | 2.1         |

<img src="img/CLIP Training Throughput Comparison.svg"/>

#### 7.2.3. Inference Performance Results

Latency times are taken as starting with an image on CPU and text input (of length 64) and stopped on output.
For framework we use the Torch Automated Mixed Precision (AMP) for FP16 computation. For TRT, we export the various
models
with the FP16 acceleration. We use the optimized TRT engine setup present in the deployment directory to get the numbers
in the same environment as the framework.

GPU: NVIDIA DGX A100 (1x A100 80 GB)
Batch Size: Number of Images in a Batch

| Model     | Batch Size | TRT FP16 Latency (s) | FW FP 16 (AMP) Latency (s) | TRT vs FW Speedup (x) |
|-----------|------------|----------------------|----------------------------|-----------------------|
|           | 1          | 0.014                | 0.032                      | 2.3                   |
|           | 2          | 0.014                | 0.033                      | 2.4                   |
| CLIP B/32 | 4          | 0.014                | 0.028                      | 2.0                   |
|           | 8          | 0.015                | 0.028                      | 1.9                   |

### 7.3. Stable Diffusion Results

#### 7.3.1. Training Accuracy Results

We evaluate Stable Diffusion model with FID-CLIP curve, and comparing it to other open-source ckpt at same scale of
consumed sample.

FID (Fréchet Inception Distance) is a metric used to evaluate the quality of generated images in machine learning. It
measures the distance between the real image distribution and the distribution of generated images using the features
extracted by a pre-trained Inception model.

The VIT-L/14 version of the CLIP model was utilized to assess the relevance between image prompts and generated images.

The evaluation was conducted using different classifier-free guidance scales, specifically 1.5, 2.0, 3.0, 4.0, 5.0, 6.0,
7.0, and 8.0. The evaluation process involved generating 30,000 images from randomly selected prompts from the COCO2014
validation dataset, with 50 PLMS steps, and evaluating the results at a resolution of 256x256.

We have referred to but made certain modifications to the training recipe outlined
in [Stable Diffusion Model Cards posted on Huggingface](https://huggingface.co/CompVis/stable-diffusion-v1-4).

\**Our multimodal dataset is originated from Common Crawl with custom filtering.*

Below, we present the outcomes obtained from our own checkpoint following Section 5.3.3, which can be compared to those
of the open-source Stable Diffusion 1.5.

<img src='img/Stable Diffusion FID-CLIP.png'/>

#### 7.3.2. Training Performance Results

We measured the throughput of training Stable Diffusion models on
different numbers of DGX A100 nodes and DGX H100 nodes, and we achieved near-linear
scaling on both platforms.

We are comparing the out-of-box performance on DGX H100 machines with the same configuration from DGX A100 machines.
This comparison is an apple-to-apple assessment, ensuring that we evaluate the relative performance of the two machine
types under equivalent conditions and configurations.

The tables and charts below show the performance results.

- NVIDIA DGX SuperPODs (16 x 8 x A100 80GB for Stable Diffusion Res=512 model)

|                          |                                  |        |        |        | Nodes   |         |
|--------------------------|----------------------------------|--------|--------|--------|---------|---------|
|                          |                                  | 1      | 2      | 4      | 8       | 16      |
|                          | Samples per Second               | 199.98 | 390.60 | 786.78 | 1504.99 | 2952.49 |
| Stable Diffusion Res=512 | Perfect Linear Scaling (Samples) | 199.98 | 399.97 | 799.94 | 1599.87 | 3199.75 |
|                          | Speedup                          | 1x     | 1.95x  | 3.93x  | 7.53x   | 14.76x  |

<img src="img/Stable Diffusion (Res=512) NeMo Megatron Throughput (A100).svg"/>

- NVIDIA DGX SuperPODs (16 x 8 x H100 80GB for Stable Diffusion Res=512 model)

|                          |                                  |        |        |         | Nodes   |         |
|--------------------------|----------------------------------|--------|--------|---------|---------|---------|
|                          |                                  | 1      | 2      | 4       | 8       | 16      |
|                          | Samples per Second               | 419.47 | 840.86 | 1591.79 | 3090.85 | 6056.48 |
| Stable Diffusion Res=512 | Perfect Linear Scaling (Samples) | 419.47 | 838.93 | 1677.86 | 3355.73 | 6711.45 |
|                          | Speedup                          | 1x     | 2x     | 3.79x   | 7.37x   | 14.44x  |

<img src="img/Stable Diffusion (Res=512) NeMo Megatron Throughput (H100).svg"/>

- DGX A100 vs. DGX H100: A Comparative Analysis of Stable Diffusion Training

| Model                      | Nodes | Global Batch | Micro Batch | Precision | Sec/Batch (A100) | Sec/Batch (H100) | Speedup (x) |
|----------------------------|-------|--------------|-------------|-----------|------------------|------------------|-------------|
| Stable Diffusion (Res=256) | 4     | 4096         | 128         | amp fp16  | 0.829            | 1.709            | 2.1         |
| Stable Diffusion (Res=512) | 4     | 1024         | 32          | amp fp16  | 0.758            | 1.603            | 2.1         |

<img src="img/Stable Diffusion Training Throughput Comparison.svg"/>

#### 7.3.3. Inference Performance Results

Latency times are started directly before the text encoding (CLIP) and stopped directly after the output image
decoding (VAE).
For framework we use the Torch Automated Mixed Precision (AMP) for FP16 computation. For TRT, we export the various
models
with the FP16 acceleration. We use the optimized TRT engine setup present in the deployment directory to get the numbers
in the same environment as the framework.

GPU: NVIDIA DGX A100 (1x A100 80 GB)
Batch Size: Synonymous with `num_images_per_prompt`

| Model                      | Batch Size | Sampler | Inference Steps | TRT FP 16 Latency (s) | FW FP 16 (AMP) Latency (s) | TRT vs FW Speedup (x) |
|----------------------------|------------|---------|-----------------|-----------------------|----------------------------|-----------------------|
| Stable Diffusion (Res=512) | 1          | PLMS    | 50              | 0.9                   | 3.3                        | 3.7                   |
| Stable Diffusion (Res=512) | 2          | PLMS    | 50              | 1.7                   | 5.2                        | 3.1                   |
| Stable Diffusion (Res=512) | 4          | PLMS    | 50              | 2.9                   | 9.2                        | 3.2                   |  

### 7.4. Instruct Pix2Pix Results

#### 7.4.1. Training Quality Results

Instruct Pix2Pix is an image editing tool that transforms original images based on user instructions. For example, when
provided with a photo of cute toy duck, the AI can seamlessly edit the image according to your creative vision.

Here are some examples generated using our NeMo Stable Diffusion 1.2 model, fine-tuned with NeMo Instruct Pix2Pix. For
each instruction, we showcase 8 distinct images generated from different seeds:

- Original photo
  <img src="img/toy_duck.jpg" width="30%" />
- Instruction: Make it in desert
  <img src="img/make_it_in_desert_7.5_1.2_1234_combine.jpg"/>
- Instruction: Make it Van Gogh style
  <img src="img/make_it_Van_Gogh_style_7.5_1.2_1234_combine.jpg"/>
- Instruction: Make it in a pool
  <img src="img/make_it_in_a_pool_7.5_1.2_1234_combine.jpg"/>

#### 7.4.2. Inference Performance Results

Latency times are started directly before the text encoding (CLIP) and stopped directly after the output image
decoding (VAE).
For framework we use the Torch Automated Mixed Precision (AMP) for FP16 computation. For TRT, we export the various
models
with the FP16 acceleration. We use the optimized TRT engine setup present in the deployment directory to get the numbers
in the same environment as the framework.

GPU: NVIDIA DGX A100 (1x A100 80 GB)
Batch Size: Synonymous with `num_images_per_prompt`

| Model                      | Batch Size | Sampler | Inference Steps | TRT FP 16 Latency (s) | FW FP 16 (AMP) Latency (s) | TRT vs FW Speedup (x) |
|----------------------------|------------|---------|-----------------|-----------------------|----------------------------|-----------------------|
| Instruct Pix2Pix (Res=256) | 1          | N/A     | 100             | 1.0                   | 3.6                        | 3.6                   |
| Instruct Pix2Pix (Res=256) | 2          | N/A     | 100             | 1.3                   | 3.7                        | 2.8                   |
| Instruct Pix2Pix (Res=256) | 4          | N/A     | 100             | 2.2                   | 4.9                        | 2.2                   |

### 7.5. DreamBooth Results

#### 7.5.1. Training Quality Results

Here we show some insteresting results as an example of dreambooth script.

Prompt: A 'sks' dog in a bucket.

<img src="img/Dreambooth dog in a bucket.png" width="30%">

Prompt: A 'sks' dog in Acropolis.

<img src="img/Dreambooth dog at Acropolis.png" width="30%">

Prompt: A 'sks' dog in front of Eiffel tower.

<img src="img/Dreambooth Eiffel towel.png" width="30%">

Prompt: A 'sks' dog mecha robot.

<img src="img/Dreambooth mecha robot.png" width="30%">

#### 7.5.2. Inference Performance Results

Latency times are started directly before the text encoding (CLIP) and stopped directly after the output image
decoding (VAE).
For framework we use the Torch Automated Mixed Precision (AMP) for FP16 computation. For TRT, we export the various
models
with the FP16 acceleration. We use the optimized TRT engine setup present in the deployment directory to get the numbers
in the same environment as the framework.

GPU: NVIDIA DGX A100 (1x A100 80 GB)
Batch Size: Synonymous with `num_images_per_prompt`

| Model                | Batch Size | Sampler | Inference Steps | TRT FP 16 Latency (s) | FW FP 16 (AMP) Latency (s) | TRT vs FW Speedup (x) |
|----------------------|------------|---------|-----------------|-----------------------|----------------------------|-----------------------|
| Dreambooth (Res=256) | 1          | DDIM    | 100             | 2.0                   | 5.6                        | 2.8                   |
| Dreambooth (Res=256) | 2          | DDIM    | 100             | 3.1                   | 9.0                        | 2.9                   |
| Dreambooth (Res=256) | 4          | DDIM    | 100             | 5.7                   | 16.0                       | 2.8                   |

## 8. Changelog

**NeMo Megatron 23.01**

* BERT with tensor parallelism support (training only)
* BERT with pipeline parallelism support (training only)
* Sequence Parallelism and Selective Activation Checkpointing for BERT (training only)
* Interleaved Pipeline Scheduling for BERT
* Distributed Adam Optimizer for BERT
* AugoConfigurator for BERT
* 110M, 4B, 20B, and 100B BERT training configurations
* Support for the Mixture of Experts for T5 (no expert parallelism, training only)
* Performance improvement for GPT-3 P-Tuning (20% - 25% speed-up)
* ALiBi Position Embeddings for T5 and mT5 (training only)
* Log total model size (across modal parallel ranks) for GPT-3, T5, mT5, and BERT

**NeMo Megatron 22.11**

* Interleaved Pipeline Scheduling for GPT-3 (training only)
* FP8 support using Transformer Engine (training only)
* Distributed Adam Optimizer for T5 and mT5
* P-Tuning and Prompt Tuning for GPT-3 with Sequence Parallelism
* Training configurations improved throughput by 7.9% (5B GPT-3), 9.6% (3B T5), 4.3% (11B T5), 52.4% (23B T5), and
  26.6% (41B T5)

**NeMo Megatron 22.09**

* NeMo Megatron supports training and inference containers on OCI. For detail orchestration scripts, reach out
  to [oci_nm@nvidia.com](mailto:oci_nm@nvidia.com)
* P-Tuning and Prompt Tuning for T5 and mT5 with pipeline parallelism (training only)
* Adapter learning for GPT-3 and T5 with tensor parallelism and pipeline parallelism (training only)
* IA3 learning for GPT-3 and T5 with tensor parallelism and pipeline parallelism (training only)
* AugoConfigurator to find the highest throughput configs for training on Base Command Platform
* AugoConfigurator: parallel inference hyperparameter search for GPT-3 on Base Command Manager

**NeMo Megatron 22.08.01**

* Cloud service providers: support for Amazon Web Services (performance validated up to 20 `p4d.24xlarge` instances)
* Cloud service providers: switched orchestration from Azure CycleCloud to NVIDIA Nephele for Microsoft Azure

**NeMo Megatron 22.08**

* Distributed Adam Optimizer for GPT-3
* Asymmetric encoder-decoder configuration for T5 and mT5
* Support for untying embeddings from the classifier layer for T5 and mT5
* Relative Position Embeddings for T5 and mT5 (pipeline parallelism>=3)
* P-Tuning and Prompt Tuning for T5 and mT5 with tensor parallelism (training only)
* Code refactor - improved consistency and readability of configurations and logs
* SQuAD fine-tuning and evaluation support for T5 with pipeline parallelism =<2
* XQuAD fine-tuning and evaluation support for mT5 with pipeline parallelism =<2

**NeMo Megatron 22.06-hotfix.01**

* Fix: AugoConfigurator for T5 and mT5
* Fix: Evaluation harness in GPT-3
* Fix: Prompt learning in GPT-3
* Fix: Out of memory when pretraining GPT-3 with Sequence Parallelism

**NeMo Megatron 22.06**

* Sequence Parallelism and Selective Activation Checkpointing for GPT-3
* Relative Position Embeddings for T5
    * We used mC4 dataset (24 Languages) for pretraining the mT5 and verified our results on KNLI, KorQuAD, KLUE-STS,
      and XNLI tasks
* AugoConfigurator update with Sequence Parallelism and Selective Activation Checkpointing for GPT-3
* AugoConfigurator: support for DGX A100 40GB configurations for GPT-3, T5, and mT5
* P-Tuning and Prompt Tuning for GPT-3 with pipeline parallelism (training only)
* Operation fusions for higher training throughput (2%-7% speed-up)
* Default GPT-3 configurations changed to include Sequence Parallelism and Selective Activation Checkpointing: 20B (
  speed-up: 14%), 40B (speed-up: 9%), 175B (speed-up: 15%)

**NeMo Megatron 22.05.01**

* Cloud service providers: support for Microsoft Azure (performance validated up to 36 `Standard_ND96amsr_A100_v4`
  instances)
* Cluster validation tools (DGMI, NCCL)
* 20B GPT-3 training configuration improved by 2.7% for higher throughput

**NeMo Megatron 22.05**

* Asynchronous gradient all-reduce for GPT-3, T5, mT5 models with pipeline parallel size equal to 1
* P-Tuning and Prompt Tuning for GPT-3 with tensor parallelism (training only)
* AugoConfigurator to find the highest throughput configs for training and inference on Base Command Manager
* Custom tokenizer support (training only)
* GPT-3 with pipeline parallelism support on Base Command Manager (inference)
* Hyperparameters for text generation: top-p, top-k, and temperature

**NeMo Megatron 22.04**

* T5 with pipeline parallelism support (training only)
* Switched from GeLU to GeGLU as activation function for T5
* mT5 with tensor parallelism and pipeline parallelism support (training only)
* 11B, 23B, and 41B T5 training configurations
* 170M, 390M, and 3B mT5 training configurations
* Automatic and configurable Non-Uniform Memory Access (NUMA) mapping

**NeMo Megatron 22.03**

* T5 with tensor parallelism support (optimized for <20B parameters, training only)
* 220M and 3B T5 training configurations
* GLUE fine-tuning and evaluation support for T5

**NeMo Megatron 22.02**

* GPT-3 with pipeline parallelism support (training only)
* 40B and 175B GPT-3 training configurations

**NeMo Megatron 22.01**

* GPT-3 with tensor parallelism support on Base Command Platform
* O2-style AMP (accelerated training of larger models)
* Chatbot sample application using your trained GPT-3 model
* Training metric monitoring and visualization with Weights & Biases

## 9. Known Issues

Fixes for the following issues will be released shortly:

* The inference hyperparameter search is not available in this release for T5 and mT5
* Accuracy and performance measurement for GPT-3 is currently not supported. Please use the NeMo Megatron 22.05
  inference container to use this feature
* For running inference on BCP please use the NeMo Megatron 22.03 inference container
* The fine-tuning SQuAD results for T5 are lower than expected
