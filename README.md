# NeMo Multimodal

## Early Access

Scripts and code to provide end-to-end data preparation and training for
NeMo Multimodal.

The most recent version of the README can be found
at [https://ngc.nvidia.com/containers/ea-bignlp:bignlp-training](https://ngc.nvidia.com/containers/ea-bignlp:bignlp-training).

## Table of contents
- [NeMo Multimodal](#nemo-multimodal)
  - [Early Access](#early-access)
  - [Table of contents](#table-of-contents)
  - [1. Release Notes](#1-release-notes)
  - [2. Model Overview](#2-model-overview)
    - [2.1. Vision Transformer (ViT)](#21-vision-transformer-vit)
    - [2.2. CLIP](#22-clip)
    - [2.3. Stable Diffusion](#23-stable-diffusion)
    - [2.4. InstructPix2Pix](#24-instructpix2pix)
    - [2.5. DreamBooth](#25-dreambooth)
    - [2.6. ControlNet](#26-controlnet)
    - [2.7. Imagen](#27-imagen)
  - [3. Feature Matrix](#3-feature-matrix)
    - [3.1. ViT Models](#31-vit-models)
    - [3.2. CLIP Models](#32-clip-models)
    - [3.3. Stable Diffusion](#33-stable-diffusion)
    - [3.4. InstructPix2Pix / DreamBooth / ControlNet Models](#34-instructpix2pix--dreambooth--controlnet-models)
    - [3.5 Imagen Models](#35-imagen-models)
  - [4. Setup Details](#4-setup-details)
  - [5. Cloud Service Providers](#5-cloud-service-providers)
    - [5.1. Cluster Bring-Up](#51-cluster-bring-up)
      - [5.1.1. Common](#511-common)
      - [5.1.2. OCI](#512-oci)
      - [5.1.3. AWS](#513-aws)
    - [5.2. Cluster Validation](#52-cluster-validation)
      - [5.2.1. Validation Script Usage](#521-validation-script-usage)
      - [5.2.2. Running tests manually](#522-running-tests-manually)
    - [5.3. Config Modifications](#53-config-modifications)
      - [5.3.1. Set NCCL Topology](#531-set-nccl-topology)
      - [5.3.2. Environment Variables](#532-environment-variables)
        - [5.3.2.1. Azure Variables](#5321-azure-variables)
        - [5.3.2.2. AWS Variables](#5322-aws-variables)
  - [6. Quick Start Guide](#6-quick-start-guide)
    - [6.1. Getting Started with NeMo Multimodal](#61-getting-started-with-nemo-multimodal)
      - [6.1.1. Prepare Environment](#611-prepare-environment)
        - [6.1.1.1. Slurm](#6111-slurm)
        - [6.1.1.2. Base Command Platform](#6112-base-command-platform)
      - [6.1.2. Configure and Customize Pipeline](#612-configure-and-customize-pipeline)
        - [6.1.2.1. Cluster Configurations](#6121-cluster-configurations)
        - [6.1.2.2. Pipeline Configurations](#6122-pipeline-configurations)
        - [6.1.2.3. Environment Variables Configurations](#6123-environment-variables-configurations)
        - [6.1.2.4. NUMA Mapping Configurations](#6124-numa-mapping-configurations)
      - [6.1.3. Launch Pipeline](#613-launch-pipeline)
      - [6.1.4. Example: Pre-train Stable Diffusion 860M Model for 10 Epochs with Resolution 256](#614-example-pre-train-stable-diffusion-860m-model-for-10-epochs-with-resolution-256)
    - [6.2. Data Preparation](#62-data-preparation)
      - [6.2.1. ImageNet](#621-imagenet)
        - [6.2.1.1. ImageNet 1k](#6211-imagenet-1k)
        - [6.2.1.2. ImageNet 21k](#6212-imagenet-21k)
      - [6.2.2. Multimodal Datasets](#622-multimodal-datasets)
        - [6.2.2.1. Overview](#6221-overview)
        - [6.2.2.2. Running the Pipeline](#6222-running-the-pipeline)
        - [6.2.2.3. Configuration for Precaching](#6223-configuration-for-precaching)
          - [6.2.2.3.1. General Format](#62231-general-format)
          - [6.2.2.3.2. Precaching Config](#62232-precaching-config)
          - [6.2.2.3.3. Resume Precaching (Advanced)](#62233-resume-precaching-advanced)
      - [6.2.3. InstructPix2Pix](#623-instructpix2pix)
      - [6.2.4. MSCOCO for FID Evaluation](#624-mscoco-for-fid-evaluation)
        - [6.2.4.1. Download and Setup](#6241-download-and-setup)
        - [6.2.4.2. Preprocess Images and Captions](#6242-preprocess-images-and-captions)
      - [6.2.5. ControlNet](#625-controlnet)
    - [6.3. Model Training](#63-model-training)
      - [6.3.1. Vision Transformer Training](#631-vision-transformer-training)
      - [6.3.2. CLIP Training](#632-clip-training)
      - [6.3.3. Stable Diffusion Training](#633-stable-diffusion-training)
      - [6.3.4. InstructPix2Pix Training](#634-instructpix2pix-training)
      - [6.3.5. DreamBooth Training](#635-dreambooth-training)
      - [6.3.6. ControlNet Training](#636-controlnet-training)
      - [6.3.7. Imagen Training](#637-imagen-training)
    - [6.4. Checkpoint Conversion](#64-checkpoint-conversion)
    - [6.5. Model Fine-tuning](#65-model-fine-tuning)
      - [6.5.1. Vision Transformer Fine-tuning](#651-vision-transformer-fine-tuning)
    - [6.6. Model Evaluation](#66-model-evaluation)
      - [6.6.1. Vision Transformer Evaluation](#661-vision-transformer-evaluation)
      - [6.6.2. CLIP Evaluation](#662-clip-evaluation)
      - [6.6.3. Stable Diffusion Evaluation](#663-stable-diffusion-evaluation)
      - [6.6.4. Imagen Evaluation](#664-imagen-evaluation)
    - [6.7. Model Inference (in NeMo Framework)](#67-model-inference-in-nemo-framework)
      - [6.7.1. Vision Transformer Inference (in NeMo Framework)](#671-vision-transformer-inference-in-nemo-framework)
      - [6.7.2. CLIP Inference (in NeMo Framework)](#672-clip-inference-in-nemo-framework)
      - [6.7.3. Stable Diffusion Inference (in NeMo Framework)](#673-stable-diffusion-inference-in-nemo-framework)
      - [6.7.4. InstructPix2Pix Inference (in NeMo Framework)](#674-instructpix2pix-inference-in-nemo-framework)
      - [6.7.5. DreamBooth Inference (in NeMo Framework)](#675-dreambooth-inference-in-nemo-framework)
      - [6.7.6. ControlNet Inference (in NeMo Framework)](#676-controlnet-inference-in-nemo-framework)
      - [6.7.7. Imagen Inference (in NeMo Framework)](#673-imagen-inference-in-nemo-framework)
    - [6.8. Model Export](#68-model-export)
      - [6.8.1. Vision Transformer Export](#681-vision-transformer-export)
      - [6.8.2. CLIP Export](#682-clip-export)
      - [6.8.3. Stable Diffusion Export](#683-stable-diffusion-export)
      - [6.8.4. InstructPix2Pix Export](#684-instructpix2pix-export)
      - [6.8.5. DreamBooth Export](#685-dreambooth-export)
      - [6.8.6. ControlNet Export](#686-controlnet-export)
      - [6.8.7. Imagen Export](#687-imagen-export)
    - [6.9. Convert Checkpoints from External Sources to Nemo](#69-convert-checkpoints-from-external-sources-to-nemo)
  - [7. Deploying the NeMo Multimodal Model](#7-deploying-the-nemo-multimodal-model)
    - [7.1. Setup](#71-setup)
    - [7.2. Start NVIDIA Triton Inference Server](#72-start-nvidia-triton-inference-server)
      - [7.2.1. Stable Diffusion, DreamBooth](#721-stable-diffusion-dreambooth)
      - [7.2.2. InstructPix2Pix](#722-instructpix2pix)
      - [7.2.3. Vision Transformer](#723-vision-transformer)
      - [7.2.4. CLIP](#724-clip)
      - [7.2.5. ControlNet](#725-controlnet)
      - [7.2.6 Imagen](#725-imagen)
    - [7.3. Query NVIDIA Triton Inference Server](#73-query-nvidia-triton-inference-server)
      - [7.3.1. Stable Diffusion and DreamBooth](#731-stable-diffusion-and-dreambooth)
      - [7.3.2. InstructPix2Pix](#732-instructpix2pix)
      - [7.3.3 Imagen](#733-imagen)
  - [8. Performance](#8-performance)
    - [8.1. Vision Transformer Results](#81-vision-transformer-results)
      - [8.1.1. Training Accuracy Results](#811-training-accuracy-results)
      - [8.1.2. Training Performance Results](#812-training-performance-results)
      - [8.1.3. Inference Performance Results](#813-inference-performance-results)
    - [8.2. CLIP Results](#82-clip-results)
      - [8.2.1. Training Accuracy Results](#821-training-accuracy-results)
      - [8.2.2. Training Performance Results](#822-training-performance-results)
      - [8.2.3. Inference Performance Results](#823-inference-performance-results)
    - [8.3. Stable Diffusion Results](#83-stable-diffusion-results)
      - [8.3.1. Training Accuracy Results](#831-training-accuracy-results)
      - [8.3.2. Training Performance Results](#832-training-performance-results)
      - [8.3.3. Inference Performance Results](#833-inference-performance-results)
    - [8.4. InstructPix2Pix Results](#84-instructpix2pix-results)
      - [8.4.1. Training Quality Results](#841-training-quality-results)
      - [8.4.2. Inference Performance Results](#842-inference-performance-results)
    - [8.5. DreamBooth Results](#85-dreambooth-results)
      - [8.5.1. Training Quality Results](#851-training-quality-results)
      - [8.5.2. Inference Performance Results](#852-inference-performance-results)
    - [8.6. ControlNet Results](#86-controlnet-results)
      - [8.6.1 Training Performance Results](#861-traning-performace-results)
      - [8.6.2 Training Quality Results](#862-training-quality-results)
      - [8.6.3 Inference Performance Results](#863-inference-performance-results)
    - [8.7. Imagen Results](#87-imagen-results)
      - [8.7.1. Training Accuracy Results](#871-training-accuracy-results-1)
      - [8.7.2. Training Performance Results](#872-training-performance-results-1)
      - [8.7.3. Inference Performance Results](#873-inference-performance-results-1)
  - [9. Known Issues](#9-known-issues)

<!-- /TOC -->

## 1. Release Notes

**NeMo Multimodal 23.05**

1. Added support for distributed optimizer in ViT and CLIP models, enhancing memory
   efficiency when utilizing more nodes with higher data parallel values.

**NeMo Multimodal 23.03 (Initial Release)**

Added support for the following:

| Model/ Method | Training | Fine-Tuning | Evaulation | In-framework Inference| Export (to TensorRT and ONNX) | Triton deployment |
| :---        | :----:   |    ---: |    :----:  |    ---:     |    :----:   |          ---: |
|**Vision Transformer (ViT)**| &check;|&check;|&check;|&check;|&check;|&check;|
| **CLIP**   | &check;|    _|  &check;|&check;|&check;|&check;|
|  **Stable Diffusion (SD)**  | &check;|    _|  &check;|&check;|&check;|&check;|
| **InstructPix2Pix (for SD tuning)**| &check;|    _|  _|&check;|&check;|&check;|
|**DreamBooth (for SD tuning)**| &check;|    _|  _|&check;|&check;|&check;|
|**ControlNet (for SD tuning)**| &check;|    _|  _|&check;|&check;|&check;|

Accuracy metrics/plots and training/inference performance for all supported models included.

## 2. Model Overview

The NeMo Multimodal is a powerful extension of the NeMo framework, specifically designed for developers who aim
to efficiently train and scale multimodal models. With NeMo Multimodal, you can effortlessly train various
variants of multimodal models, such as CLIP, Stable Diffusion and more. This powerful tool is capable of
scaling your models to multiple nodes on NVIDIA DGX SuperPOD deployments.

The deep learning (DL) software stack is meticulously optimized for DGX SuperPOD configurations, utilizing NVIDIA's
InfiniBand technology to deliver efficient on-premises computing for training and inference of complex workloads.

The NeMo Multimodal utilizes model parallelism techniques to efficiently train large models that cannot fit
within the memory of a single GPU. During the training process, both tensor (intra-layer) and pipeline (inter-layer)
model parallelism are employed. Tensor model parallelism distributes individual transformer layers across multiple
devices, while pipeline model parallelism allocates different layers of a model to separate devices. For a more in-depth
understanding, please refer to [this paper](https://arxiv.org/pdf/2104.04473.pdf). We are in the process of
incorporating this feature into all our models. As of now, Tensor Parallelism is
available in both **Vision Transformer** and **CLIP** models.

### 2.1. Vision Transformer (ViT)

The Vision Transformer, commonly referred to as ViT [[Paper]](https://arxiv.org/pdf/2010.11929v2.pdf), is a foundation
model for image classification tasks in NeMo Multimodal. It
leverages a transformer-like architecture to process image patches, rather than relying on traditional convolutional
neural networks. In the ViT, an image is divided into fixed-size patches (usually 14x14 or 16x16), which are then
linearly embedded and augmented
with position embeddings. The resulting sequence of vectors is fed into a standard transformer encoder. To enable
classification, a learnable "classification token" is added to the sequence.

### 2.2. CLIP

Contrastive Language-Image Pre-training (CLIP) [[Paper]](https://arxiv.org/pdf/2103.00020.pdf) offers an efficient
method for learning image representations using natural language supervision. In essence, CLIP trains both an image
encoder and a text encoder from scratch. The goal is to predict the correct pairings of a batch of (image, text)
training examples by jointly training these encoders.

During pre-training, the model is designed to predict which images and texts form a semantically coherent pair by
maximizing the similarity between the correct (image, text) pairs while minimizing the similarity between incorrect
pairs. This contrastive learning approach ensures that CLIP learns meaningful and contextually rich representations of
both visual and textual data.

Upon completion of the pre-training phase, CLIP models can be fine-tuned for specialized downstream tasks or directly
employed for zero-shot learning. For instance, the learned text encoder generates high-level representations by
embedding captions in **Stable Diffusion**. This approach facilitates seamless image and text representation
learning and has demonstrated exceptional effectiveness across a diverse range of applications.

### 2.3. Stable Diffusion

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

### 2.4. InstructPix2Pix

[InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix/) introduces a method for editing images based on
human-written instructions. Given an input image and a textual directive, the model follows these instructions to modify
the image accordingly.

NeMo Multimodal offers a training pipeline for conditional diffusion models using the edit dataset.
Additionally, we provide a tool that generates modified images based on user-written instructions during the inference
process.

### 2.5. DreamBooth

DreamBooth is a fine-tuning technique and a solution to personalize large diffusion models like Stable Diffusion, which are powerful but lack the
ability to mimic subjects of a given reference set. With DreamBooth, you only need a few images of a specific subject to
fine-tune a pretrained text-to-image model, so that it learns to bind a unique identifier with a special subject. This
unique identifier can then be used to synthesize fully-novel photorealistic images of the subject contextualized in
different scenes.

DreamBooth provides a new prior preservation loss, which enables synthesizing the subject in diverse scenes, poses,
views, and lighting conditions that do not appear in the reference images. With this new approach, DreamBooth achieves
several previously-unassailable tasks, including subject recontextualization, text-guided view synthesis, appearance
modification, and artistic rendering, while still preserving the subject's key features.


### 2.6. ControlNet

[ControlNet](https://github.com/lllyasviel/ControlNet) is a neural network structure to control diffusion models by adding extra conditions.
It copys the weights of neural network blocks into a "locked" copy and a "trainable" copy. The "trainable" one learns your condition. The "locked" one preserves your model. In this way, the ControlNet can reuse the SD encoder as a deep, strong, robust, and powerful backbone to learn diverse controls.

NeMo Multimodal provides a training pipeline and example implementation for generating images based on segmentation maps. Users have the flexibility to explore other implementations using their own control input dataset and recipe.


### 2.7. Imagen

[Imagen](https://arxiv.org/abs/2205.11487) is a multi-stage text-to-image diffusion model with an unprecedented degree of photorealism and a deep level of language understanding. Given a text prompt, Imagen first generates an image at a 64x64 resolution and then upsamples the generated image to 256x256 and 1024x1024 resolutions, all using diffusion models.

NeMo Imagen provides various options to fully cusotomize the Imagen training. For super-resolution (SR) model, we support both regular UNet and efficient UNet as proposed in the paper.
## 3. Feature Matrix

### 3.1. ViT Models

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
| Distributed Optimizer    | Yes                                                      | N/A                                                                                                                                           |
| TorchInductor            | No                                                       | N/A                                                                                                                                           |
| Flash Attention          | No                                                       | N/A                                                                                                                                           |

### 3.2. CLIP Models

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
| Distributed Optimizer    | Yes                                                      | N/A                                                                                                                                           |
| TorchInductor            | No                                                       | N/A                                                                                                                                           |
| Flash Attention          | No                                                       | N/A                                                                                                                                           |

### 3.3. Stable Diffusion

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
| NHWC GroupNorm           | Yes                                                      | Yes                                                                                                                                           |

### 3.4. InstructPix2Pix / DreamBooth / ControlNet Models

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

### 3.5 Imagen Models
| Feature                  | Training                                                 | Inference                                                                                                                                     |
|--------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Data parallelism         | Yes                                                      | N/A                                                                                                                                           |
| Tensor parallelism       | No                                                       | No                                                                                                                                            |
| Pipeline parallelism     | No                                                       | No                                                                                                                                            |
| Sequence parallelism     | No                                                       | No                                                                                                                                            |
| Activation checkpointing | Yes                                                      | Yes                                                                                                                                           |
| FP32/TF32                | Yes                                                      | Yes                                                                                                                                           |
| AMP/FP16                 | No                                                       | Yes                                                                                                                                          |
| AMP/BF16                 | Yes                                                      | Yes                                                                                                                                           |
| BF16 O2                  | No                                                       | No                                                                                                                                            |
| TransformerEngine/FP8    | No                                                       | No                                                                                                                                            |
| Multi-GPU                | Yes                                                      | Yes                                                                                                                                           |
| Multi-Node               | Yes                                                      | Yes                                                                                                                                           |
| Inference deployment     | N/A                                                      | [NVIDIA Triton supported](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton) |
| SW stack support         | Slurm DeepOps/Base Command Manager/Base Command Platform | Slurm DeepOps/Base Command Manager/Base Command Platform                                                                                      |
| NVfuser                  | No                                                       | N/A                                                                                                                                           |
| Distributed Optimizer    | No                                                       | N/A                                                                                                                                           |
| Flash Attention          | Yes                                                      | N/A                                                                                                                                           |

## 4. Setup Details

| Software             | EA                 |
|----------------------|--------------------|
| NVIDIA Triton        | 2.31.0             |
| FasterTransformer    | v5.3+4402759e      |
| PyTorch              | 1.14.0a0+44dac51   |
| NeMo                 | 1.17.0+b36a96a     |
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

## 5. Cloud Service Providers

### 5.1. Cluster Bring-Up

#### 5.1.1. Common

To set up a Slurm cluster for NeMo Multimodal, we recommend using [Nephele](https://github.com/nvidia/nephele). This
cluster deployment tool has been tested on Azure, AWS, and Oracle Cloud.
We recommend hosting Nephele on a new VM instance in the CSP of your choice. To get started:

- Clone the Nephele repo
- Install the dependencies
- Modify `nephele.conf`
    - Add your CSP credentials
    - Change `REPLICAS_x8a100` to the number of nodes in your desired cluster

You can then run `./nephele init` and `./nephele create`.

We also recommend mounting an external persistent NFS once the cluster is up and running (ensure it is mounted on all
nodes) and using this to configure and run NeMo Multimodal.

The above steps apply to all CSPs, including Azure, AWS, and OCI.
Some modifications are necessary for OCI and AWS and are detailed below.
Note that for OCI, a custom image must be imported, which should be done before running `./nephele create`.

#### 5.1.2. OCI

NeMo Multimodal supports running training and inference containers on OCI. For detail orchestration scripts, reach out
to [oci_nm@nvidia.com](mailto:oci_nm@nvidia.com)

#### 5.1.3. AWS

To launch jobs on AWS, the EFA driver and NCCL plugin first need to be installed on top of the training container.
We recommend building a new container image with Docker, then creating an Enroot image.

On the scheduler node:

- Install Docker
- Build the image with EFA drivers and NCCL plugin from `csp_tools/aws/Dockerfile`
- Run the following command on the Docker image to create an Enroot image:

```
    enroot import --output nemo_megatron_training.sqsh dockerd://<image name>:<tag>
```

- Move the `.sqsh` file to the root of NeMo-Megatron-Launcher
- Set the container path in `launcher_scripts/conf/config.yaml` to the new Enroot image:

```
container: /path/to/nemo_megatron_launcher/nemo_megatron_training.sqsh
```

### 5.2. Cluster Validation

Before running the cluster validation script, ensure your NGC credentials have been added
to `~/.config/enroot/.credentials` on all nodes.

The cluster validation script at `csp_tools/<csp>/cluster_validation.sh` will run GPU diagnostics and test NCCL
node-to-node bus bandwidth.
The logs from these tests will be stored at `results/cluster_validation`. The script will list any nodes that fail these
tests.
These nodes should be replaced or restarted through the CSP UI.

#### 5.2.1. Validation Script Usage

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

To test all 8 idle nodes, run the script as follows:

```
bash cluster_validation.sh --nodes=8 --nodelist=x8a100-[0000-0007] --partition=x8a100
```

By default, the script will run both the GPU diagnostics and the NCCL test. You can choose to run only one or the other
by specifying:

- `--dcgm`: to run GPU diagnostics only
- `--nccl`: to run NCCL test only

See `bash cluster_validation.sh -h` for more information.

#### 5.2.2. Running tests manually

The `cluster_validation.sh` script is essentially a wrapper of the 2 Slurm job scripts in the CSP directories. If you
prefer, you can run these jobs manually.
Make sure to use the slurm job script in your corresponding CSP's path (`csp_tools/<csp>/dcgmi_diag.sh`
and `csp_tools/<csp>/nccl.sh`)

For the GPU diagnostics job, provide these arguments when submitting the job to Slurm:

```
sbatch -p <partition> -w <node list> -o <job log file> dcgmi_diag.sh
```

For the NCCL test job, `cluster_validation.sh` performs a pair-wise sweep of the nodes, as this is a sufficient test,
but you can test with a different number of nodes if desired.

1. Build the test binaries:

```
sbatch -N 1 build-nccl-tests.sh
```

2. Run a 2-node `all_reduce_perf` job:

```
sbatch -w <node 1>,<node 2> -o <job log file> nccl.sh
```

To run the job with more nodes, add the node names to the `-w` flag in the same comma-separated list format.

### 5.3. Config Modifications

Before launching jobs some changes to the config must be made.

#### 5.3.1. Set NCCL Topology

The NCCL topology file is unique for each CSP, and can be found in their corresponding
folders (`csp_tools/<csp>/topo.xml`)

In `launcher_scripts/conf/config.yaml`, mount the directory containing the topology file:

```
container_mounts:
  - /path/to/nemo_megatron_launcher/csp_tools/<csp>/:/nccl
```

Then set the path of the file in the container:

```
env_vars:
    NCCL_TOPO_FILE: /nccl/topo.xml
```

#### 5.3.2. Environment Variables

##### 5.3.2.1. Azure Variables

Set these environment variables in `config.yaml` (these are only needed for Azure):

```
env_vars:
  UCX_IB_PCI_RELAXED_ORDERING: auto
  NCCL_IB_PCI_RELAXED_ORDERING: 2
  NCCL_IB_TIMEOUT: 22
  NCCL_DEBUG: INFO
```

##### 5.3.2.2. AWS Variables

AWS recommends setting the following flag to avoid data corruption:

```
env_vars:
  NCCL_PROTO: simple
```

Setting this flag reduces training throughput by roughly 2%.

## 6. Quick Start Guide

### 6.1. Getting Started with NeMo Multimodal

#### 6.1.1. Prepare Environment

<!--
The whole solution uses a set of Docker containers executed at the Slurm
cluster using the pyxis plug-in Base Command Platform cluster. The training
container also includes conversion scripts and NVIDIA Triton Model Navigator.
The inference container is just the NVIDIA Triton Inference Server with the
FasterTransformer backend installed.    For Base Command Platform, the NeMo Multimodal
scripts repository (bcp branch) will be part of the container image. It is
recommended to create a nemo_megatron_ws_scripts_<username> workspace in your ace and
copy the nemo_megatron_launcher directory there    either from the container image or
from git clone of the above repository if you have access.    Install the NeMo Multimodal
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

##### 6.1.1.1. Slurm

The NeMo Multimodal codebase is included as part of the training container. To
copy it to a local directory in the cluster, it needs to be extracted from the
container. To copy the code to a directory named /path/to/local/dir the
following command can be executed. The NeMo Multimodal repository for
Slurm has been verified on both Slurm-based DeepOps clusters as well as Base
Command Manager.

```
srun -p [partition] -N 1 --container-mounts=/path/to/local/dir:/workspace/mount_dir --container-image=[container_tag] bash -c "cp -r /opt/NeMo-Megatron-Launcher/launcher_scripts /workspace/mount_dir/"
```

Install the NeMo Multimodal scripts dependencies on the head node of the cluster:

```
pip install -r requirements.txt
```

You can use virtualenv to prevent polluting your head node environment for
other Python projects. If your configuration lacks pip, then you can
install pip using use [get_pip.py](https://github.com/pypa/get-pip) with just `python3`.

##### 6.1.1.2. Base Command Platform

The NeMo Multimodal Launcher codebase is included as part of the training
container. Before starting, set up the ngc cli and configuration as described
in the Base Command Platform User Guide. In this guide, we will mainly
use two Base Command Platform workspaces, one for storing the training dataset,
and another for storing the results, checkpoints and logs. Therefore, start by
creating these workspaces (e.g. `nemo_megatron_data_ws` and `nemo_megatron_results_ws`). See
the Base Command Platform User Guide for how to create and work with Base
Command Platform workspaces.

#### 6.1.2. Configure and Customize Pipeline

This section provides instructions for configuring and customizing the pipeline in NeMo-Megatron-Launcher. It covers
four areas: cluster configurations, pipeline configurations, environment variables configurations, and NUMA mapping
configurations.

##### 6.1.2.1. Cluster Configurations

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

##### 6.1.2.2. Pipeline Configurations

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

##### 6.1.2.3. Environment Variables Configurations

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

##### 6.1.2.4. NUMA Mapping Configurations

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

#### 6.1.3. Launch Pipeline

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

#### 6.1.4. Example: Pre-train Stable Diffusion 860M Model for 10 Epochs with Resolution 256

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
2. **Select the `stable_diffusion` model with `860m_res_256_pretrain` configuration**: Update the `training` field in
   the `defaults` section of `conf/config.yaml`:
   ```yaml
   training: stable_diffusion/860m_res_256_pretrain
   ```
3. **Change the training epochs**: Navigate to the `conf/training/stable_diffusion/860m_res_256_pretrain.yaml` file and update
   the `max_epochs` field under the `trainer` section:
   ```yaml
    trainer:
      max_epochs: 10
   ```
4. **Pipeline Execution**: With these customizations in place, the pipeline will now execute only the `training` stage,
   using
   the `stable_diffusion` model with the `860m_res_256_pretrain` configuration, and train for a total of `10` epochs.
   To run the customized pipeline, simply execute:
   ```
   python3 main.py
   ```

Instead of manually editing the configuration files, you can also use **Hydra's override feature** to achieve the same
customizations in a single command. This allows you to quickly test different configurations without modifying the
original files. To run the customized pipeline according to the instructions provided earlier, use the following
command:

```
python3 main.py stages=[training] training=stable_diffusion/860m_res_256_pretrain training.trainer.max_epochs=10
```

**Note**: When using Hydra's override feature, make sure to include the stage name (training in this example) for
overriding a stage configuration found in conf/(stage_name)/(model_type)/(model_name).yaml. This ensures that the
correct stage and configuration file are targeted for the override.

### 6.2. Data Preparation

#### 6.2.1. ImageNet

_Note: It is the responsibility of each user to check the content
of the dataset, review the applicable licenses, and determine if it is suitable for their intended use.
Users should review any applicable links associated with the dataset before placing the data on their machine._

Please note that according to the ImageNet terms and conditions, automated scripts for downloading the dataset are not
provided. Instead, kindly follow the steps outlined below to download and extract the data.

##### 6.2.1.1. ImageNet 1k

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

##### 6.2.1.2. ImageNet 21k

1. Create an account on [ImageNet](http://image-net.org/download-images) and download "ImageNet21k" to
   `data/imagenet_21k`.
2. Extract the data:

  ```bash
  tar -xvf winter21_whole.tar.gz && rm -f winter21_whole.tar.gz
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  ```

#### 6.2.2. Multimodal Datasets

_Note: It is the responsibility of each user to check the content
of the dataset, review the applicable licenses, and determine if it is suitable for their intended use.
Users should review any applicable links associated with the dataset before placing the data on their machine._

##### 6.2.2.1. Overview

For all multimodal models (except Instruct-Pix2Pix and ControlNet; see Section 6.2.3 and Section 6.2.5), we provide a generic pipeline as detailed below
to download and prepare the dataset. The pipeline is suitable for any multimodal datasets hosted on the
[Hugging Face data repository](https://huggingface.co/datasets?task_categories=task_categories:text-to-image)
where the data is stored as one or more parquet files. The pipeline processes the dataset into the
[WebDataset](https://github.com/webdataset/webdataset) format, consisting of tar files of equal sizes for
efficient training.

The 6 sub-stages are as follows.

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
6. `merge_source_tar`: (Optional) After precaching, this sub-stage can copy and append any additional objects
(such as original image or metadata files) from the source tar files to the result tar files.

Depending on your specific circumstance, not all sub-stages need to be run all at once.
For example, for parquet datasets not hosted on HuggingFace or whose format is not parquet,
sub-stages 2-6 can be used to process locally downloaded datasets.
For webdatasets already downloaded locally, sub-stages 4-6 can be used to precache the encoding to reduce training time.
For models that encode image and text on-the-fly, only sub-stages 1-3 need to be run.

Instruction for configuring each sub-stage is provided as a comment next to each field in
`conf/data_preparation/multimodal/download_multimodal.yaml`

##### 6.2.2.2. Running the Pipeline

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

##### 6.2.2.3. Configuration for Precaching

###### 6.2.2.3.1. General Format

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

###### 6.2.2.3.2. Precaching Config

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

###### 6.2.2.3.3. Resume Precaching (Advanced)

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



#### 6.2.3. InstructPix2Pix

_Note: It is the responsibility of each user to check the content
of the dataset, review the applicable licenses, and determine if it is suitable for their intended use.
Users should review any applicable links associated with the dataset before placing the data on their machine._

To download and prepare the custom dataset used for training Instruct-Pix2Pix, please follow the instruction from
the official [Instruct-Pix2Pix](https://github.com/timothybrooks/instruct-pix2pix#generated-dataset)
repository.

Move the downloaded data to `${data_dir}/instruct_pix2pix/clip-filtered-dataset`

#### 6.2.4. MSCOCO for FID Evaluation

_Note: It is the responsibility of each user to check the content
of the dataset, review the applicable licenses, and determine if it is suitable for their intended use.
Users should review any applicable links associated with the dataset before placing the data on their machine._

For more details on the evaluation workflow, please see Section 5.6.3.

##### 6.2.4.1. Download and Setup

1. Review the terms of use from the official [COCO](https://cocodataset.org/#download) website.
2. Download the 2014 validation images, and extract the images to `${data_dir}/fid_evaluation/coco2014/val2014`
3. Download the 2014 train/val annotations, and extract `captions_val2014.json` to
   `${data_dir}/fid_evaluation/coco2014/captions_val2014.json`
4. Review the terms of use of [COCO API](https://github.com/cocodataset/cocoapi), then install the Python API following
   the instructions.
5. Install the dependencies for the preprocessing script: `pip install matplotlib cython Pillow`

##### 6.2.4.2. Preprocess Images and Captions

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

#### 6.2.5. ControlNet

_Note: It is the responsibility of each user to check the content
of the dataset, review the applicable licenses, and determine if it is suitable for their intended use.
Users should review any applicable links associated with the dataset before placing the data on their machine._

Controlnet needs an extra conditioning input given in image format, following [Section 6.2.2](#622-multimodal-datasets),
the dataset should be organized into tarfiles in the following way:

```text
contolnet0001.tar
|---- 00000.png (conditioning image)
|---- 00000.jpg (target image)
|---- 00000.txt (text prompt)
|---- 00001.png (conditioning image)
|---- 00001.jpg (target image)
|---- 00001.txt (text prompt)
...
```

To utilize segmentation maps as conditioning input, the conditioning image can be obtained through a detector model, while text prompts can be derived from blip captioning. For further guidance on preparing your own dataset, you may find the documentation of [ControlNet](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md) helpful.


### 6.3. Model Training

We provide predefined training configurations for all released model types, which can be found in the `conf/training/`
directory. These configurations include carefully selected hyper parameters that serve as a guideline for creating
custom model configurations. To choose the desired configuration, simply update the training parameter in
the `conf/config.yaml` file. For additional guidance on customizing configurations, please refer
to [Section 6.1](#61-getting-started-with-nemo-multimodal) in the
documentation.

For the Base Command Platform (BCP), it is important to note that all jobs must be launched in multi-node mode. This
requirement ensures proper setup of BCP pytorch environment.

#### 6.3.1. Vision Transformer Training

We have curated 5 configurations with suggested hyperparameters specifically for the NVIDIA DGX SuperPOD, which is
equipped with 8 NVIDIA A100 80GB GPUs. The configurations for the curated models can be found in the `conf/training/vit`
directory. You can access and modify the parameters to adjust the hyperparameters for your specific training runs. By
customizing these settings, you can tailor the model's performance and training efficiency to better suit your needs and
requirements.

| Model | Model size (M) | Hidden size | FFN_dim | Attention heads | Number of layers | Batch Size per GPU | Accumulated Global Batch Size | Precision | AMP Level | Total Training Samples Seen |
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

#### 6.3.2. CLIP Training

We have curated 3 configurations with suggested hyperparameters specifically for the NVIDIA DGX SuperPOD, which is
equipped with 8 NVIDIA A100 80GB GPUs. The configurations for the curated models can be found in
the `conf/training/clip` directory. You can access and modify the parameters to adjust the hyperparameters for your
specific training runs. By customizing these settings, you can tailor the model's performance and training efficiency to
better suit your needs and requirements.

| Model    | Image size | Text Model size (M) | Image Model size (M) | Output dim | Batch Size per GPU | Accumulated Global Batch Size | Precision | AMP Level | Total Training Samples Seen |
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

#### 6.3.3. Stable Diffusion Training

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

\**Our multimodal dataset is originated from Common Crawl with custom filtering and contains 670M image-caption pairs.*

| Stage       | Resolution | Unet model size (M) | Text conditioning model       | Batch Size per GPU | Accumulated Global Batch Size | Precision | AMP Level | Effective Dataset size| Dataset Filtering       | Total Training Samples Seen  |
|-------------|------------|---------------------|-------------------------------|--------------------|-------------------------------|-----------|-----------|-----------------------|-------------------------|------------------------|
| Pretraining | 256        | 859                 | openai/clip-vit-large-patch14 | 128                | 8192                          | FP16      | O1        | 676M                  | None                    | 680M                   |
| SD v1.1     | 512        | 859                 | openai/clip-vit-large-patch14 | 32                 | 8192                          | FP16      | O1        | 39.5M                 | Resolution >= 1024x1024 | 409M                   |
| SD v1.2     | 512        | 859                 | openai/clip-vit-large-patch14 | 32                 | 8192                          | FP16      | O1        | 218M                  | Resolution >= 512x512   | 1.23B                  |
| SD v1.5     | 512        | 859                 | openai/clip-vit-large-patch14 | 32                 | 8192                          | FP16      | O1        | 218M                  | Resolution >= 512x512   | 1.32B                  |

For SD v2.0 base, the text conditioning model is replaced with OpenCLIP-ViT/H. Training stages are similar to the original configuration, which contain pretraining with 256x256 resolution and follow by finetuing with 512x512 resolution. We can use the datasets recommended in the Huggingface model cards to reproduce the result of SD v2.0 base.

| Stage       | Resolution | Unet model size (M) | Text conditioning model       | Batch Size per GPU | Accumulated Global Batch Size | Precision | AMP Level | Effective Dataset size| Dataset Filtering       | Total Training Samples Seen  |
|-------------|------------|---------------------|-------------------------------|--------------------|-------------------------------|-----------|-----------|-----------------------|-------------------------|------------------------|
| SD v2.0 Pretraining | 256        | 865                 | OpenCLIP-ViT/H | 128                | 8192                          | FP16      | O1        | 676M                  | None                    | 680M                   |
| SD v2.0 Base        | 512        | 865                 | OpenCLIP-ViT/H | 32                 | 8192                          | FP16      | O1        | 218M                  | Resolution >= 512x512   | 1.32B                  |

To enable the training stage with Stable Diffusion, make sure:

1. In the `defaults` section, update the `training` field to point to the desired Stable Diffusion configuration file.
   For example,
   if you want to start the pretraining from scratch, change the training field to `stable_diffusion/860m_res_256.yaml`.
   ```yaml
    defaults:
      - _self_
      - cluster: bcm
      - data_preparation: multimodal/download_multimodal
      - training: stable_diffusion/860m_res_256_pretrain.yaml
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

1.To continue training the Stable Diffusion model from the pretraining results, we reset the training process
by only loading the UNet weights. You can do this by using the last checkpoint from the previous training and passing it
to `training.model.unet_config.from_pretrained`. Due to different naming in model parameters, indicating you are loading
from checkpoint trained by NeMo , set `training.model.unet_config.from_NeMo=True`. If you are resuming training from a
Huggingface checkpoint, you can also load the Unet weights from that source. In this case, you need to
set `training.model.unet_config.from_NeMo=False`.

2.To improve the quality of generated images, it is recommended to utilize pretrained checkpoints for AutoencoderKL and CLIP. We have compiled a list of recommended sources for these checkpoints, but please note that the AutoencoderKL checkpoint cannot be downloaded via the provided script. Instead, you must download it locally and ensure that the correct path is specified in the configuration file before proceeding.

Please be advised the scripts that NVIDIA provides are optional to use and will download models that
are based on public data which may contain copyrighted material. Consult your legal department before using these scripts.

The following are the pretrained checkpoints for SD v1
| model | link | download by script |
|---------------|--------------------------------------------------------------------------------|-----------------|
| AutoencoderKL |  [link](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main/vae)    |      No           |
| CLIP          |  [link](https://huggingface.co/openai/clip-vit-large-patch14)                  |     Yes            |


The following are the pretrained checkpoints for SD v2.0
| model | link | download by script |
|---------------|--------------------------------------------------------------------------------|-----------------|
| AutoencoderKL |  [link](https://huggingface.co/stabilityai/stable-diffusion-2-base/tree/main/vae)    |      No           |
| OpenCLIP      |  [link](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)                  |     Yes            |


In the latest update, we have introduced support for using the Clip encoder provided by NeMo. To learn how to convert weights to NeMo Clip checkpoints, please refer to [Section 6.9](#69-convert-checkpoints-from-external-sources-to-nemo) in the documentation.
If you prefer to restore the previous behavior and use the HF Clip encoder, you can find instructions in the comments within the stable diffusion configuration files.

Note: If you use NeMo Clip checkpoints as the Clip encoder, the Clip checkpoint need to be kept in the same path as specified in the configuration file when you load a stable diffusion checkpoint.

3.There is no guarantee that training Stable Diffusion for an extended period will necessarily result in improved FID/CLIP scores. To achieve best results, we suggest evaluating various checkpoints during the late stages of convergence.

#### 6.3.4. InstructPix2Pix Training

InstructPix2Pix essentially performs tuning on top of an existing Stable Diffusion checkpoint. The recommended
configuration can be found in the `conf/training/instruct_pix2pix` directory. You can access and modify the parameters
to customize the hyperparameters according to your specific training requirements.

To enable the training stage with an InstructPix2Pix model, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `training` field to point to the desired InstructPix2Pix
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

**Remarks**:
1.You can feed the trained Stable Diffusion checkpoint into InstructPix2Pix training by
specifying `training.model.ckpt_path` (or set `ckpt_path` field in the `model` section of `860m_sd_edit.yaml`). The
checkpoint can be sourced from either NeMo or Hugging Face in the form of a `.ckpt` file.

2.In order to train InstructPix2Pix, a pretrained Stable Diffusion model is required. However, it is important to note that only the UNet component needs to be fine-tuned, while AutoencoderKL and CLIP remain unaltered. We recommend training the base Stable Diffusion model with AutoencoderKL and CLIP, using the pretrained checkpoints for initialization. For further details on this process, please refer to Section 6.3.3. Please be advised the download scripts that NVIDIA provides are optional to use and will download models that are based on public data which may contain copyrighted material. Consult your legal department before using these scripts.

#### 6.3.5. DreamBooth Training

DreamBooth is also fine-tuning on top of an existing Stable Diffusion checkpoint. The recommended configuration can be
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
above in section [Section 6.3.3](#633-stable-diffusion-training).

3.In order to train DreamBooth, a pretrained Stable Diffusion model is required. However, it is important to note that only the UNet component needs to be fine-tuned, while AutoencoderKL and CLIP remain unaltered. We recommend training the base Stable Diffusion model with AutoencoderKL and CLIP, using the pretrained checkpoints for initialization.  See Section 6.3.3 for details.
Please be advised the scripts that NVIDIA provides are optional to use and will download models that are based on public data which may contain copyrighted material. Consult your legal department before using these scripts.

4.By default, DreamBooth training results are not stored in NEMO checkpoint format. However, a customized conversion stage is available to convert DreamBooth checkpoint files to '.nemo', enabling compatibility with Stable Diffusion inference pipelines.

#### 6.3.6. ControlNet Training

ControlNet essentially performs tuning on top of an existing Stable Diffusion checkpoint. The recommended
configuration can be found in the `conf/training/controlnet` directory. You can access and modify the parameters
to customize the hyperparameters according to your specific training requirements.

To enable the training stage with an ControlNet model, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `training` field to point to the desired InstructPix2Pix
   configuration file. For example,
   if you want to use the `controlnet_v1-5.yaml`, change the `training` field to `controlnet/controlnet_v1-5.yaml`.
   ```yaml
    defaults:
      - _self_
      - cluster: bcm
      - data_preparation: null
      - training: controlnet/controlnet_v1-5.yaml
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
1. ControlNet copies encoder and middle blocks from Stable Diffusion and finetune a copy of these blocks, thus providing
a pretrained checkpoint of Stable Diffusion needs to be passed into the config file, for both `control_stage_config.from_pretrained` and
`unet_config.from_pretrained`.


#### 6.3.7. Imagen Training

We have curated configurations with suggested hyperparameters specifically for the NVIDIA DGX SuperPOD, which is
equipped with 8 NVIDIA A100 80GB GPUs. The configurations for the curated models can be found in
the `conf/training/imagen` directory. You can access and modify the parameters to adjust the hyperparameters
for your
specific training runs. By customizing these settings, you can tailor the model's performance and training efficiency to
better suit your needs and requirements.

The training process for Imagen typically involves multiple stages of models at different resolutions (64x64, 256x256, 1024x1024). Datasets
are deliberately alternated to achieve superior image quality. We provide ??? training configurations here:

Base model:
 - base64-2b: Training 2B params 64x64 model as described in Imagen paper Appendix F.1
 - base64-500m: Training 500m params 64x64 model with reduced channel size
SR256 model:
 - sr256-600m: Training 600m params 256x256 EfficientUNet model as described in Imagen paper Appendix F.2
 - sr256-400m: Training 400m params 256x256 UNet model with similar configurattion as DeepFloyd IF-II-M
SR1024 model:
 - sr1024-600m: Training 600m params 1024x1024 EfficientUNet model as described in Imagen paper Appendix F.3


\**Our multimodal dataset is originated from Common Crawl with custom filtering and contains 670M image-caption pairs.*
| Model         | Resolution | Unet model size (M) | Text conditioning model | Batch size per GPU | Accumulated Global Batch size | Precision | AMP Level | Dataset Filtering  | Total Training Samples |
|---------------|------------|---------------------|-------------------------|--------------------|-------------------------------|-----------|-----------|--------------------|------------------------|
| 500m_res_64   | 64         | 524                 | "t5-11b"                | 64                 | 4096                          | BF16      | O1        | None               | 680M                   |
| 2b_res_64     | 64         | 2100                | "t5-11b"                | 16                 | 2048                          | BF16      | O1        | None               | 680M                   |
| 600m_res_256  | 256        | 646                 | "t5-11b"                | 64                 | 4096                          | BF16      | O1        | Resolution >= 256  | 544M                   |
| 400m_res_256  | 256        | 429                 | "t5-11b"                | 16                 | 2048                          | BF16      | O1        | Resolution >= 256  | 544M                   |
| 600m_res_1024 | 1024       | 427                 | "t5-11b"                | 64                 | 4096                          | BF16      | O1        | Resolution >= 1024 | 409M                   |
To enable the training stage with Imagen, make sure:

1. In the `defaults` section, update the `training` field to point to the desired Imagen configuration file.
   For example,
   if you want to start training base64 500m model from scratch, change the training field to `imagen/500m_res_64.yaml`.
   ```yaml
    defaults:
      - _self_
      - cluster: bcm
      - data_preparation: multimodal/download_multimodal
      - training: imagen/500m_res_64.yaml
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

1.There is no dependency in training different resolution models. Ideally it is possible to train all 64x64, 256x256, 1024x1024 at the same time independently given the sufficient computing resources.

2.We recommend to pre-process the training dataset with precached embeddings. Imagen typically use T5 embedding, and T5 encoder are giant in size and can significantly reduce training throughput if loading it while training. We noticed significant batch size drop and throughput drop if using online-encoding option

3.Despite the fact that Imagen paper they state that EfficientUNet has better throughput and does not harm visual quality, Emperically we found that training the regular UNet for SR model still yeilds more visually-appealing images.

4.We provide two scheduling/sampling for Imagen Training: Continous DDPM and EDM. Continous DDPM is the default schema used in the original paper. EDM ([Elucidating the Design Space of Diffusion-Based Generaive Models](https://arxiv.org/abs/2206.00364)). Emperically, we found that EDM yeilds lower FID score.

5.While in paper they use T5-xxl (4096 dimension) encoder, We use T5-11b (1024 dimension) encoder during training due to space considerations.

Please be advised the scripts that NVIDIA provides are optional to use and will download models that
are based on public data which may contain copyrighted material. Consult your legal department before using these scripts.

| model | link | download by script |
|---------------|--------------------------------------------------------------------------------|-----------------|
| T5-11b |  [link](https://huggingface.co/t5-11b)    |      Yes           |
| T5-xxl          |  [link](https://huggingface.co/google/t5-v1_1-xxl)                  |     Yes            |


5.There is no guarantee that training Imagen for an extended period will necessarily result in improved FID/CLIP scores. To achieve best results, we suggest evaluating various checkpoints during the late stages of convergence.


### 6.4. Checkpoint Conversion

We provide a convenient tool for converting checkpoints from the `.ckpt` format to the `.nemo` format. The `.nemo`
format checkpoints can be used later in evaluation and inference stages. Users don't need to run the checkpoint
conversion explicitly, as a `.nemo` checkpoint will be automatically generated and saved in the checkpoints folder at
the end of training or fine-tuning. However, if you want to perform inference with an intermediate checkpoint, you will
need to use the conversion script to convert the checkpoint from the `.ckpt` format to the `.nemo` format.

The usage of the conversion script is consistent across different model types. All conversion configuration files can be
found in the `conf/conversion` folder. For additional guidance on customizing configurations, please refer
to [Section 6.1](#61-getting-started-with-nemo-multimodal) in the
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

### 6.5. Model Fine-tuning

We provide predefined fine-tuning configurations for Vision Transformer models, which can be found in
the `conf/fine_tuning/`
directory. These configurations include carefully selected hyper parameters that serve as a guideline for creating
custom model configurations. For additional guidance on customizing configurations, please refer
to [Section 6.1](#61-getting-started-with-nemo-multimodal) in the
documentation.

#### 6.5.1. Vision Transformer Fine-tuning

We provide a predefined fine-tuning configuration for the `ViT B/16` model on ImageNet-1K, which can be found in
the `conf/fine_tuning/imagenet1k.yaml` file. The following table highlights the key differences between ViT pretraining
and fine-tuning:

| Aspect                | ViT Pretraining           | ViT Fine-tuning              |
|-----------------------|---------------------------|------------------------------|
| Configuration Folder  | `conf/training/vit`       | `conf/fine_tuning/vit`       |
| Training Samples Seen | 400M                      | 10M                          |
| Optimizer             | Fused AdamW               | SGD                          |
| Resolution            | 224x224                   | 384x384                      |
| Classification Head   | MLP with one hidden layer | MLP with single linear layer |

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

### 6.6. Model Evaluation

In NeMo Multimodal, we also provide simple scripts for users to benchmark their trained models, including ViT,
CLIP and Stable Diffusion. The configuration files for these evaluations can be found in the `conf/evaluation`
directory. These scripts allow you to assess the performance of your trained models on various metrics. For additional
guidance on customizing configurations, please refer to [Section 6.1](#61-getting-started-with-nemo-multimodal)
in the
documentation.

#### 6.6.1. Vision Transformer Evaluation

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

#### 6.6.2. CLIP Evaluation

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

#### 6.6.3. Stable Diffusion Evaluation

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
4. To evaluate the quality of generated images, a pretrained Inception network and CLIP model are necessary.  We list below
   the recommended ckpt sources. Please note that the scripts that NVIDIA provides are optional to use and will download models that are based on public data which may contain copyrighted material. Consult your legal department before using these scripts.

| model | link | download by script |
|---------------|--------------------------------------------------------------------------------|-----------------|
| TFinceptionV3 |  [link](https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth)    |      Yes           |
| CLIP          |  [link](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/pretrained.py#L121)                  |     Yes            |

#### 6.6.4. Imagen Evaluation

Our evaluation script performs image generation for the captions provided in the validation subset of the MS COCO
dataset, computes the FID score between real and generated images, computes the CLIP score betweel generated images and
teh corresponding captions, and plots the FID-CLIP graph. This is a multi-stage evaluation, and our scripts will
automatically generate SLURM jobs with dependencies.

To configure the configuration files and enable the evaluation stage for Stable Diffusion, follow the steps outlined
below:

1. In the `defaults` section of `conf/config.yaml`, update the `evaluation` field to point to the desired Imagen configuration file. For example,
   if you want to use the `imagen/fid_clip` configuration, change the `evaluation` field
   to `imagen/fid_clip`.
   ```yaml
    defaults:
      - evaluation: imagen/fid_clip
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `evaluation` stage is included. For example,
   ```yaml
    stages:
      - evaluation
      ...
   ```
3. Configure `conf/evaluation/imagen/fid_clip.yaml` to specify `node_array_size` and `ntasks_per_node`, as
   well as which sub-stages to run.
   ```yaml
    generate_images: True
    compute_fid_scores: True
    compute_clip_scores: True
    plot_fid_clip: True
   ```

**Remarks**:

1. To load a pretrained checkpoint for inference, set the `restore_from_path` field in the `models` section to the path
   of the pretrained checkpoint in `.nemo` format in `conf/evaluation/imagen/fid_clip.yaml`.
2. We highly recommend users to use the same precision (i.e. `trainer.precision`) for evaluation as was used during
   training.
3. The `generate_images` sub-stage involves a multi-node run, whereas the other stages utilize only a single GPU.
4. It is possible to save intermediate stage images (lower resolution images) along with the final image by setting `fid.save_all_res=True`. Only the final images are used for evaluation.
5. To save the text along with images for manual checking, set `fid.save_text=True`.
6. To evaluate the quality of generated images, a pretrained Inception network and CLIP model are necessary.  We list below
   the recommended ckpt sources. Please note that the scripts that NVIDIA provides are optional to use and will download models that are based on public data which may contain copyrighted material. Consult your legal department before using these scripts.

| model | link | download by script |
|---------------|--------------------------------------------------------------------------------|-----------------|
| TFinceptionV3 |  [link](https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth)    |      Yes           |
| CLIP          |  [link](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/pretrained.py#L121)                  |     Yes            |

### 6.7. Model Inference (in NeMo Framework)

In NeMo Multimodal, we provide scripts to perform inference directly via NeMo framework, rather than using
NVIDIA Triton Inference Server. This allows you to infer with your pretrained models directly without the need for a
separate deployment or inference server. It is particularly useful when you want to experiment with different model
configurations, perform quick evaluations, or prototype a solution before deploying it at scale with Triton Inference
Server or another deployment option.

Our framework inference configurations are provided in the folder `conf/fw_inference`. For additional guidance on
customizing configurations, please refer to [Section 6.1](#61-getting-started-with-nemo-multimodal) in the
documentation.

#### 6.7.1. Vision Transformer Inference (in NeMo Framework)

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

#### 6.7.2. CLIP Inference (in NeMo Framework)

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

#### 6.7.3. Stable Diffusion Inference (in NeMo Framework)

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

**Remarks**:

We have supported three types of inference samplers, 'DDIM', 'PLMS' and 'DPM', which can be changed by from config files. 'DPM'
sampler is added in recent updates and able to achieve similar image quality with half of steps needed for inference.

#### 6.7.4. InstructPix2Pix Inference (in NeMo Framework)

For InstructPix2Pix models, our inference script processes an original image based on a provided edit prompt, modifies
the image accordingly, and saves the edited image as a new file.

To enable the inference stage with a InstructPix2Pix model, configure the configuration files:

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

#### 6.7.5. DreamBooth Inference (in NeMo Framework)

For DreamBooth, the inference script generates images from text prompts defined in the config file, similar to section
5.7.3. Note that, dreambooth is a fine-tuning model based on diffusion models to link a special token with certain
subject, so make sure the special token you trained on is included in the text prompt. For
example, `a photo of sks dog sleeping`.

To enable the inference stage with DreamBooth, configure the configuration files:

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

**Remarks**:

Please refer to [6.3.5. DreamBooth Training](#635-dreambooth-training), the inference stage of DreamBooth should be conducted
subsequent to the DreamBooth conversion process. This conversion transforms the DreamBooth ckpt into a '.nemo' format and meanwhile
remapping the parameter keys into Stable Diffusion style, allowing for a consistent inference pipeline.

#### 6.7.6. ControlNet Inference (in NeMo Framework)

For ControlNet, the inference script generates images from text prompts defined in the config file, similar to section
5.7.3. Note that, an image conditioning is required besides text prompt.

To enable the inference stage with ControlNet, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `fw_inference` field to point to the desired DreamBooth
   inference configuration file. For example,
   if you want to use the `controlnet/controlnet_infer.yaml` configuration, change the `fw_inference` field
   to `controlnet/controlnet_infer`.
   ```yaml
    defaults:
      - fw_inference: controlnet/controlnet_infer
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `fw_inference` stage is included. For example,
   ```yaml
    stages:
      - fw_inference
      ...
   ```
3. Configure `prompts` and `num_images_per_prompt` fields of `conf/fw_inference/controlnet/controlnet_infer.yaml`.
   Set `model.restore_from_path` to the ckpt generated from dreambooth training.

**Remarks**:
We have implemented an example processing function to extract segmentation map from a target image, and use that as the
conditioning to generate images. To utilize that function, please specify `infer.control_image_preprocess=seg2img`. In other use cases, the input image specified at `infer.control` should be the conditioning image
instead of target image, and set `infer.control_image_preprocess=null`.


#### 6.7.7. Imagen Inference (in NeMo Framework)

For text-to-image models, the inference script generates images from text prompts defined in the config file.

To enable the inference stage with Imagen, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `fw_inference` field to point to the desired Stable
   Diffusion inference configuration file. For example,
   if you want to use the `imagen/text2img.yaml` configuration, change the `fw_inference` field to `imagen/text2img`.
   ```yaml
    defaults:
      - fw_inference: imagen/text2img
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `fw_inference` stage is included. For example,
   ```yaml
    stages:
      - fw_inference
      ...
   ```
3. Configure `infer.texts` and `infer.num_images_per_prompt` fields of `conf/fw_inference/imagen/text2img.yaml`.
   Set `model.customized_model.base_ckpt&sr256_ckpt&sr1024_ckpt` to the `.nemo` ckpt you want generate images with. Set `infer.target_resolution` to the desired resolution.

**Remarks**:

We provide both DDPM and EDM sampler. We recommend for EDM training, at least 30 steps of inference is required; for DDPM training, at least 250 steps of inference is required.

### 6.8. Model Export

In NeMo Multimodal, we provide scripts to perform export directly via NeMo framework to ONNX and NVIDIA
TensorRT. This allows us to run accelerated inference on the NVIDIA Triton Inference Server detailed in the next
section, section 6.
For the CLIP and ViT models, setting `infer.max_batch_size`, will create ONNX and NVIDIA TensorRT models that accept
batch_sizes
from `1` to `infer.max_batch_size`. For the Stable Diffusion, InstructPix2Pix, and DreamBooth pipelines,
the `infer.num_images_per_prompt` (`edit.num_images_per_prompt` in InstructPix2Pix) will
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

#### 6.8.1. Vision Transformer Export

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

#### 6.8.2. CLIP Export

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

#### 6.8.3. Stable Diffusion Export

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

#### 6.8.4. InstructPix2Pix Export

For InstructPix2Pix models, the export script generates four different optimized inference models.
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

#### 6.8.5. DreamBooth Export

For DreamBooth, the export script generates three different optimized inference models.
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

#### 6.8.6. ControlNet Export

For ControlNet, the export script generates four different optimized inference models.
The first model is the VAE Decoder, the second model is the UNet, the third model is the CLIP Encoder and the fourth is the control model.

1. In the `defaults` section of `conf/config.yaml`, update the `export` field to point to the desired ControlNet
   inference configuration file. For example,
   if you want to use the `controlnet/export_controlnet.yaml` configuration, change the `export` field
   to `controlnet/export_controlnet`.
   ```yaml
    defaults:
      - export: controlnet/export_controlnet
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `export` stage is included. For example,
   ```yaml
    stages:
      - export
      ...
   ```
3. Configure `infer.num_images_per_prompt` of the `conf/export/controlnet/export_controlnet.yaml` file to set the
   batch_size to use for the ONNX and
   NVIDIA TensorRT models.

**Remarks**:

1. To load a pretrained checkpoint for inference, set the `restore_from_path` field in the `model` section to the path
   of the pretrained checkpoint in `.nemo` format in `conf/export/controlnet/export_controlnet.yaml`.

2. Only `num_images_per_prompt: 1` is supported for now.

#### 6.8.7. Imagen Export

For text-to-image models, the export script generates two different optimized inference models.
The first model is the UNet, and the second model is the T5 encoder. The script generates separate UNet model for different resolutions (e.g. 64x64, 256x256, 1024x1024)

1. In the `defaults` section of `conf/config.yaml`, update the `export` field to point to the desired Stable Diffusion
   inference configuration file. For example,
   if you want to use the `imagen/export_imagen.yaml` configuration, change the `export` field
   to `imagen/export_imagen`.
   ```yaml
    defaults:
      - export: imagen/export_imagen
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `export` stage is included. For example,
   ```yaml
    stages:
      - export
      ...
   ```
3. Configure `infer.num_images_per_prompt` of the `conf/export/imagen/export_imagen.yaml` file to
   set the batch_size to use for the ONNX and
   NVIDIA TensorRT models.

**Remarks**:

1. To load a pretrained checkpoint for inference, set the `base_ckpt`, `sr256_ckpt`, `sr1024_ckpt` field in the `model.customized_model` section to the path
   of the pretrained checkpoint in `.nemo` format in `conf/export/imagen/export_imagen.yaml`. Make sure `model.target_resolution` is set to desired resolution.


### 6.9. Convert Checkpoints from External Sources to Nemo

We provide a convenient tool for converting checkpoints from external sources to the `.nemo` format. The `.nemo`
format checkpoints can be used in NeMo multimodal training, e.g., in Stable Diffusion. Currently, we support
the conversion of the CLIP model from Huggingface or OpenCLIP into the `.nemo` format.

All external conversion configuration files can be
found in the `conf/external_conversion` folder. For additional guidance on customizing configurations, please refer
to [Section 6.1](#61-getting-started-with-nemo-multimodal) in the
documentation.

To enable the `external_conversion` stage and configure external conversion settings, configure the configuration files:

1. In the `defaults` section of `conf/config.yaml`, update the `external_conversion` field to point to the desired model type's
   configuration file. For example, if you want to convert a CLIP model, change the `external_conversion` field
   to `clip/convert_external_clip`.
   ```yaml
    defaults:
      - external_conversion: clip/convert_external_clip
      ...
   ```
2. In the `stages` field of `conf/config.yaml`, make sure the `external_conversion` stage is included. For example,
   ```yaml
    stages:
      ...
      - external_conversion
      ...
   ```
3. In the target external conversion YAML file, modify required fields like `version` and `arch`. Meanwhile, you also
   need to prepare a NeMo configuration YAML file for model initialization. You can copy a YAML file from
   `conf/training/clip` and modify the architecture parameters correspondingly based on the source external checkpoint.
   For example, to convert a CLIP H/14 model from OpenCLIP, modify or override the following fields
   inside `conf/external_conversion/clip/convert_external_clip.yaml`.
   ```yaml
   model:
     arch: ViT-H-14
     version: laion2b_s32b_b79k
     hparams_file: /path/to/modified_hparam.yaml
   ```

**Remark**:

1. If you are converting from Huggingface, simply set `version` to `huggingface` and `arch` to Huggingface
   model name.


## 7. Deploying the NeMo Multimodal Model

### 7.1. Setup

Prior to deploying a model or pipeline, the model or pipeline must be exported following the steps in [Section 6.8](#68-model-export)
No other additional setup is required as the NeMo container comes with the relevant NVIDIA Triton Inference Server
libraries
preinstalled and ready to go.

### 7.2. Start NVIDIA Triton Inference Server

Starting the NVIDIA Triton Inference Server is a simple command. First, however, please read the model specific section
below
to make sure everything is in the correct place.
To start the NVIDIA Triton Inference Server

```
/opt/tritonserver/bin/tritonserver --log-verbose 2 --model-repository /opt/NeMo-Megatron-Launcher/deployment/server --model-control-mode=explicit --load-model <model>
```

`<model>` can be substituted for the `stable_diffusion`, `instruct_pix2pix`, `clip_trt`, `clip_vision_trt`, `vit_trt`, `controlnet`.

#### 7.2.1. Stable Diffusion, DreamBooth

For Stable Diffusion and DreamBooth, copy the generated `plan` directory to the `deployment/server/stable_diffusion/1/`
directory.

#### 7.2.2. InstructPix2Pix

For InstructPix2Pix, copy the generated `plan` directory to the `deployment/server/instruct_pix2pix/1/` directory.

#### 7.2.3. Vision Transformer

Move the generated `.plan` file to `deployment/server/vit_trt/1/model.plan`.

#### 7.2.4. CLIP

Move the generated `.plan` file to `deployment/server/clip_vision_trt/1/model.plan`. For this model, two separate Triton
models need to be loaded
`--load-model clip_vision_trt --load-model clip_trt`. Querying `clip_trt` will provide tokenization and automatically
call `clip_vision_trt` using BLS.

#### 7.2.5. ControlNet

Copy the generated `plan` directory to the `deployment/server/controlnet/1/` directory.

#### 7.2.6 Imagen

For Imagen, copy the generated `plan` directory to the `deployment/server/imagen/1/`
directory.

### 7.3. Query NVIDIA Triton Inference Server

In a separate instance of the NeMo container, we can setup a client to query the server. In `deployment/client`, there
are a few examples of the clients.

#### 7.3.1. Stable Diffusion and DreamBooth

At query time, the values, `seed`, `unconditional_guidance_scale`, `inference_steps`, `eta` can be used as optional
inputs. If these are not set, the defaults are the values set during export.
The return is a single numpy array containing `num_images_per_prompt` images.

#### 7.3.2. InstructPix2Pix

At query time, the values, `seed`, `text_cfg_scale`, `steps`, `image_cfg_scale` can be used as optional inputs. If these
are not set, the defaults are the values set during export.
The return is a single numpy array containing `num_images_per_prompt` images. In the client example, make sure to set
the path to the input image.

#### 7.3.3. ControlNet

At query time, the values, `seed`, `unconditional_guidance_scale`, `inference_steps`, `eta`, `guess_mode`, `strength` can be used as optional inputs. If these are not set, the defaults are the values set during export.
The return is a single numpy array containing `num_images_per_prompt` images.  In the client example, make sure to set
the path to the control image.

#### 7.3.4 Imagen

At query time, the values, `seed`, `cfg` can be used as optional
inputs. If these are not set, the defaults are the values set during export.
The return is a single numpy array containing `num_images_per_prompt` images.

## 8. Performance

### 8.1. Vision Transformer Results

#### 8.1.1. Training Accuracy Results

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

#### 8.1.2. Training Performance Results

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
|          | Samples per Second               |   720 |   1386 |   2830 |   5599 |  10934 |
| ViT g/14 | Perfect Linear Scaling (Samples) |   720 |   1440 |   2880 |   5761 |  11522 |
|          | Speedup                          |    1x |  1.92x |  3.93x |  7.77x | 15.18x |

<img src="img/ViT g_14 NeMo Megatron Throughput (A100) (2305).svg"/>

- NVIDIA DGX SuperPODs (16 x 8 x H100 80GB for ViT g/14 model)

|          |                                  |      |       |       | Nodes |        |
|----------|----------------------------------|------|-------|-------|-------|--------|
|          |                                  | 1    | 2     | 4     | 8     | 16     |
|          | Samples per Second               |   1449 |   2850 |   5795 |  11247 |  22825 |
| ViT g/14 | Perfect Linear Scaling (Samples) |   1449 |   2898 |   5795 |  11590 |  23181 |
|          | Speedup                          |     1x |  1.97x |     4x |  7.76x | 15.75x |

<img src="img/ViT g_14 NeMo Megatron Throughput (H100) (2305).svg"/>

- DGX A100 vs. DGX H100: A Comparative Analysis of Vision Transformer Training

| Model       | Nodes | Global Batch Size | Micro Batch Size | Precision | Global Batch / Sec (A100) | Global Batch / Sec (H100) | Speedup (x) |
|-------------|-------|-------------------|------------------|-----------|---------------------------|---------------------------|-------------|
| ViT B/16    | 2     | 4096              | 256              | bf16 (O2) |                      3.06 |                      5.11 |         1.7 |
| ViT L/16    | 2     | 4096              | 256              | bf16 (O2) |                      1.33 |                      2.76 |         2.1 |
| ViT H/14    | 4     | 4096              | 128              | bf16 (O2) |                      1.07 |                      2.23 |         2.1 |
| ViT g/14    | 4     | 4096              | 64               | bf16 (O2) |                      0.70 |                      1.40 |         2.0 |
| ViT bigG/14 | 4     | 4096              | 32               | bf16 (O2) |                      0.43 |                      0.91 |         2.1 |

<img src="img/Vision Transformer Training Throughput Comparison (2305).svg"/>

#### 8.1.3. Inference Performance Results

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
| ViT B/16 | 4          | 0.011                | 0.015                      | 1.4                   |
|          | 8          | 0.018                | 0.017                      | 1.0                   |

### 8.2. CLIP Results

#### 8.2.1. Training Accuracy Results

Training Accuracy: NVIDIA DGX SuperPOD (8 x 8 x A100 80GB for CLIP B/32 Model)

We followed the training recipe from [Open CLIP blog](https://laion.ai/blog/large-openclip/#12b-samples-seen) to verify
our training pipeline. Our results are displayed in the table below:

| Framework | Dataset               | Model Name | Batch Size | Samples Seen | ImageNet Top-1 |
|-----------|-----------------------|------------|------------|--------------|----------------|
| OpenCLIP  | LAION 400M            | B/32       | 32k        | 12B          | 62.90%         |
| NeMo      | Our Multimodal Blend* | B/32       | 32k        | 12B          | 60.13%         |

\**Our multimodal dataset is originated from Common Crawl with custom filtering and contains 670M image-caption pairs.*

We believe the final accuracy difference is due to the dataset, as LAION 400M is filtered with CLIP scores. To ensure
our implementation is consistent with OpenCLIP, we trained OpenCLIP with our dataset and found out that the loss curve
and validation accuracy were nearly identical to NeMo's CLIP.

#### 8.2.2. Training Performance Results

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
|           | Samples per Second               |    575 |   1171 |  2267 |  4567 |   8863 |
| CLIP g/14 | Perfect Linear Scaling (Samples) |    575 |   1150 |  2300 |  4599 |   9199 |
|           | Speedup                          |     1x |  2.04x | 3.94x | 7.94x | 15.42x |

<img src="img/CLIP g_14 NeMo Megatron Throughput (A100) (2305).svg"/>

- NVIDIA DGX SuperPODs (16 x 8 x H100 80GB for CLIP g/14 model)

|           |                                  |         |         |         | Nodes   |          |
|-----------|----------------------------------|---------|---------|---------|---------|----------|
|           |                                  | 1       | 2       | 4       | 8       | 16       |
|           | Samples per Second               |    930 |   1845 |   3482 |  6461 |  13672 |
| CLIP g/14 | Perfect Linear Scaling (Samples) |    930 |   1859 |   3718 |  7436 |  14873 |
|           | Speedup                          |     1x |  1.98x |  3.75x | 6.95x | 14.71x |

<img src="img/CLIP g_14 NeMo Megatron Throughput (H100) (2305).svg"/>

- DGX A100 vs. DGX H100: A Comparative Analysis of CLIP Training

| Model     | Nodes | Global Batch Size | Micro Batch Size | Precision | Global Batch / Sec (A100) | Global Batch / Sec (H100) | Speedup (x) |
|-----------|-------|-------------------|------------------|-----------|---------------------------|---------------------------|-------------|
| CLIP B/32 | 4     | 16000             | 500              | bf16 (O2) |                      3.01 |                      6.68 |         2.2 |
| CLIP H/14 | 4     | 3584              | 112              | bf16 (O2) |                      0.90 |                      1.92 |         2.1 |
| CLIP g/14 | 4     | 2560              | 80               | bf16 (O2) |                      0.89 |                      2.25 |         2.5 |

<img src="img/CLIP Training Throughput Comparison (2305).svg"/>

#### 8.2.3. Inference Performance Results

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

### 8.3. Stable Diffusion Results

#### 8.3.1. Training Accuracy Results

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

Below, we present the outcomes obtained from our own checkpoint following [Section 6.3.3](#633-stable-diffusion-training) , which can be compared to those
of the open-source Stable Diffusion 1.5.

<img src="img/Stable Diffusion FID-CLIP.png"/>

For Stable Diffusion 2.0 base, we followed the same configuration but evaluated CLIP score by VIT-g/14 version of the CLIP model. The result are presented below, our own checkpoint can be compared to open-source Stable Diffusion 2.0 base.

<img src="img/Stable Diffusion 2.0 FID-CLIP.png"/>

#### 8.3.2. Training Performance Results

We measured the throughput of training Stable Diffusion models on
different numbers of DGX A100 nodes and DGX H100 nodes, and we achieved near-linear
scaling on both platforms.

We are comparing the out-of-box performance on DGX H100 machines with the same configuration from DGX A100 machines.
This comparison is an apple-to-apple assessment, ensuring that we evaluate the relative performance of the two machine
types under equivalent conditions and configurations.

The tables and charts below show the performance results for SD v1.

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

The following is SD v2.0 result
- NVIDIA DGX SuperPODs (16 x 8 x A100 80GB for Stable Diffusion Res=512 model)
|                          |                                  |        |        |        | Nodes   |         |
|--------------------------|----------------------------------|--------|--------|--------|---------|---------|
|                          |                                  | 1      | 2      | 4      | 8       | 16      |
|                          | Samples per Second               | 197.68 | 391.39 | 775.02 | 1511.43 | 2844.44 |
| Stable Diffusion Res=512 | Perfect Linear Scaling (Samples) | 197.68 | 395.36 | 790.92 | 1581.44 | 3162.88 |
|                          | Speedup                          | 1x     | 1.98x  | 3.92x  | 7.65x   | 14.39x  |


#### 8.3.3. Inference Performance Results

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

The following is SD v2.0 performance
GPU: NVIDIA DGX A100 (1x A100 80 GB)
Batch Size: Synonymous with `num_images_per_prompt`

| Model                      | Batch Size | Sampler | Inference Steps | TRT FP 16 Latency (s) | FW FP 16 (AMP) Latency (s) | TRT vs FW Speedup (x) |
|----------------------------|------------|---------|-----------------|-----------------------|----------------------------|-----------------------|
| Stable Diffusion (Res=512) | 1          | PLMS    | 50              | 1.2                   | 3.2                        | 2.7                   |
| Stable Diffusion (Res=512) | 2          | PLMS    | 50              | 2.4                   | 5.0                        | 2.1                   |
| Stable Diffusion (Res=512) | 4          | PLMS    | 50              | 3.9                   | 8.5                        | 2.2                   |

### 8.4. InstructPix2Pix Results

#### 8.4.1. Training Quality Results

InstructPix2Pix is an image editing tool that transforms original images based on user instructions. For example, when
provided with a photo of [Toy Jensen](https://blogs.nvidia.com/blog/2022/12/22/toy-jensen-jingle-bells/), the AI can seamlessly edit the image according to your creative vision.

Here are some examples generated using our NeMo Stable Diffusion 1.2 model, fine-tuned with NeMo InstructPix2Pix. For
each instruction, we showcase 8 distinct images generated from different seeds:

- Original image
  <img src="img/tjjingle-1280x712.png" width="30%" />
- Instruction: Add fireworks to the background
  <img src="img/add_fireworks_to_the_background_7.5_1.2_1234_combine.jpg"/>
- Instruction: Make it on a beach
  <img src="img/make_it_in_on_a_beach_7.5_1.2_1234_combine.jpg"/>
- Instruction: Make it Van Gogh style
  <img src="img/make_it_Van_Gogh_style_7.5_1.2_1234_combine.jpg"/>

#### 8.4.2. Inference Performance Results

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
| InstructPix2Pix (Res=256) | 1          | N/A     | 100             | 1.0                   | 3.6                        | 3.6                   |
| InstructPix2Pix (Res=256) | 2          | N/A     | 100             | 1.3                   | 3.7                        | 2.8                   |
| InstructPix2Pix (Res=256) | 4          | N/A     | 100             | 2.2                   | 4.9                        | 2.2                   |

### 8.5. DreamBooth Results

#### 8.5.1. Training Quality Results

Here we show some insteresting results as an example of dreambooth script.

Prompt: A 'sks' dog in a bucket.

<img src="img/Dreambooth dog in a bucket.png" width="30%">

Prompt: A 'sks' dog in Acropolis.

<img src="img/Dreambooth dog at Acropolis.png" width="30%">

Prompt: A 'sks' dog in front of Eiffel tower.

<img src="img/Dreambooth Eiffel towel.png" width="30%">

Prompt: A 'sks' dog mecha robot.

<img src="img/Dreambooth mecha robot.png" width="30%">

The original source of images used for above results are from [link](https://github.com/google/dreambooth/tree/main/dataset/dog6)
and is subject to following [license](https://github.com/google/dreambooth/blob/main/dataset/references_and_licenses.txt).



#### 8.5.2. Inference Performance Results

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
| DreamBooth (Res=256) | 1          | DDIM    | 100             | 2.0                   | 5.6                        | 2.8                   |
| DreamBooth (Res=256) | 2          | DDIM    | 100             | 3.1                   | 9.0                        | 2.9                   |
| DreamBooth (Res=256) | 4          | DDIM    | 100             | 5.7                   | 16.0                       | 2.8                   |


### 8.6. ControlNet Results

#### 8.6.1. Traning Performace Results

We applied multiple optimizations to speedup the training throughput of controlnet. The following numbers are got from running on a single A100 GPU.

| Model      | Batch Size | Flash Attention | Channels Last   | Inductor   | samples per second | Memory Usage | Weak Scaling |
|------------|------------|-----------------|-----------------|------------|--------------------|--------------|--------------|
| ControlNet | 8          | NO              | NO              | NO         | 11.68              | 76G          | 1.0          |
| ControlNet | 8          | YES             | NO              | NO         | 16.40              | 33G          | 1.4          |
| ControlNet | 8          | YES             | YES             | NO         | 20.24              | 29G          | 1.73         |
| ControlNet | 8          | YES             | YES             | YES        | 21.52              | 29G          | 1.84         |
| ControlNet | 32         | YES             | YES             | YES        | 27.20              | 66G          | 2.33         |


#### 8.6.2. Training Quality Results

Here we show the examples of controlnet generations. The left column is the original input (upper) and conditioning image (lower).

Prompt: House.

<img src="img/ControlNet%20house.png" width="30%">

Prompt: House in oil painting style.

<img src="img/ControlNet%20painting%20house.png" width="30%">

Prompt: Bear.

<img src="img/ControlNet%20bears.png" width="30%">


#### 8.6.3. Inference Performance Results

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
| ControlNet (Res=512) | 1          | DDIM    | 50              | 1.7                   | 6.5                        | 3.8                   |
| ControlNet (Res=512) | 2          | DDIM    | 50              | 2.6                   | 7.1                        | 2.8                   |
| ControlNet (Res=512) | 4          | DDIM    | 50              | 4.4                   | 11.1                       | 2.5                   |
| ControlNet (Res=512) | 8          | DDIM    | 50              | 8.2                   | 21.1                       | 2.6                   |

### 8.7. Imagen Results

#### 8.7.1. Training Accuracy Results

We evaluate Imagen model with FID-CLIP curve, and comparing it to other open-source ckpt at same scale of
consumed sample.

FID (Fréchet Inception Distance) is a metric used to evaluate the quality of generated images in machine learning. It
measures the distance between the real image distribution and the distribution of generated images using the features
extracted by a pre-trained Inception model.

The VIT-L/14 version of the CLIP model was utilized to assess the relevance between image prompts and generated images.

The evaluation was conducted using different classifier-free guidance scales, specifically 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, and 6.0. The evaluation process involved generating 30,000 images from randomly selected prompts from the COCO2014
validation dataset, with 30 EDM steps on the base64 model 20 EDM steps on the sr256 model, and evaluating the results at a resolution of 256x256.

We have referred to but made certain modifications to the training recipe outlined
in [Imagen Paper](https://arxiv.org/abs/2205.11487).

Note that our curve is not fully comparable with the plots on the paper for the few reasons: 1. Our dataset is different and smaller than the one Imagen paper uses. 2. We chose to train a much smaller variation of the model (500M) for convergence instead of the proposed one (2B) 3. While in paper they use T5-xxl (4096 dimension) encoder, We use T5-11b (1024 dimension) encoder during training due to space considerations. The FID score is slightly higher than our Stable Diffusion results as for the fact that we only used a subset of dataset to train Imagen because caching T5 embedding is expensive on disk resource.

\**Our multimodal dataset is originated from Common Crawl with custom filtering.*

Below, we present the outcomes obtained from our own checkpoint following [Section 6.3.6](#636-imagen-training).

##ADD Imagen FID-CLIP curve

#### 8.7.2. Training Performance Results

We measured the throughput of training Imagen models on
different numbers of DGX A100 nodes and DGX H100 nodes, and we achieved near-linear
scaling on both platforms.

We are comparing the out-of-box performance on DGX H100 machines with the same configuration from DGX A100 machines.
This comparison is an apple-to-apple assessment, ensuring that we evaluate the relative performance of the two machine
types under equivalent conditions and configurations.

The tables and charts below show the performance results.

- NVIDIA DGX SuperPODs (16 x 8 x A100 80GB for Imagen Base 2B model)

|                         |                                  |             |             |             | Nodes       |             |
|-------------------------|----------------------------------|-------------|-------------|-------------|-------------|-------------|
|                         |                                  |           1 |           2 |           4 |           8 |          16 |
|                         | Samples per Second               | 344.09 | 632.88 | 1256.44 | 2500.61 | 4940.89 |
| ImagenBase (2B, Res=64) | Perfect Linear Scaling (Samples) | 344.09 |  688.17 | 1376.34 | 2752.69 | 5505.37 |
|                         | Speedup                          |          1x |       1.84x |       3.65x |       7.27x |      14.36x |

<img src="img/ImagenBase (2B, Res=64) NeMo Megatron Throughput (A100).svg"/>

- NVIDIA DGX SuperPODs (16 x 8 x A100 80GB for Imagen Base 500M model)

|                         |                                  |        |         |         | Nodes   |          |
|-------------------------|----------------------------------|--------|---------|---------|---------|----------|
|                         |                                  |      1 |       2 |       4 |       8 |       16 |
|                         | Samples per Second               | 645.65 | 1216.15 | 2412.25 | 4870.39 |  9737.31 |
| ImagenBase (2B, Res=64) | Perfect Linear Scaling (Samples) | 645.65 | 1291.30 | 2582.60 | 5165.20 | 10330.39 |
|                         | Speedup                          | 1x     | 1.88x   | 3.74x   | 7.54x   | 15.08x   |

<img src="img/ImagenBase (500M, Res=64) NeMo Megatron Throughput (A100).svg"/>

- NVIDIA DGX SuperPODs (16 x 8 x H100 80GB for Imagen Base 2B model)

|                         |                                  |        |         |         | Nodes   |          |
|-------------------------|----------------------------------|--------|---------|---------|---------|----------|
|                         |                                  |      1 |       2 |       4 |       8 |       16 |
| ImagenBase (2B, Res=64) | Samples per Second               | 717.89 | 1339.26 | 2641.90 | 5073.07 |  9260.68 |
|                         | Perfect Linear Scaling (Samples) | 717.89 | 1435.78 | 2871.56 | 5743.13 | 11486.26 |
|                         | Speedup                          |     1x |   1.87x |   3.68x |   7.07x |    12.9x |

<img src="img/ImagenBase (2B, Res=64) NeMo Megatron Throughput (H100).svg"/>

- DGX A100 vs. DGX H100: A Comparative Analysis of Imagen Training

| Model                     | Nodes | Global Batch | Micro Batch | Precision | Sec/Batch (A100) | Sec/Batch (H100) | Speedup (x) |
|---------------------------|-------|--------------|-------------|-----------|------------------|------------------|-------------|
| ImagenBase (500M, Res=64) |     4 |         4096 |         128 | bf16 (O1) |            1.198 |            2.364 |         2.0 |
| ImagenBase (2B, Res=64)   |     4 |         4096 |         128 | bf16 (O1) |            1.269 |            2.514 |         2.0 |
| ImagenSR (400M, Res=256)  |     4 |         4096 |         128 | bf16 (O1) |            1.213 |            2.247 |         1.9 |
| ImagenSR (600M, Res=256)  |     4 |         4096 |         128 | bf16 (O1) |            1.007 |            1.885 |         1.9 |
| ImagenSR (600M, Res=1024) |     4 |         1024 |          32 | bf16 (O1) |            1.006 |            1.830 |         1.8 |

<img src="img/Imagen Training Throughput Comparison.svg"/>

#### 8.7.3. Inference Performance Results

Latency times are started directly before the text encoding (T5) and stopped directly after the output image (UNet).
For framework we provide both using the Torch Automated Mixed Precision (AMP) or pure FP16 computation. For TRT, we export the various
models
with the FP16 acceleration. We use the optimized TRT engine setup present in the deployment directory to get the numbers
in the same environment as the framework.

GPU: NVIDIA DGX A100 (1x A100 80 GB)
Batch Size: Synonymous with `num_images_per_prompt`

| Model                   | Batch Size | Sampler | Inference Steps | TRT FP16 Latency (s) | FW FP16 Latency (s) | FW FP16 (AMP) Latency (s) | TRT vs FW Speedup (x) |
|-------------------------|------------|---------|-----------------|----------------------|---------------------|---------------------------|-----------------------|
| Imagen Base64x64 500M   | 4          | EDM     | 30              | 2.72                 | 5.011               | 5.805                     | 1.84 \| 2.13          |
| Imagen SR256x256 600M   | 4          | EDM     | 20              | 1.79                 | 3.181               | 3.549                     | 1.78 \| 1.98          |
| Imagen SR1024x1024 600M | 4          | EDM     | 20              | 16.26                | 23.588              | 27.799                    | 1.45 \| 1.71          |

## 9. Known Issues

* Due to a [Lightning DDP limitation](https://github.com/Lightning-AI/lightning/issues/3325), the precaching module may
drop about 0.01% to 0.1% of input data. The specific ratio will depend on the cluster configuration, tarfile chunk_size,
precaching batch_size and dataset size, but will be consistent across runs. We anticipate that dropping a small percentage of data will not have a significant impact on model training.
* The CLIP model trained with BF16 shows inconsistent predictions at FP32 precision (FP16/BF16 predictions are correct though).

