# NeMo Framework
## Open Beta

Scripts and code to provide end-to-end data preparation and training for
NeMo Framework.

The most recent version of the README can be found at [https://ngc.nvidia.com/containers/ea-bignlp:nemofw-training](https://ngc.nvidia.com/containers/ea-bignlp:nemofw-training).

## Table of contents
- [1. Model Overview](#1-model-overview)
- [2. Feature Matrix](#2-feature-matrix)
  * [2.1. GPT Models](#21-gpt-models)
  * [2.2. T5 and mT5 Models](#22-t5-and-mt5-models)
  * [2.3. BERT Models](#23-bert-models)
- [3. Setup](#3-setup)
  * [3.1. Support Matrix](#31-support-matrix)
- [4. Cloud Service Providers](#4-cloud-service-providers)
  * [4.1. Cluster Bring-Up](#41-cluster-bring-up)
    + [4.1.1. Common](#411-common)
    + [4.1.2. OCI](#412-oci)
    + [4.1.3. AWS](#413-aws)
  * [4.2. Cluster Validation](#42-cluster-validation)
    + [4.2.1. Validation Script Usage](#421-validation-script-usage)
    + [4.2.2 Running tests manually](#422-running-tests-manually)
  * [4.3. Config Modifications](#43-config-modifications)
    + [4.3.1 Set NCCL Topology](#431-set-nccl-topology)
    + [4.3.2 Environment Variables](#432-environment-variables)
      - [4.3.2.1 Azure Variables](#4321-azure-variables)
      - [4.3.2.2 AWS Variables](#4322-aws-variables)
- [5. Quick Start Guide](#5-quick-start-guide)
  * [5.1. Training NeMo Framework Models](#51-training-nemo-framework-models)
    + [5.1.1. Prepare Environment](#511-prepare-environment)
      - [5.1.1.1. Slurm](#5111-slurm)
      - [5.1.1.2. Base Command Platform](#5112-base-command-platform)
      - [5.1.1.3. General Configuration](#5113-general-configuration)
    + [5.1.2. Data Preparation](#512-data-preparation)
      - [5.1.2.1. Data Preparation for GPT Models](#5121-data-preparation-for-gpt-models)
        * [5.1.2.1.1. Slurm](#51211-slurm)
        * [5.1.2.1.2. Base Command Platform](#51212-base-command-platform)
        * [5.1.2.1.3. Common](#51213-common)
      - [5.1.2.2. Data Preparation for T5 Models](#5122-data-preparation-for-t5-models)
        * [5.1.2.2.1. Slurm](#51221-slurm)
        * [5.1.2.2.2. Base Command Platform](#51222-base-command-platform)
        * [5.1.2.2.3. Common](#51223-common)
      - [5.1.2.3. Data Preparation for mT5 Models](#5123-data-preparation-for-mt5-models)
        * [5.1.2.3.1. Slurm](#51231-slurm)
        * [5.1.2.3.2. Base Command Platform](#51232-base-command-platform)
        * [5.1.2.3.3. Common](#51233-common)
      - [5.1.2.4. Data Preparation for BERT Models](#5124-data-preparation-for-bert-models)
        * [5.1.2.4.1. Slurm](#51241-slurm)
        * [5.1.2.4.2. Base Command Platform](#51242-base-command-platform)
        * [5.1.2.4.3. Common](#51243-common)
        * [5.1.2.4.4. LDDL](#51244-lddl)
  * [5.2. Training with Predefined Configurations](#52-training-with-predefined-configurations)
    + [5.2.1. Predefined Configurations of GPT Models](#521-predefined-configurations-of-gpt-models)
    + [5.2.2. Predefined Configurations of T5 Models](#522-predefined-configurations-of-t5-models)
    + [5.2.3. Predefined Configurations of mT5 Models](#523-predefined-configurations-of-mt5-models)
    + [5.2.4. Training Logs with TensorBoard and Weights and Biases](#524-training-logs-with-tensorboard-and-weights-and-biases)
    + [5.2.5. Predefined Configurations of BERT Models](#525-predefined-configurations-of-bert-models)
  * [5.3. Using AutoConfigurator to Find the Optimal Configuration](#53-using-autoconfigurator-to-find-the-optimal-configuration)
    + [5.3.1. AutoConfigurator Capabilities](#531-autoconfigurator-capabilities)
      - [5.3.1.1. Model Size Recommendation](#5311-model-size-recommendation)
      - [5.3.1.2. Base Config Generation](#5312-base-config-generation)
      - [5.3.1.3. Training AutoConfigurator HP Search](#5313-training-autoconfigurator-hp-search)
      - [5.3.1.4. Inference AutoConfigurator HP Search](#5314-inference-autoconfigurator-hp-search)
    + [5.3.2. Usage](#532-usage)
      - [5.3.2.1. General Configuration](#5321-general-configuration)
        * [5.3.2.1.1. Slurm](#53211-slurm)
        * [5.3.2.1.2. Base Command Platform](#53212-base-command-platform)
      - [5.3.2.2. Running Predefined Configs](#5322-running-predefined-configs)
        * [5.3.2.2.1. Model Config](#53221-model-config)
        * [5.3.2.2.2. Base Config Generation](#53222-base-config-generation)
        * [5.3.2.2.3. Training AutoConfigurator HP Search](#53223-training-autoconfigurator-hp-search)
        * [5.3.2.2.4. Inference AutoConfigurator HP Search](#53224-inference-autoconfigurator-hp-search)
      - [5.3.2.3. Running Custom Model Size Configs](#5323-running-custom-model-size-configs)
      - [5.3.2.4. Interpreting the Results](#5324-interpreting-the-results)
      - [5.3.2.5. Logging Runs with Weights and Biases](#5325-logging-runs-with-weights-and-biases)
  * [5.4. Training with Custom Configurations](#54-training-with-custom-configurations)
    + [5.4.1. Example: Changing Embedding Type for T5 Models](#541-example--changing-embedding-type-for-t5-models)
  * [5.5. Bring Your Own Dataset](#55-bring-your-own-dataset)
    + [5.5.1. Slurm](#551-slurm)
    + [5.5.2. Base Command Platform](#552-base-command-platform)
    + [5.5.3. Common](#553-common)
  * [5.6. Model Training](#56-model-training)
    + [5.6.1. GPT Training](#561-gpt-training)
      - [5.6.1.1. Slurm](#5611-slurm)
      - [5.6.1.2. Base Command Platform](#5612-base-command-platform)
    + [5.6.2. T5 Training](#562-t5-training)
      - [5.6.2.1. Slurm](#5621-slurm)
      - [5.6.2.2. Base Command Platform](#5622-base-command-platform)
    + [5.6.3. mT5 Training](#563-mt5-training)
      - [5.6.3.1. Slurm](#5631-slurm)
      - [5.6.3.2. Base Command Platform](#5632-base-command-platform)
    + [5.6.4. BERT Training](#564-bert-training)
      - [5.6.4.1. Slurm](#5641-slurm)
      - [5.6.4.2. Base Command Platform](#5642-base-command-platform)
  * [5.7. Resuming Training with Different Number of Nodes](#57-resuming-training-with-different-number-of-nodes)
  * [5.8. Checkpoint Conversion](#58-checkpoint-conversion)
    + [5.8.1. GPT Conversion](#581-gpt-conversion)
      - [5.8.1.1. Common](#5811-common)
      - [5.8.1.2. Slurm](#5812-slurm)
      - [5.8.1.3. Base Command Platform](#5813-base-command-platform)
    + [5.8.2. T5 Conversion](#582-t5-conversion)
      - [5.8.2.1. Common](#5821-common)
      - [5.8.2.2. Slurm](#5822-slurm)
      - [5.8.2.3. Base Command Platform](#5823-base-command-platform)
    + [5.8.3. mT5 Conversion](#583-mt5-conversion)
      - [5.8.3.1. Common](#5831-common)
      - [5.8.3.2. Slurm](#5832-slurm)
      - [5.8.3.3. Base Command Platform](#5833-base-command-platform)
  * [5.9. Model Fine-tuning](#59-model-fine-tuning)
    + [5.9.1. T5 Fine-tuning](#591-t5-fine-tuning)
      - [5.9.1.1. Common](#5911-common)
      - [5.9.1.2. Slurm](#5912-slurm)
      - [5.9.1.3. Base Command Platform](#5913-base-command-platform)
    + [5.9.2. mT5 Fine-tuning](#592-mt5-fine-tuning)
      - [5.9.2.1. Common](#5921-common)
      - [5.9.2.2. Slurm](#5922-slurm)
      - [5.9.2.3. Base Command Platform](#5923-base-command-platform)
    + [5.9.3. GPT Supervised Fine-tuning](#593-gpt-supervised-fine-tuning)
      - [5.9.3.1. Common](#5931-common)
      - [5.9.3.2. Slurm](#5932-slurm)
      - [5.9.3.3. Base Command Platform](#5933-base-command-platform)
    + [5.9.4. Fine-tuning on Custom Tasks](#594-fine-tuning-on-custom-tasks)
      - [5.9.4.1. T5 and mT5](#5941-t5-and-mt5)
      - [5.9.4.2. GPT](#5942-gpt)
  * [5.10. Model Prompt Learning](#510-model-prompt-learning)
    + [5.10.1. GPT Prompt Learning](#5101-gpt-prompt-learning)
      - [5.10.1.1. Common](#51011-common)
      - [5.10.1.2. Slurm](#51012-slurm)
      - [5.10.1.3. Base Command Platform](#51013-base-command-platform)
    + [5.10.2. T5 and mT5 Prompt Learning](#5102-t5-and-mt5-prompt-learning)
      - [5.10.2.1. Common](#51021-common)
      - [5.10.2.2. Slurm](#51022-slurm)
      - [5.10.2.3. Base Command Platform](#51023-base-command-platform)
  * [5.11. Model Adapter Learning and IA3 Learning](#511-model-adapter-learning-and-ia3-learning)
    + [5.11.1. GPT Adapter Learning and IA3 Learning](#5111-gpt-adapter-learning-and-ia3-learning)
      - [5.11.1.1. Common](#51111-common)
      - [5.11.1.2. Slurm](#51112-slurm)
      - [5.11.1.3. Base Command Platform](#51113-base-command-platform)
    + [5.11.2. T5 Adapter Learning and IA3 Learning](#5112-t5-adapter-learning-and-ia3-learning)
      - [5.11.2.1. Common](#51121-common)
      - [5.11.2.2. Slurm](#51122-slurm)
      - [5.11.2.3. Base Command Platform](#51123-base-command-platform)
  * [5.12 LoRA Model and Generalized PEFT Framework](#512-lora-model-and-generalized-peft-framework)
    + [5.12.1 PEFT Training and Inference for GPT-style Models](#5121-peft-training-and-inference-for-gpt-style-models)
      - [5.12.1.1 PEFT Training and Inference](#51211-peft-training-and-inference)
      - [5.12.2 PEFT Training and Inference for mT5/T5-style Models](#5122-peft-training-and-inference-for-mt5-t5-style-models)
      - [5.12.2.1 PEFT Training and Inference](#51221-peft-training-and-inference)
  * [5.13. Model Evaluation](#513-model-evaluation)
    + [5.13.1. GPT Evaluation](#5131-gpt-evaluation)
      - [5.13.1.1. Common](#51311-common)
      - [5.13.1.2. Slurm](#51312-slurm)
      - [5.13.1.3. Base Command Platform](#51313-base-command-platform)
      - [5.13.1.4 Interleaved Pipeline Parallelism](#51314-interleaved-pipeline-parallelism)
    + [5.13.2. T5 Evaluation](#5132-t5-evaluation)
      - [5.13.2.1. Common](#51321-common)
      - [5.13.2.2. Slurm](#51322-slurm)
      - [5.13.2.3. Base Command Platform](#51323-base-command-platform)
    + [5.13.3. mT5 Evaluation](#5133-mt5-evaluation)
      - [5.13.3.1. Common](#51331-common)
      - [5.13.3.2. Slurm](#51332-slurm)
      - [5.13.3.3. Base Command Platform](#51333-base-command-platform)
    + [5.13.4. Prompt Learned GPT Evaluation](#5134-prompt-learned-gpt-evaluation)
      - [5.13.4.1. Common](#51341-common)
      - [5.13.4.2. Slurm](#51342-slurm)
      - [5.13.4.3. Base Command Platform](#51343-base-command-platform)
    + [5.13.5. Prompt Learned T5 and mT5 Evaluation](#5135-prompt-learned-t5-and-mt5-evaluation)
      - [5.13.5.1. Common](#51351-common)
      - [5.13.5.2. Slurm](#51352-slurm)
      - [5.13.5.3. Base Command Platform](#51353-base-command-platform)
    + [5.13.6. Adapter Learned and IA3 Learned GPT Evaluation](#5136-adapter-learned-and-ia3-learned-gpt-evaluation)
      - [5.13.6.1. Common](#51361-common)
      - [5.13.6.2. Slurm](#51362-slurm)
      - [5.13.6.3. Base Command Platform](#51363-base-command-platform)
    + [5.13.7. Adapter Learned and IA3 Learned T5 Evaluation](#5137-adapter-learned-and-ia3-learned-t5-evaluation)
      - [5.13.7.1. Common](#51371-common)
      - [5.13.7.2. Slurm](#51372-slurm)
      - [5.13.7.3. Base Command Platform](#51373-base-command-platform)
  * [5.14. Model Export](#514-model-export)
    + [5.14.1. GPT Export](#5141-gpt-export)
      - [5.14.1.1. Common](#51411-common)
      - [5.14.1.2. Slurm](#51412-slurm)
      - [5.14.1.3. Base Command Platform](#51413-base-command-platform)
    + [5.14.2. T5 Export](#5142-t5-export)
      - [5.14.2.1. Common](#51421-common)
      - [5.14.2.2. Slurm](#51422-slurm)
      - [5.14.2.3. Base Command Platform](#51423-base-command-platform)
    + [5.14.3. mT5 Export](#5143-mt5-export)
      - [5.14.3.1. Common](#51431-common)
      - [5.14.3.2. Slurm](#51432-slurm)
      - [5.14.3.3. Base Command Platform](#51433-base-command-platform)
  * [5.15 Instruction Following via Supervised Finetuning (SFT)](#515-instruction-following-via-supervised-finetuning--sft-)
    + [5.15.1 SFT Data Formatting](#5151-sft-data-formatting)
    + [5.15.2 SFT Training](#5152-sft-training)
  * [5.16. Reinforcement Learning from Human Feedback](#516-reinforcement-learning-from-human-feedback)
    + [5.16.1. Reward Model Training](#5161-reward-model-training)
      - [5.16.1.1 Data preprocessing](#51611-data-preprocessing)
      - [5.16.1.2 Training a Reward Model](#51612-training-a-reward-model)
      - [5.16.1.3 Reward Model Evaluation](#51613-reward-model-evaluation)
    + [5.16.2. PPO Training](#5162-ppo-training)
      - [5.16.2.1 Launching the Reward Model Inference Server](#51621-launching-the-reward-model-inference-server)
      - [5.16.2.2 Launching the Initial Policy Inference Server](#51622-launching-the-initial-policy-inference-server)
      - [5.16.2.3 Launching the PPO Critic Training and Inference Server](#51623-launching-the-ppo-critic-training-and-inference-server)
      - [5.16.2.4 Launching the PPO Actor Training](#51624-launching-the-ppo-actor-training)
      - [5.16.2.5 Launching all jobs at once with SLURM](#51625-launching-all-jobs-at-once-with-slurm)
      - [5.16.2.6 Ensuring consistency between jobs](#51626-ensuring-consistency-between-jobs)
      - [5.16.2.7 PPO Hyper-parameters](#51627-ppo-hyper-parameters)
    + [5.16.3. Future Work](#5163-future-work)
  * [5.17 Curating pretraining datasets with the NeMo Data Curator](#517-curating-pretraining-datasets-with-the-nemo-data-curator)
- [6. Deploying the NeMo Megatron Model](#6-deploying-the-nemo-megatron-model)
  * [6.1. Run NVIDIA Triton Server with Generated Model Repository](#61-run-nvidia-triton-server-with-generated-model-repository)
- [6.2. GPT Text Generation with Ensemble](#62-gpt-text-generation-with-ensemble)
- [6.3. UL2 Checkpoint Deployment](#63-ul2-checkpoint-deployment)
- [7. Performance](#7-performance)
  * [7.1. GPT Results](#71-gpt-results)
    + [7.1.1. Training Accuracy Results](#711-training-accuracy-results)
    + [7.1.2. Training Performance Results](#712-training-performance-results)
    + [7.1.3. Inference Performance](#713-inference-performance)
  * [7.2. T5 Results](#72-t5-results)
    + [7.2.1. Training Accuracy Results](#721-training-accuracy-results)
    + [7.2.2. Training Performance Results](#722-training-performance-results)
    + [7.2.3. Inference Performance](#723-inference-performance)
  * [7.3. mT5 Results](#73-mt5-results)
    + [7.3.1. Training Accuracy Results](#731-training-accuracy-results)
    + [7.3.2. Training Performance Results](#732-training-performance-results)
    + [7.3.3. Inference Performance](#733-inference-performance)
  * [7.4. BERT Results](#74-bert-results)
    + [7.4.1. Training Accuracy Results](#741-training-accuracy-results)
    + [7.4.2. Training Performance Results](#742-training-performance-results)
    + [7.4.3. Training Performance Results (LDDL)](#743-training-performance-results--lddl-)
- [8. Changelog](#8-changelog)
- [9. Known Issues](#9-known-issues)

<!-- /TOC -->

## 1. Model Overview
<a id="markdown-model-overview" name="model-overview"></a>

NeMo Framework allows developers to effectively train and scale language
models to billions of parameters. With NeMo Framework, you can train different variants of GPT, Bert and T5 style models,
and scale them to multiple nodes on NVIDIA DGX SuperPOD deployments. This deep learning (DL) software stack is optimized for DGX
SuperPOD configurations using NVIDIA InfiniBand technology to provide efficient on-premises compute for training
and inferring complex workloads.

The model parallelism techniques of NeMo Framework enable the efficient training of large models that do not fit in
the memory of a single GPU. In the training tasks, tensor (intra-layer) and pipeline (inter-layer) model parallelism
are adopted. Tensor model parallelism partitions individual transformer layers over multiple devices. Pipeline
model parallelism stripes layers of a model over multiple devices. For more details, refer to
[this paper](https://arxiv.org/pdf/2104.04473.pdf).

Our latest techniques, sequence parallelism and selective activation recomputation, bring up to `~30%` faster 
training time for GPT models ranging from 20B to 1T parameters.
Sequence parallelism expands tensor-level model parallelism, by 
noticing that the regions of a transformer layer that have not previously been parallelized are independent 
along the sequence dimension. By splitting these layers along the sequence dimension we can distribute 
the compute and, most importantly, the activation memory for these regions across the tensor parallel devices.
Selective activation recomputation improves cases where memory constraints force us to recompute some, 
but not all, of the activations. For more details, refer to [this paper](https://arxiv.org/abs/2205.05198).

**GPT architecture**

<img src="img/model_overview.png"/>

Figure 1: The GPT family architecture. The 5B variant includes 24 transformer layers, a hidden size of 4096, and 32 attention heads. The sequence length is 2048, and the optimizer is Adam. This variant uses tensor parallelism of 2.

## 2. Feature Matrix
<a id="markdown-feature-matrix" name="feature-matrix"></a>

### 2.1. GPT Models
<a id="markdown-gpt-models" name="gpt-models"></a>

| Feature                                                 | Training                             | Inference                                                                                                                                                                                                                                                                                                                 |
| ------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data parallelism                | Yes                    | N/A                                                                                                                                                                  |
| Tensor parallelism              | Yes                    | Yes                                                                                                                                                               |
| Pipeline parallelism            | Yes                     | Yes (Megatron-LM checkpoints)                                                                                                                          |
| Interleaved Pipeline Parallelism Schedule            | Yes                     | N/A                                                                                                                          |
| Sequence parallelism            | Yes                     | No                                                                                                                       |
| Selective activation checkpointing | Yes                     | No                                                                                                                       |
| Gradient checkpointing          | Yes                    | N/A                                                                                                                                                                  |
| Partial gradient checkpointing  | Yes                    | N/A                                                                                                                                                                  |
| FP32/TF32                       | Yes                    | Yes (FP16 enabled by default)                                                                                                                                     |
| AMP/FP16                        | No | Yes                                                                                                                                                               |
| BF16                            | Yes  | Yes                                                                                                                                                                |
| TransformerEngine/FP8                      | Yes  | No                                                                                                                                                                |
| Multi-GPU                       | Yes                    | Yes                                                                                                                                                               |
| Multi-Node                      | Yes                    | Yes                                                                                                                                                               |
| Inference deployment            | N/A                    | [NVIDIA Triton supported](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton), Faster Transformer |
| SW stack support                | Slurm DeepOps/Base Command Manager/Base Command Platform          | Slurm DeepOps/Base Command Manager/Base Command Platform                                                                                                                                                     |
| Distributed data preprocessing | Yes (the Pile only)       | N/A                                                                                                                                                                  |
| NVfuser                         | No             | N/A                                                                                                                                                                  |
| P-Tuning and Prompt Tuning                | Yes             | N/A                                                                                                                                                                  |
| IA3 and Adapter learning                | Yes             | N/A                                                                                                                                                                  |
| Distributed Optimizer   | Yes             | N/A                                                                                                                                                                  |

### 2.2. T5 and mT5 Models
<a id="markdown-t5-and-mt5-models" name="t5-and-mt5-models"></a>

| Feature                          | Training                                                 | Inference |
|----------------------------------|----------------------------------------------------------|:---------:|
| Data parallelism                 | Yes                                                      |    N/A    |
| Tensor parallelism               | Yes                                                      |    No     |
| Pipeline parallelism             | Yes                                                      |    No     |
| Sequence parallelism            | No                     | No                                                                                                                       |
| Selective activation checkpointing | No                     | No                                                                                                                       |
| Gradient checkpointing           | Yes                                                      |    N/A    |
| Partial gradient checkpointing   | Yes                                                      |    N/A    |
| FP32/TF32                        | Yes                                                      |    No     |
| AMP/FP16                         | No                                                       |    No     |
| BF16                             | Yes                                                      |    No     |
| Multi-GPU                        | Yes                                                      |    No     |
| Multi-Node                       | Yes                                                      |     No    |
| Inference deployment             | N/A                                                      |    No     |
| SW stack support                 | Slurm DeepOps/Base Command Manager/Base Command Platform |    No     |
| Distributed data preprocessing   | Yes (the Pile dataset for T5, mC4 dataset for mT5)       |    N/A    |
| NVfuser                          | No                                                       |    N/A    |
| P-Tuning and Prompt Tuning                | Yes             | N/A                                                                                                                                                                  |
| IA3 and Adapter learning                | Yes             | N/A                                                                                                                                                                  |
| AutoConfigurator                          | Yes                                                       |    N/A    |
| Distributed Optimizer   | Yes             | N/A      |
| Mixture of Experts   | Yes (no expert parallelism)            | N/A      |

### 2.3. BERT Models
<a id="markdown-bert-models" name="bert-models"></a>

| Feature                                                 | Training                             | Inference                                                                                                                                                                                                                                                                                                                 |
| ------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data parallelism                | Yes                    | N/A                                                                                                                                                                  |
| Tensor parallelism              | Yes                    | N/A                                                                                                                                                               |
| Pipeline parallelism            | Yes                     | N/A                                                                                                                           |
| Sequence parallelism            | Yes                     | N/A                                                                                                                        |
| Selective activation checkpointing | Yes                     | N/A                                                                                                                        |
| Gradient checkpointing          | Yes                    | N/A                                                                                                                                                                  |
| Partial gradient checkpointing  | Yes                    | N/A                                                                                                                                                                  |
| FP32/TF32                       | Yes                    | N/A                                                                                                                                      |
| AMP/FP16                        | No | N/A                                                                                                                                                               |
| BF16                            | Yes  | N/A                                                                                                                                                                |
| Multi-GPU                       | Yes                    | N/A                                                                                                                                                                |
| Multi-Node                      | Yes                    | N/A                                                                                                                                                                |
| Inference deployment            | N/A                    | N/A  |
| SW stack support                | Slurm DeepOps/Base Command Manager/Base Command Platform          |N/A                                                                                                                                                     |
| Distributed data preprocessing | Yes (the Pile only)       | N/A                                                                                                                                                                  |
| NVfuser                         | Yes             | N/A                                                                                                                                                                  |
| P-Tuning and Prompt Tuning                | N/A             | N/A                                                                                                                                                                  |
| IA3 and Adapter learning                | N/A             | N/A                                                                                                                                                                  |
| Distributed Optimizer              | Yes             | N/A                                                                                                                                                                  |

## 3. Setup
<a id="markdown-setup" name="setup"></a>

### 3.1. Support Matrix
<a id="markdown-support-matrix" name="support-matrix"></a>

| Software                | Version          |
|-------------------------|------------------|
| NVIDIA Triton           | 2.24.0           |
| FasterTransformer       | v5.3+f8e42aa     |
| TransformerEngine       | v0.11+b172bad    |
| MegatronCore            | 4f8e9ac          |
| PyTorch                 | 2.1.0a0+fe05266  |
| NeMo                    | 1.20.0+2baef81   |
| PyTorch Lightning       | 1.9.4            |
| Hydra                   | 1.2.0            |
| CUDA                    | NVIDIA CUDA 12.1 |
| cuBLAS                  | 12.1.3.1         |
| cuDNN                   | 8.9.0.131        |
| NCCL                    | 2.17.1           |
| Container OS            | Ubuntu 20.04     |
| rdma-core               | 36.0             |
| GDRcopy                 | 2.3              |
| HPC-X                   | 2.13             |
| Base Command Manager    | 1.0.0            |
| DeepOps                 | 21.06            |

## 4. Cloud Service Providers
<a id="markdown-cloud-service-providers" name="cloud-service-providers"></a>

### 4.1. Cluster Bring-Up
<a id="markdown-cluster-bring-up" name="cluster-bring-up"></a>

#### 4.1.1. Common
<a id="markdown-common" name="common"></a>

To set up a Slurm cluster for NeMo Framework, we recommend using [Nephele](https://github.com/nvidia/nephele). This cluster deployment tool has been tested on Azure, AWS, and Oracle Cloud.
We recommend hosting Nephele on a new VM instance in the CSP of your choice. To get started:
- Clone the Nephele repo
- Install the dependencies
- Modify `nephele.conf`
    - Add your CSP credentials
    - Change `REPLICAS_x8a100` to the number of nodes in your desired cluster

You can then run `./nephele init` and `./nephele create`.

We also recommend mounting an external persistent NFS once the cluster is up and running (ensure it is mounted on all nodes) and using this to configure and run NeMo Framework.

The above steps apply to all CSPs, including Azure, AWS, and OCI.
Some modifications are necessary for OCI and AWS and are detailed below.
Note that for OCI, a custom image must be imported, which should be done before running `./nephele create`.

#### 4.1.2. OCI
<a id="markdown-oci" name="oci"></a>

NeMo Framework supports running training and inference containers on OCI. For more details about orchestration scripts, reach out to [oci_nm@nvidia.com](mailto:oci_nm@nvidia.com)

#### 4.1.3. AWS
<a id="markdown-aws" name="aws"></a>
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
<a id="markdown-cluster-validation" name="cluster-validation"></a>

Before running the cluster validation script, ensure your NGC credentials have been added to `~/.config/enroot/.credentials` on all nodes.

The cluster validation script at `csp_tools/<csp>/cluster_validation.sh` will run GPU diagnostics and test NCCL node-to-node bus bandwidth.
The logs from these tests will be stored at `results/cluster_validation`. The script will list any nodes that fail these tests.
These nodes should be replaced or restarted through the CSP UI.

#### 4.2.1. Validation Script Usage
<a id="markdown-validation-script-usage" name="validation-script-usage"></a>

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

By default, the script will run both the GPU diagnostics and the NCCL test. You can choose to run only one or the other by specifying:
- `--dcgm`: run GPU diagnostics only
- `--nccl`: run NCCL test only

See `bash cluster_validation.sh -h` for more information.

#### 4.2.2 Running tests manually
<a id="markdown-running-tests-manually" name="running-tests-manually"></a>

The `cluster_validation.sh` script is essentially a wrapper of the 2 Slurm job scripts in the CSP directories. If you prefer, you can run these jobs manually.
Make sure to use the Slurm job script in your corresponding CSP's path (`csp_tools/<csp>/dcgmi_diag.sh` and `csp_tools/<csp>/nccl.sh`)

For the GPU diagnostics job, provide these arguments when submitting the job to Slurm:

```
sbatch -p <partition> -w <node list> -o <job log file> dcgmi_diag.sh
```

For the NCCL test job, `cluster_validation.sh` performs a pair-wise sweep of the nodes, as this is a sufficient test, but you can test with a different number of nodes if desired.

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
<a id="markdown-config-modifications" name="config-modifications"></a>
Before launching jobs some changes to the config must be made.

#### 4.3.1 Set NCCL Topology
<a id="markdown-generate-nccl-topology" name="generate-nccl-topology"></a>
The NCCL topology file is unique for each CSP, and can be found in their corresponding folders (`csp_tools/<csp>/topo.xml`)

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
<a id="markdown-environment-variables" name="environment-variables"></a>

##### 4.3.2.1 Azure Variables
<a id="markdown-azure-variables" name="azure-variables"></a>
Set these environment variables in `config.yaml` (these are only needed for Azure):
```
env_vars:
  UCX_IB_PCI_RELAXED_ORDERING: auto
  NCCL_IB_PCI_RELAXED_ORDERING: 2
  NCCL_IB_TIMEOUT: 22
  NCCL_DEBUG: INFO
```

##### 4.3.2.2 AWS Variables
<a id="markdown-aws-variables" name="aws-variables"></a>
AWS recommends setting the following flag to avoid data corruption:
```
env_vars:
  NCCL_PROTO: simple
```

Setting this flag reduces training throughput by roughly 2%.

## 5. Quick Start Guide
<a id="markdown-quick-start-guide" name="quick-start-guide"></a>

### 5.1. Training NeMo Framework Models
<a id="markdown-training-nemo-megatron-models" name="training-nemo-megatron-models"></a>

#### 5.1.1. Prepare Environment
<a id="markdown-prepare-environment" name="prepare-environment"></a>

<!--
The whole solution uses a set of Docker containers executed at the Slurm
cluster using the pyxis plug-in Base Command Platform cluster. The training
container also includes conversion scripts and NVIDIA Triton Model Navigator.
The inference container is just the NVIDIA Triton Inference Server with the
FasterTransformer backend installed.    For Base Command Platform, the NeMo Framework
scripts repository (bcp branch) will be part of the container image. It is
recommended to create a nemo_megatron_ws_scripts_<username> workspace in your ace and
copy the nemo_megatron_launcher directory there    either from the container image or
from git clone of the above repository if you have access.    Install the NeMo Framework
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
<a id="markdown-slurm" name="slurm"></a>

The NeMo Framework codebase is included as part of the training container. To
copy it to a local directory in the cluster, it needs to be extracted from the
container. To copy the code to a directory named /path/to/local/dir the
following command can be executed. The NeMo Framework repository for 
Slurm has been verified on both Slurm-based DeepOps clusters as well as Base 
Command Manager. 


```
srun -p [partition] -N 1 --container-mounts=/path/to/local/dir:/workspace/mount_dir --container-image=[container_tag] bash -c "cp -r /opt/NeMo-Megatron-Launcher/launcher_scripts /opt/NeMo-Megatron-Launcher/auto_configurator /opt/FasterTransformer /opt/nemo-data-curator /opt/nemo-rlhf /workspace/mount_dir/"
```

Install the NeMo Framework scripts dependencies on the head node of the cluster:

```
pip install -r requirements.txt
```
You can use virtualenv to prevent polluting your head node environment for
other Python projects. If your configuration lacks pip, then you can
install pip using use [get_pip.py](https://github.com/pypa/get-pip) with just `python3`.

##### 5.1.1.2. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>

The nemo_megatron_launcher codebase is included as part of the training
container. Before starting, set up the ngc cli and configuration as described 
in the Base Command Platform User Guide. In this guide, we will mainly 
use two Base Command Platform workspaces, one for storing the training dataset,
and another for storing the results, checkpoints and logs. Therefore, start by 
creating these workspaces (e.g. `nemo_megatron_data_ws` and `nemo_megatron_results_ws`). See 
the Base Command Platform User Guide for how to create and work with Base 
Command Platform workspaces.

##### 5.1.1.3. General Configuration
<a id="markdown-general-configuration" name="general-configuration"></a>

The first parameter that must be set is the `launcher_scripts_path` parameter inside the
`conf/config.yaml` file.    This parameter must point to the absolute path where
the `nemo_megatron_launcher` repository is stored in the file system.    
Additionally, if using a Slurm based 
cluster, the config file in the subfolder of `conf/cluster/bcm.yaml` has the 
parameters to set the generic cluster related information, such as the 
`partition` or `account` parameters.

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

**Interactive**: In order to run the launcher in an interactive job or locally on a workstation, 
set `cluster_type=interactive` in `conf/config.yaml`.

**Slurm**: The `launcher_scripts_path` parameter will automatically be mounted to the
container at the same path as in the local file system. Any additional
directories that should be mounted must be specified using the
`container_mounts` parameter. If the paths contain the colon character (`:`), 
the code will assume both the source and destination paths are provided. 
Otherwise, the given paths will be mounted to the same path inside the container.
The `data_dir` parameter can also be
modified to point to where the dataset will be loaded from or saved. The 
`base_results_dir` can also be modified to point to where the results, 
checkpoints and logs will be stored. These last two parameters will be 
automatically mounted into the container. The parameters `cluster` and `cluster_type`
must be set to `bcm` for all the tasks.

**Base Command Platform**: The `launcher_scripts_path` should be set to 
/opt/NeMo-Megatron-Launcher/launcher_scripts , which is the default location where the scripts 
are located inside the container. The `data_dir` parameter can also be
modified to point to where the dataset will be loaded from or saved. The 
`base_results_dir` can also be modified to point to where the results, 
checkpoints and logs will be stored. In the case of Base Command Platform, we recommend 
that `data_dir` points to one of the workspaces, and `base_results_dir` 
points to the other. They should both be mounted in read and write (RW) 
mode. The parameter `cluster_type` must be set to `bcp` for all the tasks.

`main.py` is the main file that needs to be executed to run the data
preparation, training, conversion, fine-tuning, and evaluation pipelines. Each of these 
pipelines has a parameter in the `conf/config.yaml` file that decides whether 
to run that pipeline or not. In slurm based clusters, all of them can be set 
to `True` at the same time, and they will be executed in order. However, in Base Command Platform, 
only one of them should be set to `True` at a time.

[//]: # (##### 5.1.1.3.1. Settings for GPT Models )

[//]: # (<a id="markdown-settings-for-gpt-models" name="settings-for-gpt-models"></a>)

**Settings for GPT Models**: Default settings for GPT models are in the `config/config.yaml` file:

```yaml
stages:
  - data_preparation
  - training
  - conversion
  - evaluation
  - export
```

[//]: # (##### 4.1.1.3.2. Settings for T5 Models )

[//]: # (<a id="markdown-settings-for-t5-models" name="settings-for-t5-models"></a>)

**Settings for T5 Models**: Default settings for T5 models are in the `config/config.yaml` file:
```yaml
# default values:
cluster: bcm  # Leave it as bcm even if using bcp. It will be ignored for bcp.
data_preparation: t5/download_t5_pile
training: t5/220m
conversion: t5/convert_t5
fine_tuning: t5/squad
evaluation: t5/squad
export: t5/export_t5

stages:
  - data_preparation
  - training
  - conversion
  - fine_tuning
  - prompt_learning
  - evaluation
  - export
```

**Settings for mT5 Models**: Default settings for T5 models are in the `config/config.yaml` file:
```yaml
# default values:
cluster: bcm  # Leave it as bcm even if using bcp. It will be ignored for bcp.
data_preparation: mt5/download_mc4
training: mt5/390m
conversion: mt5/convert_mt5
fine_tuning: mt5/xquad
evaluation: mt5/xquad
export: mt5/export_mt5

stages:
  - data_preparation
  - training
  - conversion
  - fine_tuning
  - prompt_learning
  - evaluation
  - export
```

**Settings for Bert Models**: Default settings for T5 models are in the `config/config.yaml` file:
```yaml
# default values:
cluster: bcm  # Leave it as bcm even if using bcp. It will be ignored for bcp.
data_preparation: bert/download_bert_pile
training: bert/4b

stages:
  - data_preparation
  - training
```

To run these pipelines execute:

```
python3 main.py
```

The entire repository uses `hydra/omegaconf` to handle job configuration using
YAML files, so look at the documentation for those projects to learn more.

#### 5.1.2. Data Preparation
<a id="markdown-data-preparation" name="data-preparation"></a>

**The Pile**: We provide utilities to download and prepare [the Pile](https://pile.eleuther.ai/)
dataset ([mirror](https://mystic.the-eye.eu/public/AI/pile/train/)),
which is formed by 22 smaller datasets. The dataset is already blended
by using the mix described in their [paper](https://arxiv.org/pdf/2101.00027.pdf).
It is recommended to store this repository and the datasets in a file system
shared by all the nodes (gpfs) in the case of Slurm based clusters, and in a shared 
workspace with RW permissions in the case of Base Command Platform based clusters.

The Pile dataset consists of 30 shards. Downloading, extracting, and
preprocessing each file takes approximately 1 hour assuming a 30 MB/s download
speed. The data preparation can be parallelized by using up to 30 nodes. 


**mC4**: We provide utilities to download and prepare [mC4](https://www.tensorflow.org/datasets/catalog/c4)
dataset ([allen-ai version](https://huggingface.co/datasets/allenai/c4)). Multilingual C4 (mC4) 
has 101 languages and is generated from 71 [Common Crawl](https://commoncrawl.org/) dumps. 
It is recommended to store this repository and the datasets in a file system
shared by all the nodes (gpfs) in the case of Slurm based clusters, and in a shared 
workspace with RW permissions in the case of Base Command Platform based clusters.

Our scripts give user options to choose any subset of 101 languages to download and preprocess.
We curated 24 languages as our default language list. The raw size of default languages is around 5 TB.
Parallelization is enabled in downloading and preprocessing scripts. It will help to automatically
distribute and balance the work on multi-node systems and provide significant speed up.
Downloading and preprocessing the default language list takes approximately 7 hours 
assuming a 30 MB/s download speed and parallelization by using 20 nodes. The preprocessed dataset has a size 
of around 12 TB. It's recommended to use a file system with larger than 20 TB storage to prepare the data.

Currently, we don't support training with more than 25 languages, see [Known Issues].

The configuration used for data preparation for the Pile dataset or mC4 dataset must be specified in the
`conf/config.yaml` file and `data_preparation` must be included in `stages` to run it.


##### 5.1.2.1. Data Preparation for GPT Models
<a id="markdown-data-preparation-for-gpt-model" name="data-preparation-for-gpt-model"></a>
The `data_preparation` parameter in `conf/config.yaml` specifies which file to use for data preparation
configuration purposes. The default value is set to `download_gpt3_pile`, which can be
found in `conf/data_preparation/download_gpt3_pile.yaml`. It is used to download, extract,
and preprocess the Pile dataset for GPT model. The parameters can be
modified to perform the different tasks and to decide where to store the
datasets, vocab, and merge files.

To download a reduced portion of the dataset to run tests, the 
`file_numbers` parameter can be updated to download only one of the 
shards by changing “0-29” to “0” (the syntax must be a combination of
numbers separated by dashes "-" or commas ",") For example, 
`file_numbers`="0,3,5-7" will download and prepare 
files 0, 3, 5, 6, and 7.

###### 5.1.2.1.1. Slurm
<a id="markdown-41211-slurm" name="41211-slurm"></a>

First, ensure the cluster related configuration in the `conf/cluster/bcm.yaml` file is correct.
The `cluster` and `cluster_type` parameters in `conf/config.yaml` must be set to `bcm`.
Then, modify the `time_limit` or any other parameter related to the job in the `download_gpt3_pile.yaml`
file for GPT models.
The data preparation can be parallelized by using up to 30 nodes to download all 30 files in parallel.

Example:

To run only the data preparation pipeline and not the training, evaluation or
inference pipelines, set the `conf/config.yaml` file to:
```yaml
stages:
  - data_preparation
```

And then run:
```
python3 main.py
```

###### 5.1.2.1.2. Base Command Platform
<a id="markdown-41212-base-command-platform" name="41212-base-command-platform"></a>

In order to run the data preparation script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. 
By default, the data preparation script will download the data into the `data/` directory.
We recommend that the `data_dir` parameter is set to a workspace, so that the data 
is visible across multiple jobs later on. The vocab and merge files should also be 
stored to the same workspace as the dataset, for later usage. The data preparation code 
must be launched in a multi-node job. It can be parallelized to use between 2 and 30 nodes for faster preparation of the dataset.

With Base Command Platform, the 700+ GB dataset can be downloaded once and then
shared by multiple users in the same ACE by setting the permissions of the `nemo_megatron_data_ws` workspace.

To run the data preparation pipeline for GPT models, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py stages=[data_preparation] \
cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_gpt3 \
base_results_dir=/mount/results data_preparation.file_numbers='0-29' \
data_preparation.vocab_save_dir=/mount/data/bpe data_preparation.merges_save_dir=/mount/data/bpe >> /results/data_gpt3_log.txt 2>&1
```

The command above assumes you want to prepare the entire dataset (files 0-29), and you mounted the data 
workspace in `/mount/data`, and the results workspace in `/mount/results`. Stdout and stderr are redirected to the `/results/data_gpt3_log.txt` file, so it can be downloaded from NGC. 
Any other parameter can also be added to the command to modify its behavior.

###### 5.1.2.1.3. Common
<a id="markdown-41213-common" name="41213-common"></a>

Set the configuration for the data preparation job for GPT models in the YAML file:
```yaml
run:
  name: download_gpt3_pile
  results_dir: ${base_results_dir}/${.name}
  time_limit: "4:00:00"
  dependency: "singleton"
  node_array_size: 30
  array: ${..file_numbers}
  bcp_preproc_npernode: 2 # 2 should be safe to use and x2 times faster.

dataset: pile
download_the_pile: True  # Whether to download the pile dataset from the internet.
the_pile_url: "https://mystic.the-eye.eu/public/AI/pile/train/"  # Source URL to download The Pile dataset from.
file_numbers: "0-29"  # The pile dataset consists of 30 files (0-29), choose which ones to download.
preprocess_data: True  # True to preprocess the data from a jsonl file, False otherwise.
download_vocab_url: "https://huggingface.co/gpt2/resolve/main/vocab.json"  # URL to download the vocab from.
download_merges_url: "https://huggingface.co/gpt2/resolve/main/merges.txt"  # URL to download the merges from.
vocab_save_dir: ${data_dir}/bpe
merges_save_dir: ${data_dir}/bpe
tokenizer_type: GPT2BPETokenizer
rm_downloaded: True # Extract script will remove downloaded zst after extraction
rm_extracted: True # Preprocess script will remove extracted files after preproc.
```

##### 5.1.2.2. Data Preparation for T5 Models
<a id="markdown-data-preparation-for-t5-models" name="data-preparation-for-t5-models"></a>
The `data_preparation` parameter in `conf/config.yaml` specifies which file to use for data preparation
configuration purposes. The `data_preparation` parameter needs to be specified as `t5/download_t5_pile` for
preparing the Pile dataset for T5 models. The config file can be found in 
`conf/data_preparation/t5/download_t5_pile.yaml`. GPT models and T5 models use
different tokenizer and vocab files. The default parameters can be found in the
corresponding config files.

To download a reduced portion of the dataset to run tests, the 
`file_numbers` parameter can be updated to download only one of the 
shards by changing `“0-29”` to `“0”` (the syntax must be a combination of
numbers separated by dashes "-" or commas ",").
 For example, `file_numbers`=`"0,3,5-7"` will download and prepare 
files 0, 3, 5, 6, and 7.

###### 5.1.2.2.1. Slurm
<a id="markdown-41221-slurm" name="41221-slurm"></a>

First, ensure the cluster configuration settings in the `conf/cluster/bcm.yaml` file are correct.
The `cluster` and `cluster_type` parameters in `conf/config.yaml` must be set to `bcm`.
Then, modify the `time_limit` or any other parameter related to the job in the `t5/download_t5_pile.yaml`
file for T5 models.
The data preparation can be parallelized by using up to 30 nodes to download all 30 files in parallel.

Example:

To run only the data preparation pipeline and not the training, evaluation or
inference pipelines, set the `conf/config.yaml` file to:
```yaml
stages:
  - data_preparation: True
```

And then run:
```
python3 main.py
```

###### 5.1.2.2.2. Base Command Platform
<a id="markdown-41222-base-command-platform" name="41222-base-command-platform"></a>

In order to run the data preparation script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. 
By default, the data preparation script will download the data into the `data/` directory.
We recommend that the `data_dir` parameter is set to a workspace, so that the data 
is visible across multiple jobs later on. The vocab and merge files should also be 
stored to the same workspace as the dataset. The data preparation code 
must be launched in a multi-node job, and can be parallelized to use between 2 and 30 nodes, 
for faster parallel preparation of the dataset.

With Base Command Platform, the 700+ GB dataset can be downloaded once and then
shared by multiple users in the same ACE by setting the permissions of the `nemo_megatron_data_ws` workspace.

To run the data preparation pipeline for T5 models, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py data_preparation=t5/download_t5_pile \
stages=[data_preparation] \
cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_t5 \
base_results_dir=/mount/results data_preparation.file_numbers='0-29' \
data_preparation.vocab_save_dir=/mount/data/bpe >> /results/data_t5_log.txt 2>&1
```

The command above assumes you want to prepare the entire dataset (files 0-29), and you mounted the data 
workspace in `/mount/data`, and the results workspace in `/mount/results`. The stdout and stderr outputs will
also be redirected to the `/results/data_t5_log.txt` file, to be able to download the logs from NGC. 
Any other parameter can also be added to the command to modify its behavior.

###### 5.1.2.2.3. Common
<a id="markdown-41223-common" name="41223-common"></a>

Set the configuration for the data preparation job for T5 models in the YAML file:
```yaml
dataset: pile
download_the_pile: True    # Whether to download the pile dataset from the internet.
the_pile_url: "https://mystic.the-eye.eu/public/AI/pile/train/"    # Source URL to download The Pile dataset from.
file_numbers: "0-29"    # The pile dataset consists of 30 files (0-29), choose which ones to download.
preprocess_data: True    # True to preprocess the data from a jsonl file, False otherwise.
download_vocab_url: "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt"    # URL to download the vocab from.
download_merges_url: null
vocab_save_dir: ${data_dir}/bpe
merges_save_dir: ${data_dir}/bpe
tokenizer_type: BertWordPieceCase # T5 models use BertWordPieceCase tokenizer
log_dir: ${base_results_dir}/data_preparation/t5_pile_logs    # Where to save the logs
rm_downloaded: True # Extract script will remove downloaded zst after extraction
rm_extracted: True # Preprocess script will remove extracted files after preproc.
nodes: 30
time_limit: "4:00:00"
bcp_preproc_npernode: 2 # 2 should be safe to use and x2 times faster.
```


##### 5.1.2.3. Data Preparation for mT5 Models
<a id="markdown-data-preparation-for-mt5-models" name="data-preparation-for-mt5-models"></a>
The `data_preparation` parameter in `conf/config.yaml` specifies which file to use for data preparation
configuration purposes. The `data_preparation` parameter needs to be specified as `download_mc4` for
preparing the mC4 dataset for mT5 models. The config file can be found in 
`conf/data_preparation/download_mc4.yaml`. mT5 models use SentencePiece multilingual tokenzier.

To download a reduced portion of the dataset to run tests, the 
`languages` parameter can be updated to download only one of the 
languages by changing it to `lv`. The list of all 101 languages can be
found in [mC4 dataset](https://www.tensorflow.org/datasets/catalog/c4#c4multilingual).

The data preparation can be parallelized by using multiple nodes (default 20 nodes) to download and preprocess 
all language files in parallel.


###### 5.1.2.3.1. Slurm
<a id="markdown-41231-slurm" name="41231-slurm"></a>

First, ensure the cluster configuration settings in the `conf/cluster/bcm.yaml` file are correct.
The `cluster` and `cluster_type` parameters in `conf/config.yaml` must be set to `bcm`.
Then, modify the `time_limit` or any other parameter related to the job in the `download_mc4.yaml`
file for mT5 models.

Example:

To run only the data preparation pipeline and not the training, evaluation or
inference pipelines, set the `conf/config.yaml` file to:
```yaml
stages:
  - data_preparation
```

And then run:
```
python3 main.py
```

###### 5.1.2.3.2. Base Command Platform
<a id="markdown-41232-base-command-platform" name="41232-base-command-platform"></a>

In order to run the data preparation script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. 
By default, the data preparation script will download the data into the `data/` directory.
We recommend that the `data_dir` parameter is set to a workspace, so that the data 
is visible across multiple jobs later on. The tokenizer model file should also be 
stored to the same workspace as the dataset. The data preparation code 
must be launched in a multi-node job, and can be parallelized to use between 2 and 30 nodes, 
for faster parallel preparation of the dataset.

With Base Command Platform, the dataset can be downloaded once and then
shared by multiple users in the same ACE by setting the permissions of the `nemo_megatron_data_ws` workspace.

To run the data preparation pipeline for mT5 models, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py data_preparation=mt5/download_mc4 \
stages=[data_preparation] \
cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data \
base_results_dir=/mount/results data_preparation.languages=\'cs,da,de,el,en,es,fi,fr,hi,hu,it,ja,ko,lt,lv,nl,no,pl,pt,ro,ru,sk,sv,zh\' \
data_preparation.run.node_array_size=20 data_preparation.run.workers_per_node=4 >> /results/data_mt5_log.txt 2>&1
```

The command above assumes you want to prepare the mC4 dataset with 24 languages, and you mounted the data 
workspace in `/mount/data`, and the results workspace in `/mount/results`. The stdout and stderr outputs will
also be redirected to the `/results/data_mt5_log.txt` file, to be able to download the logs from NGC. The full dataset may not fit into BCP workspaces. We recommand using a smaller subset of languages (total size is 1TB, e.g. `cs,da,de,el,fr,hi`).
Any other parameter can also be added to the command to modify its behavior.

###### 5.1.2.3.3. Common
<a id="markdown-41233-common" name="41233-common"></a>

Set the configuration for the data preparation job for mT5 models in the YAML file:
```yaml
run:
  name: download_mc4
  results_dir: ${base_results_dir}/${.name}
  time_limit: "24:00:00"
  dependency: "singleton"
  node_array_size: 20
  cpus_per_node: 256
  workers_per_node: 4 # Number of workers per node in preprocessing step.
dataset: mc4
download_mc4: True  # Whether to download the mC4 dataset from the internet.
preprocess_data: True  # True to preprocess the data from a json.gz file, False otherwise.
mc4_dir: ${data_dir}/mc4 # Path to (m)C4 dataset repo.
git_lfs_dir: ${.mc4_dir}/lfs # Path to store git lfs files.
download_vocab_url: https://storage.googleapis.com/t5-data/vocabs/mc4.250000.100extra/sentencepiece.vocab # URL to download the vocab from.
download_tokenizer_url: https://storage.googleapis.com/t5-data/vocabs/mc4.250000.100extra/sentencepiece.model # URL to download tokenizer from
vocab_save_dir: ${.mc4_dir}/bpe
tokenizer_save_dir: ${.mc4_dir}/bpe
tokenizer_model: ${.tokenizer_save_dir}/mt5_tokenizer.model
languages: cs,da,de,el,en,es,fi,fr,hi,hu,it,ja,ko,lt,lv,nl,no,pl,pt,ro,ru,sk,sv,zh # language list in mC4 dataset to download and preprocess. Use `all` to download and preprocess all languages or specify language list as `en,es,ko,zh,...`
use_cleaned_english: True # whether to use cleaned version of english data
softlinks_dir: ${.mc4_dir}/softlinks # Path to languages soft links for preprocessing
preprocessed_dir: ${.mc4_dir}/preprocessed
max_split_size: 200 # (GB) Each split will be preprocessed individually. Tune this down to accommodate short wall time on clusters
download_worker_mapping: ${.mc4_dir}/download_mapping
preprocess_worker_mapping: ${.mc4_dir}/preprocess_mapping
rm_downloaded: False # Script will not remove downloaded after preprocessing
```

##### 5.1.2.4. Data Preparation for BERT Models
<a id="markdown-data-preparation-for-bert-model" name="data-preparation-for-bert-model"></a>
The `data_preparation` parameter in `conf/config.yaml` specifies which file to use for data preparation
configuration purposes. The default value is set to `download_bert_pile`, which can be
found in `conf/data_preparation/download_bert_pile.yaml`. It is used to download, extract,
and preprocess the Pile dataset for BERT model. The parameters can be
modified to perform the different tasks and to decide where to store the
datasets, vocab etc.

To download a reduced portion of the dataset to run tests, the 
`file_numbers` parameter can be updated to download only one of the 
shards by changing “0-29” to “0” (the syntax must be a combination of
numbers separated by dashes "-" or commas ",") For example, 
`file_numbers`="0,3,5-7" will download and prepare 
files 0, 3, 5, 6, and 7.

###### 5.1.2.4.1. Slurm
<a id="markdown-51241-slurm" name="51241-slurm"></a>

First, ensure the cluster related configuration in the `conf/cluster/bcm.yaml` file is correct.
The `cluster` and `cluster_type` parameters in `conf/config.yaml` must be set to `bcm`.
Then, modify the `time_limit` or any other parameter related to the job in the `download_bert_pile.yaml`
file for BERT models.
The data preparation can be parallelized by using up to 30 nodes to download all 30 files in parallel.

Example:

To run only the data preparation pipeline and not the training, evaluation or
inference pipelines, set the `conf/config.yaml` file to:
```yaml
stages:
  - data_preparation
```

And then run:
```
python3 main.py
```

###### 5.1.2.4.2. Base Command Platform
<a id="markdown-51242-base-command-platform" name="51242-base-command-platform"></a>

In order to run the data preparation script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. 
By default, the data preparation script will download the data into the `data/` directory.
We recommend that the `data_dir` parameter is set to a workspace, so that the data 
is visible across multiple jobs later on. The vocab and merge files should also be 
stored to the same workspace as the dataset, for later usage. The data preparation code 
must be launched in a multi-node job. It can be parallelized to use between 2 and 30 nodes for faster preparation of the dataset.

With Base Command Platform, the 700+ GB dataset can be downloaded once and then
shared by multiple users in the same ACE by setting appropriate permissions of the `nemo_megatron_data_ws` the workspace.

To run the data preparation pipeline for Bert models, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py stages=[data_preparation] \
cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_bert \
base_results_dir=/mount/results data_preparation.file_numbers='0-29' \
data_preparation.vocab_save_dir=/mount/data/bpe data_preparation.merges_save_dir=/mount/data/bpe >> /results/data_bert_log.txt 2>&1
```

The command above assumes you want to prepare the entire dataset (files 0-29), and you mounted the data 
workspace in `/mount/data`, and the results workspace in `/mount/results`. Stdout and stderr are redirected to the `/results/data_bert_log.txt` file, so it can be downloaded from NGC. 
Any other parameter can also be added to the command to modify its behavior.

###### 5.1.2.4.3. Common
<a id="markdown-51243-common" name="51243-common"></a>

Set the configuration for the data preparation job for BERT models in the YAML file:
```yaml
run:
  name: download_bert_pile
  results_dir: ${base_results_dir}/${.name}
  time_limit: "4:00:00"
  dependency: "singleton"
  node_array_size: 30
  array: ${..file_numbers}
  bcp_preproc_npernode: 2 # 2 should be safe to use and x2 times faster.

dataset: pile
download_the_pile: True  # Whether to download the pile dataset from the internet.
the_pile_url: "https://mystic.the-eye.eu/public/AI/pile/train/"  # Source URL to download The Pile dataset from.
file_numbers: "0-29"  # The pile dataset consists of 30 files (0-29), choose which ones to download.
preprocess_data: True  # True to preprocess the data from a jsonl file, False otherwise.
download_vocab_url: "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt"  # URL to download the vocab from.
vocab_save_dir: ${data_dir}/bpe
tokenizer_type: BertWordPieceLowerCase
rm_downloaded: True # Extract script will remove downloaded zst after extraction
rm_extracted: True # Preprocess script will remove extracted files after preproc.
```

###### 5.1.2.4.4. LDDL
<a id="markdown-51243-LDDL" name="51243-LDDL"></a>

Language Datasets and Data Loaders (LDDL) is a utility library that minimizes the friction during dataset retrieval, preprocessing and loading for the language models.  LDDL provides dataset preprocesssing and dataloaders that allow for efficient training of Bert with dynamic sequence lengths in order to maximize training performance. LDDL currently is not installed by default in the NeMo FW container.  It can be installed with `pip install git+https://github.com/NVIDIA/lddl.git`. The directions for how to preprocess data into the LDDL binned format that can be used with NeMo can be found [here] (https://github.com/NVIDIA/LDDL#bert) for preprocessing data with binning.

With the data preprocessed in binned LDDL format the LDDL dataset can be used with the following changes to the YAML file:

```yaml
trainer:
  data:
    data_prefix: 
      - /path/to/train/LDDL/Dataset
      - /path/to/val/LDDL/Dataset
      - /path/to/test/LDDL/Dataset
    dataloader_type: LDDL

```

Note: Nemo FW currently only works with LDDL datasets that have been preprocessed with binning.


### 5.2. Training with Predefined Configurations
<a id="markdown-training-with-predefined-configurations" name="training-with-predefined-configurations"></a>

#### 5.2.1. Predefined Configurations of GPT Models
<a id="markdown-predefined-configurations-of-gpt-models" name="predefined-configurations-of-gpt-models"></a>

We provide nine configurations for several different GPT model sizes: 126M, 400M_improved, 1B_improved, 5B, 7B_improved, 20B, 
40B, 40B_improved, and 175B parameters. These configurations include carefully selected
hyperparameters, which should be used as a guideline for any custom model
configurations. All these configurations are provided in the `conf/training/gpt3/`
directory. The desired configuration can be chosen by selecting the `training` 
parameter in the `conf/config.yaml` file.
For Base Command Platform, all jobs must be launched in multi-node mode.

**126M configuration:**

The 126M model uses the bf16 data type. It can be trained in about 20 hours using 8 nodes with 8 GPUs per node. The model includes 12 transformer layers, a hidden size of 768,
and 12 attention heads. The sequence length is 2048, and the optimizer is
Distributed Adam. This model does not use any model parallelism. See the `gpt3/126m.yaml` config file for parameter details.

To train a 126M model on a Slurm cluster, modify the `conf/config.yaml` file to set:
```yaml
- training: gpt3/126m
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 126M GPT model on Base Command Platform cluster on 8 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=gpt3/126m \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_gpt3 \
base_results_dir=/mount/results training.trainer.num_nodes=\$NGC_ARRAY_SIZE \
training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.json \
training.model.tokenizer.merge_file=/mount/data/bpe/merges.txt cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively, and that the $NGC_ARRAY_SIZE will use the number of nodes selected when 
creating the job (number of replicas). 

To train with fewer or a different number of nodes, the relevant parameters 
can be adjusted either in the yaml config file or 
from the command line. More on this in [section 5.7](#57-resuming-training-from-fewer-nodes). 
For Base Command Platform, all jobs must be launched in multi-node mode.

**5B configuration:**

The 5B model uses the bf16 data type. It can be trained in about 5 days using 16 nodes with 8 GPUs per node. The model includes 24
transformer layers, a hidden size of 4096, and 32 attention heads. The
sequence length is 2048, and the optimizer is Distributed Adam. This model uses tensor
parallelism of 1. For the details on all the parameters, see the 5b.yaml
config file.

To train a 5B GPT model, modify the `conf/config.yaml` file to set:
```yaml
- training: gpt3/5b
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 5B GPT model on Base Command Platform cluster on 16 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=gpt3/5b \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_gpt3 \
base_results_dir=/mount/results training.trainer.num_nodes=\$NGC_ARRAY_SIZE \
training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.json \
training.model.tokenizer.merge_file=/mount/data/bpe/merges.txt cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively, and that the $NGC_ARRAY_SIZE will use the number of nodes selected when 
creating the job (number of replicas).


**20B configuration:**

The 20B model uses the bf16 data type. It can be trained in about 6 days using 64 nodes with 8 GPUs per node. The model includes 44
transformer layers, a hidden size of 6144, and 48 attention heads. The
sequence length is 2048, and the optimizer is Distributed Adam. This model uses tensor
parallelism of 4 and pipeline parallelism of 1. For the details on all the parameters, see the 20b.yaml
config file.

To train a 20B GPT model, modify the `conf/config.yaml` file to set:
```yaml
- training: gpt3/20b
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 20B GPT model on Base Command Platform cluster on 64 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=gpt3/20b \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_gpt3 \
base_results_dir=/mount/results training.trainer.num_nodes=\$NGC_ARRAY_SIZE \
training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.json \
training.model.tokenizer.merge_file=/mount/data/bpe/merges.txt cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively, and that the $NGC_ARRAY_SIZE will use the number of nodes selected when 
creating the job (number of replicas).

**40B configuration:**

The 40B model uses the bf16 data type. It can be trained in about 6 days using 128 nodes with 8 GPUs per node. The model includes 48
transformer layers, a hidden size of 8192, and 64 attention heads. The
sequence length is 2048, and the optimizer is Distributed Adam. This model uses tensor
parallelism of 8 and pipeline parallelism of 1. 
For the details on all the parameters, see the 40b.yaml config file.

To train a 40B GPT model, modify the `conf/config.yaml` file to set:
```yaml
- training: gpt3/40b
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 40B GPT model on Base Command Platform cluster on 128 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=gpt3/40b \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_gpt3 \
base_results_dir=/mount/results training.trainer.num_nodes=\$NGC_ARRAY_SIZE \
training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.json \
training.model.tokenizer.merge_file=/mount/data/bpe/merges.txt cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively, and that the $NGC_ARRAY_SIZE will use the number of nodes selected when 
creating the job (number of replicas).

**175B configuration:**

The 175B model uses the bf16 data type. It can be trained in about 24 days using 128 nodes with 8 GPUs per node. The model includes 96
transformer layers, a hidden size of 12288, and 96 attention heads. The
sequence length is 2048, and the optimizer is Distributed Adam. This model uses tensor
parallelism of 8 and pipeline parallelism of 16. This model uses interleaved pipeline scheduling, 
with a virtual pipeline chunk size of 6.
For the details on all the parameters, see the 175b.yaml config file.

To train a 175B GPT model, modify the `conf/config.yaml` file to set:
```yaml
- training: gpt3/175b
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 175B GPT model on Base Command Platform cluster on 128 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=gpt3/175b \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_gpt3 \
base_results_dir=/mount/results training.trainer.num_nodes=\$NGC_ARRAY_SIZE \
training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.json \
training.model.tokenizer.merge_file=/mount/data/bpe/merges.txt cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively, and that the $NGC_ARRAY_SIZE will use the number of nodes selected when 
creating the job (number of replicas).


**FP8 with Transformer Engine**
Transformer Engine (TE) is a library for accelerating Transformer-based models on **NVIDIA Hopper GPUs**. It enables using 8-bit floating point (FP8) precision to provide better performance with lower memory utilization in both training and inference. NVIDIA open-sourced TE on [github](https://github.com/NVIDIA/TransformerEngine).

In NeMo Framework, you can now use `fp8` to pre-train GPT models. For example, if you want to turn on `fp8` to pre-train a 
GPT3 5B model, you can modify `gpt3/5b` training config inside `conf/training/gpt3/5b.yaml` file as following. To run a job with fp8, please set `transformer_engine=True` and `fp8=True`. Other fp8-associated knobs are set accordingly in the baseline pre-training scripts, which are ignored in bf16 training.
```yaml
  ## Transformer Engine
  transformer_engine: True # turn on Transformer Engine
  fp8: True # enables fp8 in TransformerLayer forward
  fp8_e4m3: False # sets fp8_format = recipe.Format.E4M3
  fp8_hybrid: True # sets fp8_format = recipe.Format.HYBRID
  fp8_margin: 0 # scaling margin
  fp8_interval: 1 # scaling update interval
  fp8_amax_history_len: 1024 # Number of steps for which amax history is recorded per tensor
  fp8_amax_compute_algo: max # 'most_recent' or 'max'. Algorithm for computing amax from history
  use_emha: False
```
We observed similar convergence behavior but significant speed-up comparing `fp8` and `bf16` precision.

#### 5.2.2. Predefined Configurations of T5 Models
<a id="markdown-predefined-configurations-of-t5-models" name="predefined-configurations-of-t5-models"></a>

We provide configuration files for five T5 model sizes: 220M,
3B, 11B, 23B, and 41B parameters. These configurations include carefully selected
hyperparameters, which should be used as guidelines for any custom model
configurations. The configuration files are provided in the `conf/training/t5`
directory. The desired configuration can be chosen by selecting the training
 parameter in the `conf/config.yaml` file.
For Base Command Platform, all jobs must be launched in multi-node mode.

**220M configuration:**

The 220M model uses the bf16 data type. It can be trained in about 3.5 days using 4 nodes with 8 GPUs per node. 
The model includes 12 transformer layers, a hidden size of 768, a feedforward network size of 2048,
and 12 attention heads with GeGLU activation function. The sequence length is 512, and the optimizer is
Distributed Adam. This model does not use any model parallelism. See the `t5/220m.yaml` config file for parameter details.

To train a 220M model on a Slurm cluster, modify the `conf/config.yaml` file to set:
```yaml
training: t5/220m
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 220M model on Base Command Platform cluster on 4 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=t5/220m \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_t5 \
base_results_dir=/mount/results training.trainer.num_nodes=\$NGC_ARRAY_SIZE \
training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.txt cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively. `$NGC_ARRAY_SIZE` is automatically set to the number of nodes that will be used when creating the job (number of replicas). 

To train with a different number of nodes, the relevant parameters 
(e.g. `micro_batch_size`) can be adjusted either in the appropriate yaml config file or 
from the command line. More on this in [section 5.7](#57-resuming-training-from-fewer-nodes). 
For Base Command Platform, all jobs must be launched in multi-node mode.

**3B configuration:**

The 3B model uses the bf16 data type. It can be trained in about 7.5 days using 20 nodes with 8 GPUs per node. The model includes 24
transformer layers, a hidden size of 2048, a feedforward network size of 5120, and 32 attention heads  with GeGLU activation function. The
sequence length is 512, and the optimizer is Distributed Adam. 
For the details on all the parameters, see the `t5/3b.yaml` config file.

To train a 3B model, modify the `conf/config.yaml` file to set:
```yaml
training: t5/3b
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 3B model on Base Command Platform cluster on 20 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=t5/3b \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_t5 \
base_results_dir=/mount/results training.trainer.num_nodes=\$NGC_ARRAY_SIZE \
training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.txt cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively. `$NGC_ARRAY_SIZE` is automatically set to the number of nodes that will be used when creating the job (number of replicas).



**11B configuration:**

The 11B model uses the bf16 data type. It can be trained in about 26.5 days using 20 nodes with 8 GPUs per node. The model includes 24
transformer layers, a hidden size of 4096, a feedforward network size of 10240, and 64 attention heads  with GeGLU activation function. The
sequence length is 512, and the optimizer is Distributed Adam. This model uses tensor
parallelism of 4. For the details on all the parameters, see the `t5/11b.yaml`
config file.

To train a 11B model, modify the `conf/config.yaml` file to set:
```yaml
training: t5/11b
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 11B model on Base Command Platform cluster on 20 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=t5/11b \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_t5 \
base_results_dir=/mount/results training.trainer.num_nodes=\$NGC_ARRAY_SIZE \
training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.txt cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively. `$NGC_ARRAY_SIZE` is automatically set to the number of nodes that will be used when creating the job (number of replicas).



**23B configuration:**

The 23B model uses the bf16 data type. It can be trained in about 36 days using 40 nodes with 8 GPUs per node. The model includes 36
transformer layers, a hidden size of 5120, a feedforward network size of 10880, and 64 attention heads with GeGLU activation function. The
sequence length is 512, and the optimizer is Distributed Adam. This model uses tensor
parallelism of 4 and pipeline parallelism of 2. For the details on all the parameters, see the `t5/23b.yaml`
config file.

To train a 23B model, modify the `conf/config.yaml` file to set:
```yaml
training: t5/23b
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 23B model on Base Command Platform cluster on 40 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=t5/23b \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_t5 \
base_results_dir=/mount/results training.trainer.num_nodes=\$NGC_ARRAY_SIZE \
training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.txt cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively. `$NGC_ARRAY_SIZE` is automatically set to the number of nodes that will be used when creating the job (number of replicas).


**41B configuration:**

The 41B model uses the bf16 data type. It can be trained in about 60 days using 40 nodes with 8 GPUs per node. The model includes 36
transformer layers, a hidden size of 6144, a feedforward network size of 10880, and 96 attention heads with GeGLU activation function. The
sequence length is 512, and the optimizer is Distributed Adam. This model uses tensor
parallelism of 4 and pipeline parallelism of 2. For the details on all the parameters, see the `t5/23b.yaml`
config file.

To train a 41B model, modify the `conf/config.yaml` file to set:
```yaml
training: t5/41b
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 41B model on Base Command Platform cluster on 40 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=t5/41b \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_t5 \
base_results_dir=/mount/results training.trainer.num_nodes=\$NGC_ARRAY_SIZE \
training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.txt cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively. `$NGC_ARRAY_SIZE` is automatically set to the number of nodes that will be used when creating the job (number of replicas).



#### 5.2.3. Predefined Configurations of mT5 Models
<a id="markdown-predefined-configurations-of-mt5-models" name="predefined-configurations-of-mt5-models"></a>

We provide configuration files for three mT5 model sizes: 170M, 390M, and
3B parameters. These configurations include carefully selected
hyperparameters, which should be used as guidelines for any custom model
configurations. The configuration files are provided in the `conf/training/mt5`
directory. The desired configuration can be chosen by selecting the training
 parameter in the `conf/config.yaml` file.
For Base Command Platform, all jobs must be launched in multi-node mode.

**170M configuration:**

The 170M model uses the bf16 data type. It can be trained in about 4 days using 4 nodes with 8 GPUs per node. 
The model includes 8 transformer layers, a hidden size of 512, a feedforward network size of 1024,
and 6 attention heads with GeGLU activation function. The sequence length is 512, and the optimizer is Distributed
Adam. This model does not use any model parallelism. See the `mt5/170m.yaml` config file for parameter details.

To train a 170M model on a Slurm cluster, modify the `conf/config.yaml` file to set:
```yaml
training: mt5/170m
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 170M model on Base Command Platform cluster on 4 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=mt5/170m \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data base_results_dir=/mount/results \
training.trainer.num_nodes=\$NGC_ARRAY_SIZE cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively. `$NGC_ARRAY_SIZE` is automatically set to the number of nodes that will be used when creating the job (number of replicas). 

To train with a different number of nodes, the relevant parameters 
(e.g. `micro_batch_size`) can be adjusted either in the appropriate yaml config file or 
from the command line. More on this in [section 5.7](#57-resuming-training-from-fewer-nodes). 
For Base Command Platform, all jobs must be launched in multi-node mode.



**390M configuration:**

The 390M model uses the bf16 data type. It can be trained in about 4 days using 8 nodes with 8 GPUs per node. 
The model includes 8 transformer layers, a hidden size of 512, a feedforward network size of 2048,
and 12 attention heads with GeGLU activation function. The sequence length is 512, and the optimizer is Distributed 
Adam. This model does not use any model parallelism. See the `mt5/390m.yaml` config file for parameter details.

To train a 390M model on a Slurm cluster, modify the `conf/config.yaml` file to set:
```yaml
training: mt5/390m
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 390M model on Base Command Platform cluster on 8 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=mt5/390m \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data base_results_dir=/mount/results \
training.trainer.num_nodes=\$NGC_ARRAY_SIZE cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively. `$NGC_ARRAY_SIZE` is automatically set to the number of nodes that will be used when creating the job (number of replicas). 



**3B configuration:**

The 3B model uses the bf16 data type. It can be trained in about 14 days using 20 nodes with 8 GPUs per node. The model includes 24
transformer layers, a hidden size of 2048, a feedforward network size of 5120, and 32 attention heads with GeGLU activation function. The
sequence length is 512, and the optimizer is Distributed Adam. This model uses tensor
parallelism of 2. For the details on all the parameters, see the `mt5/3b.yaml`
config file.

To train a 3B model, modify the `conf/config.yaml` file to set:
```yaml
training: mt5/3b
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 3B model on Base Command Platform cluster on 20 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=mt5/3b \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data base_results_dir=/mount/results \
training.trainer.num_nodes=\$NGC_ARRAY_SIZE cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively. `$NGC_ARRAY_SIZE` is automatically set to the number of nodes that will be used when creating the job (number of replicas).



#### 5.2.4. Training Logs with TensorBoard and Weights and Biases
<a id="markdown-training-with-tb-wandb" name="training-with-tb-wandb"></a>
The training code can log the model and system related metrics to both TensorBoard and 
Weights & Biases (W&B). The local files will be stored in the directory specified in the 
`training.exp_manager.explicit_log_dir` parameter. TensorBoard logs are saved by default.

However, W&B needs the API key to be specified to work properly. To upload the logs to W&B, 
the user must first store the W&B API key to a file (on the first line of the file), and 
select the path to the file that contains the key using the `wandb_api_key_file` parameter. 
For Base Command Platform, this file can be stored in a dataset or workspace mounted to the job.
To enable the logging of the training metrics to W&B, the following training parameters must be set:
```yaml
exp_manager:
        create_wandb_logger: True
        wandb_logger_kwargs:
            project: [W&B project name]
            name: [W&B run name]
```

The logs show the reduced_train_loss, val_loss, train_step_timing (which is the best way 
to measure the time it takes to finish each global step), and other relevant metrics.

#### 5.2.5. Predefined Configurations of BERT Models
<a id="markdown-predefined-configurations-of-bert-models" name="predefined-configurations-of-bert-models"></a>

We provide configuration files for four BERT model sizes: 110M, 4B, 20B, 
and 100B parameters. These configurations include carefully selected
hyperparameters, which should be used as guidelines for any custom model
configurations. The configuration files are provided in the `conf/training/bert`
directory. The desired configuration can be chosen by selecting the training
 parameter in the `conf/config.yaml` file.
For Base Command Platform, all jobs must be launched in multi-node mode.

**110M configuration:**

The 110M model uses the bf16 data type. The model includes 12 transformer layers, a hidden size of 768, 
a feedforward network size of 3072 and 12 attention heads with GeGLU activation function. The sequence length is 512,
and the optimizer is Distributed Adam. This model does not use any model parallelism. See the `bert/110m.yaml` config file for parameter details.

To train a 110M model on a Slurm cluster, modify the `conf/config.yaml` file to set:
```yaml
training: bert/110m
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 110M model on Base Command Platform cluster on 4 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=bert/110m \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_t5 \
base_results_dir=/mount/results training.trainer.num_nodes=\$NGC_ARRAY_SIZE \
training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.txt cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively. `$NGC_ARRAY_SIZE` is automatically set to the number of nodes that will be used when creating the job (number of replicas). 

To train with a different number of nodes, the relevant parameters 
(e.g. `micro_batch_size`) can be adjusted either in the appropriate yaml config file or 
from the command line. More on this in [section 5.7](#57-resuming-training-from-fewer-nodes). 
For Base Command Platform, all jobs must be launched in multi-node mode.

**4B configuration:**

The 4B model uses the bf16 data type. The model includes 48 transformer layers, a hidden size of 2560, 
a feedforward network size of 10240, and 40 attention heads  with GeGLU activation function. The
sequence length is 512, and the optimizer is Distributed Adam. For the details on all the parameters, see the `bert/4b.yaml`
config file.

To train a 4B model, modify the `conf/config.yaml` file to set:
```yaml
training: bert/4b
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 4B model on Base Command Platform cluster on 20 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=bert/4b \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_t5 \
base_results_dir=/mount/results training.trainer.num_nodes=\$NGC_ARRAY_SIZE \
training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.txt cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively. `$NGC_ARRAY_SIZE` is automatically set to the number of nodes that will be used when creating the job (number of replicas).


**20B configuration:**

The 20B model uses the bf16 data type. The model includes 48 transformer layers, a hidden size of 6144, 
a feedforward network size of 24576, and 96 attention heads  with GeGLU activation function. The
sequence length is 512, and the optimizer is Distributed Adam. For the details on all the parameters, see the `bert/20b.yaml`
config file.

To train a 20B model, modify the `conf/config.yaml` file to set:
```yaml
training: bert/20b
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 20B model on Base Command Platform cluster on 20 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=bert/20b \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_t5 \
base_results_dir=/mount/results training.trainer.num_nodes=\$NGC_ARRAY_SIZE \
training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.txt cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively. `$NGC_ARRAY_SIZE` is automatically set to the number of nodes that will be used when creating the job (number of replicas).

**100B configuration:**

The 100B model uses the bf16 data type. The model includes 96 transformer layers, a hidden size of 9216, 
a feedforward network size of 36864, and 96 attention heads  with GeGLU activation function. The
sequence length is 512, and the optimizer is Distributed Adam. For the details on all the parameters, see the `bert/100b.yaml`
config file.

To train a 100B model, modify the `conf/config.yaml` file to set:
```yaml
training: bert/100b
stages:
  - training
```

And run:
```
python3 main.py
```

To train a 100B model on Base Command Platform cluster on 20 nodes, use the command:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py training=bert/100b \
stages=[training] \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_t5 \
base_results_dir=/mount/results training.trainer.num_nodes=\$NGC_ARRAY_SIZE \
training.model.tokenizer.vocab_file=/mount/data/bpe/vocab.txt cluster_type=bcp
```
The command above assumes that the data and results workspaces are mounted in the `/mount/data` and `/mount/results` 
directories respectively. `$NGC_ARRAY_SIZE` is automatically set to the number of nodes that will be used when creating the job (number of replicas).


### 5.3. Using AutoConfigurator to Find the Optimal Configuration
<a id="markdown-using-autoconfigurator-to-find-the-optimal-configuration" name="using-autoconfigurator-to-find-the-optimal-configuration"></a>
AutoConfigurator searches for the Hyper-Parameters (HPs) that achieve the highest throughput for training and inference for
Large Language Models (LLMs) using NeMo-Megatron.

Note: The inference HP search is only available for GPT models.

#### 5.3.1. AutoConfigurator Capabilities
<a id="markdown-autoconfigurator-capabilities" name="autoconfigurator-capabilities"></a>

AutoConfigurator is intended to quickly iterate over different model configurations, 
to find the best configuration with minimal time and money spending. To achieve that, AutoConfigurator provides several different capabilities, as shown in the table below:

| Feature                              | GPT    | T5       | mT5      | Bert     |
| ------------------------------------ | -------- | -------- | -------- | -------- |
| Model Size Recommendation            | Yes      | Yes      | Yes      | Yes      |
| Base Config Generation               | Yes      | Yes      | Yes      | Yes      |
| Training HP Search                   | Yes      | Yes      | Yes      | Yes      |
| Parallel Training HP Search          | BCM Only | BCM Only | BCM Only | BCM Only |
| Inference HP Search                  | BCM Only | No       | No       | No       |
| Parallel Inference HP Search         | BCM Only | No       | No       | No       |
| Slurm Based Clusters                 | Yes      | Yes      | Yes      | Yes      |
| Base Command Platform Based Clusters | Yes      | Yes      | Yes      | Yes      |

##### 5.3.1.1. Model Size Recommendation
<a id="markdown-model-size-recommendation" name="model-size-recommendation"></a>

For users who do not know what model size they wish to train, AutoConfigurator is capable of recommending 
a model size, given the hardware and training constraints. If the number of GPUs, the TFLOPS per GPU, 
the maximum time to train, and the number of tokens to train for are known, then it tool can 
recommend a model size that can be trained with the specified hardware and time constraints.

For example, if the user has 20 NVIDIA DGX nodes available (80GB GPU memory), and wants to train a 
GPT model for a maximum of 5 days, AutoConfigurator will recommend using a 5B parameter GPT model. 


##### 5.3.1.2. Base Config Generation
<a id="markdown-base-config-generation" name="base-config-generation"></a>

If the model size is provided by the user, or after the model size is suggested, 
AutoConfigurator will generate a base configuration for the target model. This configuration will be a 
valid configuration in YAML format, which can be trained using NeMo-Megatron. However, the 
throughput optimization will happen at the next step (Training AutoConfigurator HP Search).


##### 5.3.1.3. Training AutoConfigurator HP Search
<a id="markdown-training-autoconfigurator-hp-search" name="training-autoconfigurator-hp-search"></a>

Given the input model size and the base configuration, 
AutoConfigurator will now search over four different critical Hyper-Parameters, that have great impact on the 
training throughput but do not affect model convergence: Tensor Parallelism (TP), Pipeline Parallelism (PP), 
Micro Batch Size (MBS), and Activation Checkpointing Layers (ActCkpt).

First, AutoConfigurator will use heuristics to choose good candidates for those four parameters, and generate 
the grid of candidate configurations. All the candidate configurations will be saved to the results directory, 
and will include YAML files with the corresponding config. NOTE: some of these configurations might not work, 
due to high memory usage or for other reasons. The next step will determine which configurations are valid.

Once all the candidate configurations are generated, it will use heuristics to sort the most promising 
candidate configurations. Then, it will launch the most promising candidates in parallel, where the number 
of candidates can be set by the `limit_search_runs` parameter, to perform a grid search over the four training 
parameters. This search will launch the jobs using NeMo-Megatron, and it will train each config for a maximum 
of `max_minutes_per_run` minutes and a maximum of `max_steps_per_run` training steps, whichever is reached first 
on the target cluster. During this search, the jobs will run with the number of nodes specified in the configuration 
files, using the `num_nodes` parameter. Once all the jobs have finished running, the final result will be 
summarized in a CSV file.

##### 5.3.1.4. Inference AutoConfigurator HP Search
<a id="markdown-inference-autoconfigurator-hp-search" name="inference-autoconfigurator-hp-search"></a>

AutoConfigurator can also search the best HPs for inference purposes. It will empirically measure the
throughput and latency for each given configuration in the grid search space, and return a comprehensive
table with all the numbers. It will search over three different critical HPs, which have great
impact on the inference throughput and latency: Tensor Parallelism (TP), Pipeline Parallelism (PP), and
Batch Size (BS). Technically, AutoConfigurator is also capable of searching over different input/output sequence
lengths. However, we do not recommend adding multiple different sequence lengths to the same search,
since the model that uses the shortest sequence lengths will always achieve higher throughput and lower
latency. Therefore, we recommend performing several different inference searches for different sequence
lengths. Once the search space has been defined, it will launch a job for each config in parallel, 
and measure the throughput and latency. This search will launch the jobs using NeMo-Megatron on the target cluster.
Once all the jobs have finished running, the final result will be summarized in a CSV file.

#### 5.3.2. Usage
<a id="markdown-usage" name="usage"></a>

In this section, we will explain how to run each of the stages described above. 

##### 5.3.2.1. General Configuration
<a id="markdown-general-configuration" name="general-configuration"></a>

###### 5.3.2.1.1. Slurm
<a id="markdown-slurm" name="slurm"></a>

First, our configuration setup assumes that the `/opt/NeMo-Megatron-Launcher/auto_configurator`, `/opt/NeMo-Megatron-Launcher/launcher_scripts` 
and `/opt/FasterTransformer` directories have been copied from the container to the local file system.

The first parameter that must be set is the `auto_configurator_path` parameter inside the `conf/config.yaml` 
file. This parameter must point to the absolute path where the `auto_configurator` directory is stored in  
the file system. Additionally, if using a Slurm-based cluster, the config file in the 
`conf/cluster/bcm.yaml` subfolder has the parameters to set the generic cluster related information, 
such as the `partition` or `account` parameters.

The `auto_configurator_path` parameter will automatically be mounted to the container at the same path as 
in the local file system. Any additional directories that should be mounted must be specified using the
`container_mounts` parameter. If the paths contain the colon character (`:`), the code will assume both 
the source and destination paths are provided. Otherwise, the given paths will be mounted to the same 
path inside the container.

The `launcher_scripts_path` and `fastertransformer_path` must point to the path where `launcher_scripts` and 
`FasterTransformer` directories are located in the local file system. The locations
specified in the default config should be valid if `/opt` was extracted correctly. Next, the 
`data_dir` value must point to the path where the training dataset is located. Note that the dataset 
for GPT, T5 and mT5 values will be different, so modify this parameter accordingly. Follow the data 
preparation steps to learn how to download and preprocess the datasets for each model. The dataset in 
this path does not need to be the full size dataset; only a small representative sample of the dataset 
is needed, since AutoConfigurator does not train the models to convergence. Finally, the `base_results_dir` 
parameter can be modified to point to the location where the results will be stored. See all the 
parameters for the `conf/config.yaml` file below:

```yaml
defaults:
  - _self_
  - cluster: bcm
  - search_config: gpt3/5b
  - override hydra/job_logging: stdout

run_training_hp_search: True
run_inference_hp_search: True

cluster_type: bcm  # bcm or bcp
auto_configurator_path: ???  # Path to the location of auto_configurator codebase.
launcher_scripts_path: ${auto_configurator_path}/../launcher_scripts
fastertransformer_path: ${auto_configurator_path}/../FasterTransformer
base_results_dir: ${auto_configurator_path}/results
data_dir: ${launcher_scripts_path}/data
training_container: nvcr.io/ea-bignlp/nemofw-training:23.07-py3
container_mounts:
    - null
wandb:  # Weights and Biases (W&B) logging.
  enable: False  # Whether to save logs to W&B.
  api_key_file: null # Path to the file where the w&B api key is stored. Key must be on the first line.
  project: nemo-megatron-autoconfig # Name of the W&B project to store the logs in. The name of the run will be populated automatically.
```

###### 5.3.2.1.2. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>

In Base Command Platform, the dataset, vocabulary, and merge files used for the training HP search must be available as a 
dataset and mounted accordingly. This guide assumes the dataset will be mounted to `/mount/data`. 
The results of running the AutoConfigurator will be stored in `/mount/results/auto_configurator`, so we recommend to mount a workspace 
to `/mount/results`.

The main configuration file can be found in `conf/config.yaml`. All the parameters can be overridden from the 
CLI, as we will show in the next section.


##### 5.3.2.2. Running Predefined Configs
<a id="markdown-running-predefined-configs" name="running-predefined-configs"></a>

The predefined configs we provide have been well tested, and the outputs produced by AutoConfigurator 
have been verified manually. Running one of these configs will first generate a base config file for 
the specified model size. Then, it will launch the training and inference grid search jobs. When 
all the jobs have finished, a final recommendation will be produced for both training and inference, 
which will show the optimal hyper-parameters for the given model.

The predefined configs can be found in the `conf/search_config` directory. Each YAML file shows one 
model type (GPT, T5 or mT5) and one model size (up to 175B parameters for GPT and up to 42B 
parameters for T5 and mT5). To run the desired config, we will need to modify the `search_config` 
parameter in the `conf/config.yaml` file. For example, if we wish to run a 5B GPT model, we can 
set this value to `gpt3/5b` (the .yaml ending should not be included). 

###### 5.3.2.2.1. Model Config
<a id="markdown-model-config" name="model-config"></a>

To run the `gpt3/5b` config, we need to set up the `conf/search_config/gpt3/5b.yaml` file correctly.

```yaml
train_settings:
  model_size_in_b: 5 # unit in billion parameters
  num_nodes: 16
  gpus_per_node: 8
  gpu_memory_gb: 80  # Memory per GPU, in GB. Currently 40GB and 80GB A100s supported.
  max_training_days: 5 # unit in days
  limit_search_runs: 100 # Max number of runs to be launched in parallel for grid search.
  output_top_n: 10  # The result will print the top N fastest training configs.
  max_steps_per_run: 50 # Max steps per run for the grid search.
  max_minutes_per_run: 10 # minutes per run for the grid search.
  tflops_per_gpu: 140  # Estimated tflops per GPU.
  num_tokens_in_b: 300  # Unit in billions, typically 300B for GPT3 models.
  vocab_size: 51200
  logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb  # Example base_results_dir/gpt3/126m
  tensor_parallel_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8]
  pipeline_parallel_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8, 10]
  min_model_parallel_size: auto  # auto to use our recommendation, or a value for the minimum desired parallelism
  max_model_parallel_size: auto  # auto to use our recommendation, or a value for the maximum desired parallelism
  micro_batch_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8, 16]
  act_ckpt_layers: auto  # auto to use our recommendation, or a list, such as [0, 1, 2, 3]
 
inference_settings:
  run:
    model_type: gpt3
    model_train_name: gpt3_5b
    gpus_per_node: 8
    data_type: "fp16" # fp32|fp16|bf16
    time_limit: 0:30:00
    results_dir: ${base_results_dir}/${search_config_value}_${search_config.train_settings.gpu_memory_gb}gb
    tensor_parallel_sizes: [1,2,4]
    pipeline_parallel_sizes: [1,2]
  benchmark:
    input_len: 60
    output_len: 20
    batch_sizes: [4,8,16,32,64,128,256]
    beam_width: 1
    topk: 4
    topp: 0.0
```

For the training parameters, the `model_size_in_b` parameter indicates how many billions of parameters the model should contain, and 
AutoConfigurator will provide a config and HPs for a model of that size. The `num_nodes` parameter indicates 
how many nodes AutoConfigurator should use to run each training job. The `gpus_per_node` parameter 
indicates how many GPUs are available in each 
node. To modify the behavior of the heuristics depending on whether 40GB or 80GB A100 GPUs are 
available, the `gpu_memory_gb` can be set to 40 or 80, respectively, and it will recommend 
candidate configs that are more suitable to each setting. 
The `max_training_days` parameter shows how many days this model will be trained for, when 
training to full convergence. It will be written to the final config YAML files. This parameter can 
also be used when `model_size_in_b` is set to `null`. The 
`limit_search_runs` parameter can be used to limit the number of configs that will be searched 
during the HP search stage. We recommend selecting a value between 30 and 100 for this parameter. 
AutoConfigurator will probably need to search at least 30 different configs to find the optimal one. However, 
if the computing resources are available in your cluster, we recommend increasing this parameter to a value close 
to 100. The `output_top_n` parameter can be used to configure how much details the output summary file 
will include. By default, when set to 10, it will output the top 10 configurations. The 
`max_steps_per_run` parameter indicates how many steps to train each configuration for. The 
`max_minutes_per_run` parameter indicates how long to run each configuration for, in minutes. We 
recommend using at least 20 minutes per run for the smaller models, and increasing it to over 60 
minutes for the larger models. The training run will be stopped when either `max_steps_per_run` or 
`max_minutes_per_run` is reached. The `tflops_per_gpu` parameter provides an estimate of the TFLOPs 
each GPU can achieve when training large language models with NeMo Framework. This value is only used to provide an 
estimate of how long the model will take to train to full convergence, so you can know the time to 
train before you even begin training your model. The `num_tokens_in_b` parameter indicates how many 
billions of tokens you will train your model for, when training to full convergence. It will be used 
when estimating how long it will take to train the model, to the desired number of tokens. The 
`vocab_size` parameter must show the vocabulary size that will be used during training. The `logs` 
parameter can be used to configure where the result logs will be saved. By default, this directory 
will be created inside the `base_results_dir` indicated in the `conf/config.yaml` file. Finally, 
the `tensor_parallel_sizes`, `pipeline_parallel_sizes`, `min_model_parallel_size`, `max_model_parallel_size`, 
`micro_batch_sizes`, and `act_ckpt_layers` parameters can be used to override the heuristics that choose 
the grid search space and the maximum and minimum parallelism allowed for each model. If these are left as `auto`, 
AutoConfigurator will select appropriate values. However, if you wish to override them, you can use these parameters. 
For example, if you only wish to search for configurations with Tensor Parallelism (TP) values of 1 and 2, you can set 
`tensor_parallel_sizes: [1, 2]` and leave the other parameters as `auto`.

In the inference parameters, `gpus_per_node` must be used to tell the system how many GPUs are available in each node. 
`tensor_parallel_sizes` is used to set the TP values to perform the HP search. `pipeline_parallel_sizes` is used to 
set the PP values to perform the HP search.  `batch_sizes` is used to set all the possible batch sizes for the HP 
search. `input_len` can be set to the sequence length of the input that will be passed to the model. `output_len` can 
be set to the output length that will be produced by the model. 


###### 5.3.2.2.2. Base Config Generation
<a id="markdown-base-config-generation" name="base-config-generation"></a>

Every time we call `python3 main.py`, a base configuration will be generated for the given model, 
and it will be saved to the `logs` directory indicated in your config files. The base configuration 
consists of a YAML file that can be run using the NeMo-Megatron training container. However, this 
base configuration has not yet been optimized to achieve the highest possible throughput, the 
optimization will take place in the next step (Training HP Search).


###### 5.3.2.2.3. Training AutoConfigurator HP Search
<a id="markdown-training-autoconfigurator-hp-search" name="training-autoconfigurator-hp-search"></a>


####### 5.3.2.2.3.1. Slurm
<a id="markdown-slurm" name="slurm"></a>

To run the training HP search pipeline, the parameter `run_training_hp_search` must be set to `True` 
in the `conf/config.yaml` file. The model used to search the best training HPs must be selected 
using the `search_config` parameter in `conf/config.yaml`. For example, by default, this parameter 
will be set to `gpt3/5b`, so AutoConfigurator will search the optimal training HPs for a 5B parameter GPT 
model. The configuration for this model can be found in the `conf/search_config/gpt3/5b.yaml` file. 
To configure the behavior of the HP search, the following parameters can be modified in the 
correspoinding YAML file. To run the training AutoConfigurator HP search after all the parameters are set, you should call 
`python3 main.py`.

####### 5.3.2.2.3.2. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>

To run the HP Tool in BCP, the `cluster_type` parameter must be set to `bcp`. All the parameters can be configured 
through CLI overrides. For example, to launch a training HP search for the 126m GPT model, run this command:
```
python3 /opt/NeMo-Megatron-Launcher/auto_configurator/main.py search_config=gpt3/0.126b run_inference_hp_search=False auto_configurator_path=/opt/NeMo-Megatron-Launcher/auto_configurator data_dir=/mount/data/the_pile_gpt3 base_results_dir=/mount/results/auto_configurator search_config.train_settings.num_nodes=$NGC_ARRAY_SIZE cluster_type=bcp
```

This command assumes that the dataset directory and the results directory are datasets and workspaces mounted correctly. 
The user can also override any training parameters, by overriding any parameter in the `search_config` dictionary with the 
`search_config.train_settings.*` parameter, using hydra overrides. The values that can be overridden are shown below:


```yaml
train_settings:
  model_size_in_b: 5 # unit in billion parameters
  num_nodes: 16
  gpus_per_node: 8
  gpu_memory_gb: 80  # Memory per GPU, in GB. Currently 40GB and 80GB A100s supported.
  max_training_days: 5 # unit in days
  limit_search_runs: 100 # Max number of runs to be launched in parallel for grid search.
  output_top_n: 10  # The result will print the top N fastest training configs.
  max_steps_per_run: 50 # Max steps per run for the grid search.
  max_minutes_per_run: 10 # minutes per run for the grid search.
  tflops_per_gpu: 140  # Estimated tflops per GPU.
  num_tokens_in_b: 300  # Unit in billions, typically 300B for GPT3 models.
  vocab_size: 51200
  logs: ${base_results_dir}/${search_config_value}_${.gpu_memory_gb}gb  # Example base_results_dir/gpt3/126m
  tensor_parallel_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8]
  pipeline_parallel_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8, 10]
  min_model_parallel_size: auto  # auto to use our recommendation, or a value for the minimum desired parallelism
  max_model_parallel_size: auto  # auto to use our recommendation, or a value for the maximum desired parallelism
  micro_batch_sizes: auto  # auto to use our recommendation, or a list, such as [1, 2, 4, 8, 16]
  act_ckpt_layers: auto  # auto to use our recommendation, or a list, such as [0, 1, 2, 3]
```

###### 5.3.2.2.4. Inference AutoConfigurator HP Search
<a id="markdown-inference-autoconfigurator-hp-search" name="inference-autoconfigurator-hp-search"></a>

To run the inference HP search pipeline, the parameter `run_inference_hp_search` must be set to `True`
in the `conf/config.yaml` file. The model used to search the best inference HPs must be selected
using the `search_config` parameter in `conf/config.yaml`. For example, by default, this parameter
will be set to `gpt3/5b`, so AutoConfigurator will search the optimal inference HPs for a 5B parameter GPT
model. The configuration for this model can be found in the `conf/search_config/gpt3/5b.yaml` file.
To configure the behavior of the HP search, the following parameters can be modified in the
correspoinding YAML file.

##### 5.3.2.3. Running Custom Model Size Configs
<a id="markdown-running-custom-model-size-configs" name="running-custom-model-size-configs"></a>

The HP Tool is capable of recommending a model size, based on your hardware and training time 
constraints. For instance, if you want to train a GPT model, but don't know what model size is 
appropriate, you can input the number of nodes (and GPUs per node) available in your cluster, 
the amount of time you want to spend training the model, and AutoConfigurator will recommend a model size
that can be trained in that time with your hardware. To see an example of this, you can look at 
the `conf/search_config/gpt3/unknown_size.yaml` file. In this file, the `model_size_in_b` 
parameter is set to null. This is how you can tell it to recommend a model size to you. 
For the recommendation to work correctly, the `num_nodes`, `gpus_per_node`, and `max_training_days` 
parameters must indicate the number of nodes and GPUs per node available, and how long you wish to 
train the model for. Also, AutoConfigurator needs to know the vocabulary size, number of tokens you will 
train the model for, and the estimated TFLOPS per GPU your hardware can achieve. These can be 
modified using the `vocab_size`, `num_tokens_in_b`, and `tflops_per_gpu` parameters, respectively. 
Once all these parameters are set correctly, and after selecting the `gpt3/unknown_size` as the 
config to run in the `search_config` parameter in the `conf/config.yaml` file, the training 
pipeline can be executed calling `python3 main.py`. This will produce a base configuration for 
the suggested model size. If `run_training_hp_search` or `run_inference_hp_search` are set to
`True`, it will also search for the HPs for training or inference, using the rest of the
configuration yaml file as input. AutoConfigurator will behave the same way as when using a predefined config.

##### 5.3.2.4. Interpreting the Results
<a id="markdown-interpreting-the-results" name="interpreting-the-results"></a>

When AutoConfigurator generates the base configuration for a model, it will be saved inside the directory 
specified in the `logs` parameter in your config files. By default, this will be 
`.../results/<model_name>/<model_size>_<gpu_mem_size>/`. As the default 
`search_config` value is set to `gpt3/5b` and the default `gpu_memory_gb` is set to 80, the results 
can be found in the `.../results/gpt3/5b_80gb/` directory. The base config will be 
available inside that directory, with the name `base_cfg_<model_size>.yaml`. 

If the training HP search pipeline is run, the results will be in three different directories inside 
your `logs` directory. The `candidate_configs` directory contains all the YAML files with all the 
configurations generated by the HP search. The `training_logs` directory contains all the logs of 
training each of the individual configs AutoConfigurator generated. If `limit_search_runs` was set to 30, 
then there should be 30 different directories with the logs for each model. 

Finally, after all the training runs have finished and the final run has analyzed the throughput 
of each configuration, the final model recommendation will be stored in the `final_results` 
directory. This directory will contain a log file which lists the `output_top_n` fastest configs, 
sorted from fastest to slowest. The directory will also contain a csv file with all the results 
from every config that was run with AutoConfigurator for a given model size. The results will be sorted 
from highest throughput to slowest throughput. The CSV file also includes information such as the 
samples per second achieved by each model, the time per global step, the TFLOPS per GPU achieved, 
and so on. The `final_results` directory will also contain a YAML file, which corresponds to the 
config with the lowest training time. This is the recommended model for training. 

For the inference HP search, the results can be found inside the directory specified in the
`results_dir` parameter of the YAML config file. Inside that directory, you will find:
.../inference/final_summary/final_output.csv.
This csv file will have the results of every model that was run by the AutoConfigurator HP search.

Notes: 
 - The result of the Training HP Search will vary when it is run with different numbers of nodes. 
 This is mainly caused by the new distributed optimizer, which provides higher memory savings when 
 using more nodes (i.e. higher data parallel value).

##### 5.3.2.5. Logging Runs with Weights and Biases
<a id="markdown-logging-runs-with-weights-and-biases" name="logging-runs-with-weights-and-biases"></a>

Weights and Biases (W&B) can be used to log all the training search runs. To achieve this, the 
`wandb` parameters must be modified in the `conf/config.yaml` file. First, `enable` must be set to 
`True`. Then, the `api_key_file` must be set to point to the path where the file which contains 
the W&B API key. The API key must be in the first line of that file. Finally, the `project` parameter
must have the name of the W&B project where the metrics will be stored. The name of each run does not 
need to be provided. It will be automatically generated by AutoConfigurator, using the model name, model size, 
and hyper-parameters used for each specific run.

```yaml
wandb:  # Weights and Biases (W&B) logging.
    enable: True 
    api_key_file: null
    project: nemo-megatron-autoconfig
```

### 5.4. Training with Custom Configurations
<a id="markdown-training-with-custom-configurations" name="training-with-custom-configurations"></a>

The training config files can be modified, or other files can be created to be
used for training. They should follow the same structure and guidelines as the
existing model configurations.

#### 5.4.1. Example: Changing Embedding Type for T5 Models
<a id="markdown-example-changing-embedding-time-for-t5-models" name="example-changing-embedding-time-for-t5-models"></a>

Here we show an example to change the embedding type for T5 models. Let's assume a case you want to
train a 220M T5 model. Instead of using default absolute learnable position embeddings, you 
want to use relative position embeddings.

First of all, you might want to check the training configuration file in `conf/training/(model_type)/(model_size).yaml`. 
In this case it will be `conf/training/t5/220m.yaml`. In the configuration file, you can find all the options we support.
You can find the parameters of your interests, in this case they will be
```yaml
position_embedding_type: 'learned_absolute' # Position embedding type. Options ['learned_absolute', 'relative']
relative_attention_num_buckets: 32 # Relative position number of buckets for computing the bias
relative_attention_max_distance: 128 # max_distance to keep relative distance in the attention_num_buckets.
```

For Slurm based systems, you can directly modify the configuration file in line. In this case, you can
change above three lines into
```yaml
position_embedding_type: 'relative' # Position embedding type. Options ['learned_absolute', 'relative']
relative_attention_num_buckets: 32 # Relative position number of buckets for computing the bias
relative_attention_max_distance: 128 # max_distance to keep relative distance in the attention_num_buckets.
```
and submit the training job.

For BCP, you can override the default configurations by adding argument 
`training.model.position_embedding_type='relative'` when submitting the training job. 

For more details of submitting training jobs on Slurm and BCP, please check [Section 5.6](#56-model-training). 

### 5.5. Bring Your Own Dataset
<a id="markdown-bring-your-own-dataset" name="bring-your-own-dataset"></a>
If you want to train the GPT, T5, or mT5 models on your own dataset (which is already
filtered and cleaned), you must first convert the dataset files to jsonl files.

As discussed in previous sections, the `data_preparation` parameter in `conf/config.yaml` 
specifies which file to use for data preparation
configuration purposes. The `data_preparation` parameter needs to be specified as `generic/custom_dataset` for
bringing your own dataset and `data_preparation` must be included in `stages` to run it. 
The `custom_dataset` config file can be found in `conf/data_preparation/generic/custom_dataset.yaml`.
With our scripts, you can train your own tokenizer and preprocess your own dataset into a format
that can be consumed by our training scripts. 

Custom dataset only supports SentencePiece tokenizers at the moment. You can either train 
a fresh SentencePiece tokenizer with our scripts or load existing ones.

The data preparation can be parallelized by using multiple nodes (by default 20 nodes) to preprocess 
all custom dataset files in parallel.

#### 5.5.1. Slurm
<a id="markdown-451-slurm" name="451-slurm"></a>

First, ensure the cluster related configuration in the `conf/cluster/bcm.yaml` file is correct.
The `cluster` and `cluster_type` parameters in `conf/config.yaml` must be set to bcm.
Then, modify the `time_limit` or any other parameter related to the job in the `custom_dataset.yaml`
file.
The data preparation can be parallelized by using `nodes * workers_per_node` number of workers (up to one workder for each dataset file).

Example:

To run only the data preparation pipeline and not the training, evaluation or
inference pipelines, set the `conf/config.yaml` file to:
```yaml
stages:
  - data_preparation
```

And then run:
```
python3 main.py
```

#### 5.5.2. Base Command Platform
<a id="markdown-452-base-command-platform" name="452-base-command-platform"></a>

In order to run the data preparation script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. 
By default, the data preparation script will put the preprocessed data into the `data/` directory.
We recommend that the `data_dir` parameter is set to a workspace, so that the data 
is visible across multiple jobs later on. The tokenizer model files should also be 
stored to the same workspace as the dataset, for later usage. The data preparation code 
must be launched in a multi-node job. It can be parallelized to use up to number of
nodes which is equal to the number of custom dataset files for faster preparation of the dataset.

To run the data preparation pipeline, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py stages=[data_preparation] \
cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts
data_dir=/mount/data \
base_results_dir=/mount/results data_preparation=custom_dataset \
dataprepartion.train_tokenizer_args.inp=/path/to/text/file/for/training/tokenizer \
datapreparation.raw_dataset_files=[/path/to/custom_data_files] \
>> /results/data_custom_dataset_log.txt 2>&1
```

The command above assumes you mounted the data 
workspace in `/mount/data`, and the results workspace in `/mount/results`. Stdout and stderr are redirected to the `/results/data_gpt3_log.txt` file, so it can be downloaded from NGC. 
Any other parameter can also be added to the command to modify its behavior.

#### 5.5.3. Common
<a id="markdown-453-common" name="453-common"></a>

Set the configuration for the custom data preparation job in the YAML file:
```yaml
run:
  name: custom_dataset
  results_dir: ${base_results_dir}/${.name}
  time_limit: "24:00:00"
  dependency: "singleton"
  node_array_size: 4
  cpus_per_node: 256
  workers_per_node: 4 # Number of workers per node in preprocessing step.
dataset: custom_dataset
custom_dataset_dir: ${data_dir}/custom_dataset
train_tokenizer: True # True to train a sentence piece tokenizer
train_tokenizer_args: # For all options please check: https://github.com/google/sentencepiece/blob/master/doc/options.md
   input: null # text file for training tokenizer
   input_format: "text" # text or tsv
   model_prefix: "custom_sp_tokenizer"
   model_type: "bpe" # model algorithm: unigram, bpe, word or char
   vocab_size: 8000 # Vocabulary size
   character_coverage: 0.9995 # character coverage to determine the minimum symbols
   unk_id: 1
   bos_id: 2
   eos_id: 3
   pad_id: 0
bpe_save_dir: ${.custom_dataset_dir}/bpe # Dir to save sentence piece tokenizer model and vocab files

preprocess_data: True  # True to preprocess the data from json, jsonl or json.gz files, False otherwise.
raw_dataset_files:
  - null # Each file should be input json, jsonl or json.gz file
tokenizer_model: ${.bpe_save_dir}/${data_preparation.train_tokenizer_args.model_prefix}.model # trained SentencePiece tokenizer model
preprocess_worker_mapping: ${.custom_dataset_dir}/preprocess_mapping
preprocessed_dir: ${.custom_dataset_dir}/preprocessed
```

*Note*: depending on the dataset and system, it's possible that system memory gets OOM with very large dataset shard files. The solution
to this issue is to reduce dataset shard sizes. If you see similar issue,
please consider breaking up json, jsonl or json.gz files into smaller chunks before running preprocessing. 


### 5.6. Model Training
<a id="markdown-model-training" name="model-training"></a>
We provide an easy-to-use yet powerful pipeline to perform distributed training
of both GPT, T5 and mT5 models across multiple nodes and GPUs. We also provide
well-established recipes for different sizes models, where the
throughput has been maximized, and the convergence properties of the
models have been tested and confirmed.

#### 5.6.1. GPT Training
<a id="markdown-gpt-training" name="gpt-training"></a>
The configuration used for the training pipeline must be specified in the
`conf/config.yaml` file, specifying the training parameter, specifying which file
to use for training purposes. The `training` must be included in `stages` to
run the training pipeline. The default value is set to `gpt3/5b`, which can be found
in `conf/training/gpt3/5b.yaml`. The parameters can be modified to adjust the
hyperparameters of the training runs. All supported model types and sizes can be found
in `conf/training` folder.

We support  global batch size rampup during training. It can be set by changing `rampup_batch_size` parameter under the training config. Should be a list of 3 values: `[<start_batch_size>, <batch_size_increment>, <rampup_samples>]`<br>.
Example: `rampup_batch_size=[256, 128, 50000000]`<br>.
In case of using ramp up batch size, nodes scheduler will be created. It allows the use of a smaller number of nodes for smaller batch size stages. Nodes scheduler will be created automatically according to `training.trainer.num_nodes` parameter which corresponds to the maximum number of nodes you want to use for the maximum global batch size. Please, note that ramp up batch size only works with fused_adam optimizer for now.

##### 5.6.1.1. Slurm
<a id="markdown-slurm" name="slurm"></a>

Set configuration for your Slurm cluster in the `conf/cluster/bcm.yaml` file:

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

And set the training job specific parameters in the `conf/training/(model_type)/(model_size).yaml` file, 
using the run section:
```yaml
run:
    name: gpt3_126m
    results_dir: ${base_results_dir}/${.name}
    time_limit: "1-12:00:00"
    dependency: "singleton"
```

To run only the training pipeline and not the data preparation, evaluation or
inference pipelines, set the `conf/config.yaml` file to:
```yaml
stages:
  - training
```
And then run:
```
python3 main.py
```

##### 5.6.1.2. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>

Select the cluster related configuration following the NGC documentation. 
Then, use the `python3 main.py` command to launch the job and override the 
desired parameters from the training job parameters.


#### 5.6.2. T5 Training
<a id="markdown-t5-training" name="t5-training"></a>
The configuration used for the training pipeline must be specified in the
`conf/config.yaml` file, specifying the training parameter, specifying which file
to use for training purposes. The `training` must be included in `stages` to
run the training pipeline. The `training` parameter needs to be set to `t5/(model_size)`
for T5 models. For example, one can use `t5/220m` which can be found
in `conf/training/t5/220m.yaml`. The parameters can be modified to adjust the
hyperparameters of the training runs. All supported model types and sizes can be found
in `conf/training` folder.

##### 5.6.2.1. Slurm
<a id="markdown-slurm" name="slurm"></a>

Set configuration for your Slurm cluster in the `conf/cluster/bcm.yaml` file:

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

And set the training job specific parameters in the `conf/training/(model_type)/(model_size).yaml` file, 
using the run section:
```yaml
run:
    name: t5_220m
    results_dir: ${base_results_dir}/${.name}
    time_limit: "7-00:00:00"
    dependency: "singleton"
```

To run only the training pipeline and not the data preparation, evaluation or
inference pipelines, set the `conf/config.yaml` file to:
```yaml
stages:
  - training
```
And then run:
```
python3 main.py
```

##### 5.6.2.2. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>

Select the cluster related configuration following the NGC documentation. 
Then, use the python3 main.py command to launch the job and override the 
desired parameters from the training job parameters.



#### 5.6.3. mT5 Training
<a id="markdown-mt5-training" name="mt5-training"></a>
The configuration used for the training pipeline must be specified in the
`conf/config.yaml` file, specifying the training parameter, specifying which file
to use for training purposes. The `training` must be included in `stages` to
run the training pipeline. The `training` parameter needs to be set to `t5/(model_size)`
for T5 models. For example, one can use `mt5/390m` which can be found
in `conf/training/mt5/390m.yaml`. The parameters can be modified to adjust the
hyperparameters of the training runs. All supported model types and sizes can be found
in `conf/training` folder.

##### 5.6.3.1. Slurm
<a id="markdown-slurm" name="slurm"></a>

Set configuration for your Slurm cluster in the `conf/cluster/bcm.yaml` file:

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

And set the training job specific parameters in the `conf/training/(model_type)/(model_size).yaml` file, 
using the run section:
```yaml
run:
    name: mt5_390m
    results_dir: ${base_results_dir}/${.name}
    time_limit: "7-00:00:00"
    dependency: "singleton"
```

To run only the training pipeline and not the data preparation, evaluation or
inference pipelines, set the `conf/config.yaml` file to:
```yaml
stages:
  - training
```
And then run:
```
python3 main.py
```

##### 5.6.3.2. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>

Select the cluster related configuration following the NGC documentation. 
Then, use the python3 main.py command to launch the job and override the 
desired parameters from the training job parameters.


#### 5.6.4. BERT Training
<a id="markdown-bert-training" name="bert-training"></a>
The configuration used for the training pipeline must be specified in the
`conf/config.yaml` file, specifying the training parameter, specifying which file
to use for training purposes. The `training` must be included in `stages` to
run the training pipeline. The `training` parameter needs to be set to `bert/(model_size)`
for T5 models. For example, one can use `bert/110m` which can be found
in `conf/training/bert/110m.yaml`. The parameters can be modified to adjust the
hyperparameters of the training runs. All supported model types and sizes can be found
in `conf/training` folder.

##### 5.6.4.1. Slurm
<a id="markdown-slurm" name="slurm"></a>

Set configuration for your Slurm cluster in the `conf/cluster/bcm.yaml` file:

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

And set the training job specific parameters in the `conf/training/(model_type)/(model_size).yaml` file, 
using the run section:
```yaml
run:
    name: bert_110m
    results_dir: ${base_results_dir}/${.name}
    time_limit: "7-00:00:00"
    dependency: "singleton"
```

To run only the training pipeline and not the data preparation, evaluation or
inference pipelines, set the `conf/config.yaml` file to:
```yaml
stages:
  - training
```
And then run:
```
python3 main.py
```

##### 5.6.4.2. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>

Select the cluster related configuration following the NGC documentation. 
Then, use the python3 main.py command to launch the job and override the 
desired parameters from the training job parameters.


### 5.7. Resuming Training with Different Number of Nodes
<a id="markdown-resuming-training-with-different-number-of-nodes" name="resuming-training-with-different-number-of-nodes"></a>

To be able to resume a training run with a different number of nodes, we recommend to keep
the global batch size unchanged. This ensures that each training step will be
almost identical, regardless of the number of nodes. The number of nodes selected must be 
compatible with the rest of the parameters: GBS must be a multiple of 
(MBS * num_gpus) / (tensor_parallelism * pipeline parallelism)

where MBS is the micro batch size. For instance, the default GBS for the 5B GPT
model is 1440; the MBS is 2; the number of GPUs is 20\*8 = 160; 
the `tensor_parallelism` value is set to 2; and the `pipeline_parallelism` value is set to 1.
Therefore, the GBS is set to a valid value:
```
1440 % (2 * 160) / (2 * 1) == 0
```


### 5.8. Checkpoint Conversion
<a id="markdown-checkpoint-conversion" name="checkpoint-conversion"></a>

We provide a simple tool to convert the checkpoints from `.ckpt` format to `.nemo` format, 
which will later be used for evaluation (in T5 models) and inference purposes. 

#### 5.8.1. GPT Conversion
<a id="markdown-gpt-conversion" name="gpt-conversion"></a>

The configuration used for the checkpoint conversion needs to be specified in the 
`conf/config.yaml` file, specifying the conversion parameter, which specifies the file 
to use for conversion purposes. The default value is set to `gpt3/convert_gpt3`, which can be found 
in `conf/conversion/gpt3/convert_gpt3.yaml` for GPT models. 

The `conversion` must be included in `stages` to run the conversion pipeline.

##### 5.8.1.1. Common
<a id="markdown-common" name="common"></a>
To specify the input checkpoint to be used for conversion for GPT models, use the `model` parameters
in `conf/conversion/convert_gpt3.yaml`:
```yaml
model:
    model_type: gpt # gpt or t5
    checkpoint_folder: ${conversion.run.train_dir}/results/checkpoints
    checkpoint_name: latest # latest OR name pattern of a checkpoint (e.g. megatron_gpt-*last.ckpt)
    hparams_file: ${conversion.run.train_dir}/results/hparams.yaml
    tensor_model_parallel_size: 2 # 1 for 126m, 2 for 5b, and 8 for 20b or larger models
    pipeline_model_parallel_size: 1 
    model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
    vocab_file: ${data_dir}/bpe/vocab.json
    merge_file: ${data_dir}/bpe/merges.txt
```

To specify the output location and file name of the converted `.nemo` file for GPT models, use the `run` parameters
in `conf/conversion/gpt3/convert_gpt3.yaml`:
```yaml
run:
    name: convert_${conversion.run.model_train_name}
    nodes: ${divide_ceil:${conversion.model.model_parallel_size}, 8} # 8 gpus per node
    time_limit: "2:00:00"
    ntasks_per_node: ${divide_ceil:${conversion.model.model_parallel_size}, ${.nodes}}
    convert_name: convert_nemo
    model_train_name: gpt3_5b
    train_dir: ${base_results_dir}/${.model_train_name}
    results_dir: ${.train_dir}/${.convert_name}
    output_path: ${.train_dir}/${.convert_name}
    nemo_file_name: megatron_gpt.nemo # name of nemo checkpoint; must be .nemo file
```

##### 5.8.1.2. Slurm
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

To run only the conversion pipeline and not the data preparation, training, 
evaluation or inference pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - conversion
```

then run:
```
python3 main.py
```

##### 5.8.1.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the conversion script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The conversion script must be launched in a multi-node job.

To run the conversion pipeline to convert a 126M checkpoint stored in 
`/mount/results/gpt3_126m/results/checkpoints`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py stages=[conversion] \
cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_gpt3 \
base_results_dir=/mount/results conversion.run.model_train_name=gpt3_126m conversion.model.vocab_file=/mount/data/bpe/vocab.json \
conversion.model.merge_file=/mount/data/bpe/merges.txt conversion.run.results_dir=/mount/results/gpt3_126m/convert_nemo \
conversion.model.checkpoint_folder=/mount/results/gpt3_126m/results/checkpoints conversion.model.tensor_model_parallel_size=1 \
>> /results/convert_gpt3_log.txt 2>&1
```
The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/convert_gpt3_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

#### 5.8.2. T5 Conversion
<a id="markdown-t5-conversion" name="t5-conversion"></a>

The configuration used for the checkpoint conversion needs to be specified in the 
`conf/config.yaml` file, specifying the conversion parameter, which specifies the file 
to use for conversion purposes. 
The conversion parameter needs to be set to `t5/convert_t5` for T5 models, which can be found 
in `conf/conversion/t5/convert_t5.yaml`.

The `conversion` must be included in `stages` to run the conversion pipeline.

##### 5.8.2.1. Common
<a id="markdown-common" name="common"></a>
To specify the input checkpoint to be used for conversion for T5 models, use the `model` parameters
in `conf/conversion/t5/convert_t5.yaml`:
```yaml
model:
    model_type: t5 # gpt or t5
    checkpoint_folder: ${conversion.run.train_dir}/results/checkpoints
    checkpoint_name: latest # latest OR name pattern of a checkpoint (e.g. megatron_gpt-*last.ckpt)
    hparams_file: ${conversion.run.train_dir}/results/hparams.yaml
    tensor_model_parallel_size: 1 # 1 for 220m, 2 for 3b
    pipeline_model_parallel_size: 1 
    model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
    vocab_file: ${data_dir}/bpe/vocab.txt
    merge_file: null
```

To specify the output location and file name of the converted `.nemo` file for T5 models, use the `run` parameters
in `conf/conversion/t5/convert_t5.yaml`:
```yaml
run:
    name: convert_${conversion.run.model_train_name}
    nodes: ${divide_ceil:${conversion.model.model_parallel_size}, 8} # 8 gpus per node
    time_limit: "2:00:00"
    ntasks_per_node: ${divide_ceil:${conversion.model.model_parallel_size}, ${.nodes}}
    convert_name: convert_nemo
    model_train_name: t5_220m
    train_dir: ${base_results_dir}/${.model_train_name}
    results_dir: ${.train_dir}/${.convert_name}
    output_path: ${.train_dir}/${.convert_name}
    nemo_file_name: megatron_t5.nemo # name of nemo checkpoint; must be .nemo file
```

##### 5.8.2.2. Slurm
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

To run only the conversion pipeline and not the data preparation, training, 
evaluation or inference pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - conversion
```

then run:
```
python3 main.py
```

##### 5.8.2.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the conversion script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The conversion script must be launched in a multi-node job.

To run the conversion pipeline to convert a T5 220M checkpoint stored in 
`/mount/results/t5_220m/results/checkpoints`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py conversion=convert_t5 \
stages=[conversion] \
cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_t5 \
base_results_dir=/mount/results conversion.model.vocab_file=/mount/data/bpe/vocab.txt \
conversion.run.model_train_name=t5_220m conversion.run.results_dir=/mount/results/t5_220m/results/convert_nemo \
conversion.model.checkpoint_folder=/mount/results/t5_220m/checkpoints \
conversion.model.tensor_model_parallel_size=1 conversion.model.pipeline_model_parallel_size=1 \
>> /results/convert_t5_log.txt 2>&1
```
The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/convert_t5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

#### 5.8.3. mT5 Conversion
<a id="markdown-mt5-conversion" name="mt5-conversion"></a>

The configuration used for the checkpoint conversion needs to be specified in the 
`conf/config.yaml` file, specifying the conversion parameter, which specifies the file 
to use for conversion purposes. 
The conversion parameter needs to be set to `mt5/convert_mt5` for mT5 models, which can be found 
in `conf/conversion/mt5/convert_mt5.yaml`.

The `conversion` must be included in `stages` to run the conversion pipeline.

##### 5.8.3.1. Common
<a id="markdown-common" name="common"></a>
To specify the input checkpoint to be used for conversion for mT5 models, use the `model` parameters
in `conf/conversion/mt5/convert_mt5.yaml`:
```yaml
model:
  model_type: t5 # gpt or t5, use t5 for mt5 as well
  checkpoint_folder: ${conversion.run.train_dir}/results/checkpoints
  checkpoint_name: latest # latest OR name pattern of a checkpoint (e.g. megatron_gpt-*last.ckpt)
  hparams_file: ${conversion.run.train_dir}/results/hparams.yaml
  tensor_model_parallel_size: 1 
  pipeline_model_parallel_size: 1
  model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
  vocab_file: null
  merge_file: null
  tokenizer_model: ${data_dir}/mc4/bpe/mt5_tokenizer.model
```

To specify the output location and file name of the converted `.nemo` file for mT5 models, use the `run` parameters
in `conf/conversion/convert_mt5.yaml`:
```yaml
run:
  name: convert_${conversion.run.model_train_name}
  nodes: ${divide_ceil:${conversion.model.model_parallel_size}, 8} # 8 gpus per node
  time_limit: "2:00:00"
  ntasks_per_node: ${divide_ceil:${conversion.model.model_parallel_size}, ${.nodes}}
  convert_name: convert_nemo
  model_train_name: mt5_390m
  train_dir: ${base_results_dir}/${.model_train_name}
  results_dir: ${.train_dir}/${.convert_name}
  output_path: ${.train_dir}/${.convert_name}
  nemo_file_name: megatron_mt5.nemo # name of nemo checkpoint; must be .nemo file
```

##### 5.8.3.2. Slurm
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

To run only the conversion pipeline and not the data preparation, training, 
evaluation or inference pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - conversion
```

then run:
```
python3 main.py
```

##### 5.8.3.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the conversion script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The conversion script must be launched in a multi-node job.

To run the conversion pipeline to convert a mT5 390M checkpoint stored in 
`/mount/results/mt5_390m/results/checkpoints`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py conversion=convert_mt5 \
stages=[conversion] \
cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts 
data_dir=/mount/data \
conversion.run.model_train_name=mt5_390m \
base_results_dir=/mount/results conversion.run.results_dir=/mount/results/mt5_390m/results/convert_nemo \
conversion.model.checkpoint_folder=/mount/results/mt5_390m/checkpoints \
conversion.model.tensor_model_parallel_size=1 conversion.model.pipeline_model_parallel_size=1 \
>> /results/convert_mt5_log.txt 2>&1
```
The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/convert_mt5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

### 5.9. Model Fine-tuning
<a id="markdown-model-fine_tuning" name="model-fine_tuning"></a>

We also provide an easy-to-use tool to help fine-tuning the trained checkpoints
on SQuAD for T5 and GPT models, and on XQuAD for mT5 models.

#### 5.9.1. T5 Fine-tuning
<a id="markdown-t5-fine_tuning" name="t5-fine_tuning"></a>


The configuration used for the fine-tuning needs to be specified in the
`conf/config.yaml` file, specifying the `fine_tuning` parameter, which specifies the
file to use for fine-tuning purposes. The `fine_tuning` parameter must be included in `stages` 
to run the fine-tuning pipeline. To fine-tune checkpoint on `squad` task, set
`fine_tuning` parameter to `t5/squad`, which can be found in `conf/fine_tuning/t5/squad.yaml`. The
parameters can be modified to adapt different GLUE tasks and checkpoints
in fine-tuning runs. One will need to tune the fine-tuning hyper parameters
to reach the best accuracy for a specific GLUE task. The provided hyper parameters
are only optimized for T5 220M model on `squad` task.

##### 5.9.1.1. Common
<a id="markdown-common" name="common"></a>
To specify the configuration for what tasks to run for fine_tuning, 
use the `run.task_name` parameter. 
And use all the `run` parameters to define the job specific config:
```yaml
run:
    name: ${.task_name}_${.model_train_name}
    time_limit: "04:00:00"
    dependency: "singleton"
    convert_name: convert_nemo
    model_train_name: t5_220m
    task_name: "squad"
    results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}
```

To specify which model checkpoint to load and its definition, use the `model` parameter:

```yaml
model:
    restore_from_path: ${base_results_dir}/${fine_tuning.run.model_train_name}/${fine_tuning.run.convert_name}/megatron_t5.nemo # Path to a trained T5 .nemo file
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1
```

##### 5.9.1.2. Slurm
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

To run only the evaluation pipeline and not the data preparation, training, 
conversion or inference pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - fine_tuning
```

then run:
```
python3 main.py
```

##### 5.9.1.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the fine-tuning script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The evaluation script must be launched in a multi-node job.

To run the fine-tuning pipeline to fine-tune a 220M T5 model converted checkpoint stored in 
/mount/results/t5_220m/convert_nemo, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py fine_tuning=t5/squad stages=[fine_tuning] \
 cluster_type=bcp \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data base_results_dir=/mount/results \
fine_tuning.run.model_train_name=t5_220m \
fine_tuning.model.restore_from_path=/mount/results/t5_220m/convert_nemo/results/megatron_t5.nemo \
>> /results/finetune_t5_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/finetune_t5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.



#### 5.9.2. mT5 Fine-tuning
<a id="markdown-mt5-fine_tuning" name="mt5-fine_tuning"></a>

XQuAD benchmark are supported for mT5 models.

The configuration used for the fine-tuning needs to be specified in the
`conf/config.yaml` file, specifying the `fine_tuning` parameter, which specifies the
file to use for fine-tuning purposes. The `fine_tuning` parameter must be included in `stages` 
 to run the fine-tuning pipeline. To fine-tune checkpoint on `xquad` task, set
`fine_tuning` parameter to `mt5/xquad`, which can be found in `conf/fine_tuning/mt5/xquad.yaml`.

##### 5.9.2.1. Common
<a id="markdown-common" name="common"></a>
To specify the configuration for what tasks to run for fine-tuning, 
use the `run.task_name` parameter. 
And use all the `run` parameters to define the job specific config:
```yaml
run:
  name: ${.task_name}_${.model_train_name}
  time_limit: "04:00:00"
  dependency: "singleton"
  convert_name: convert_nemo
  model_train_name: mt5_220m
  task_name: "xquad"
  results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}
```

To specify which model checkpoint to load and its definition, use the `model` parameter:

```yaml
model:
  restore_from_path: ${base_results_dir}/${fine_tuning.run.model_train_name}/${fine_tuning.run.convert_name}/megatron_mt5.nemo # Path to a trained mt5 .nemo file
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
```

##### 5.9.2.2. Slurm
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

To run only the evaluation pipeline and not the data preparation, training, 
conversion or inference pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - fine_tuning
```

then run:
```
python3 main.py
```

##### 5.9.2.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the fine-tuning script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The evaluation script must be launched in a multi-node job.

To run the fine-tuning pipeline to fine-tune a 390M mT5 model converted checkpoint stored in 
/mount/results/mt5_390m/convert_nemo, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py  fine_tuning=mt5/xquad stages=[fine_tuning] \
 cluster_type=bcp \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data base_results_dir=/mount/results \
fine_tuning.run.model_train_name=mt5_390m \
fine_tuning.model.restore_from_path=/mount/results/mt5_390m/convert_nemo/results/megatron_mt5_xquad.nemo \
>> /results/finetune_mt5_log.txt 2>&1
```

The command above assumes you mounted the data workspace in /mount/data, and the results workspace in /mount/results. 
The stdout and stderr outputs will also be redirected to the /results/finetune_mt5_log.txt file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

#### 5.9.3. GPT Supervised Fine-tuning
<a id="markdown-gpt-supervised-fine_tuning" name="gpt-supervised-fine_tuning"></a>


The configuration used for the supervised fine-tuning needs to be specified in the
`conf/config.yaml` file, specifying the `fine_tuning` parameter, which specifies the
file to use for fine-tuning purposes. The `fine_tuning` parameter must be included in `stages` 
to run the fine-tuning pipeline. To fine-tune checkpoint on `squad` task, set
`fine_tuning` parameter to `gpt3/squad`, which can be found in `conf/fine_tuning/gpt3/squad.yaml`. 
The provided hyper parameters are only optimized for GPT 126M model on `squad` task.

##### 5.9.3.1. Common
<a id="markdown-common" name="common"></a>
To specify the configuration for what tasks to run for fine_tuning, 
use the `run.task_name` parameter. 
And use all the `run` parameters to define the job specific config:
```yaml
run:
    name: ${.task_name}_${.model_train_name}
    time_limit: "04:00:00"
    dependency: "singleton"
    convert_name: convert_nemo
    model_train_name: gpt3_126m
    task_name: "squad"
    results_dir: ${base_results_dir}/${fine_tuning.run.model_train_name}/${fine_tuning.run.task_name}
```

To specify which model checkpoint to load and its definition, use the `model` parameter:

```yaml
model:
    restore_from_path: ${base_results_dir}/${fine_tuning.run.model_train_name}/${fine_tuning.run.convert_name}/megatron_gpt.nemo # Path to a trained GPT .nemo file
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1
```

##### 5.9.3.2. Slurm
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

To run only the fine-tuning pipeline and not the data preparation, training, 
conversion or inference pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - fine_tuning
```

then run:
```
python3 main.py
```

##### 5.9.3.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the fine-tuning script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The evaluation script must be launched in a multi-node job.

To run the fine-tuning pipeline to fine-tune a 126M GPT model converted checkpoint stored in 
`/mount/results/gpt3_126m/convert_nemo`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py fine_tuning=gpt3/squad stages=[fine_tuning] \
 cluster_type=bcp \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data base_results_dir=/mount/results \
fine_tuning.run.model_train_name=gpt3_126m \
fine_tuning.model.restore_from_path=/mount/results/gpt_sft/convert_nemo/results/megatron_gpt.nemo \
>> /results/finetune_gpt3_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/finetune_gpt3_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.



#### 5.9.4. Fine-tuning on Custom Tasks
<a id="markdown-fine-tuning-on-custom-tasks" name="fine-tuning-on-custom-tasks"></a>
We also support fine-tuning on custom down-stream tasks in T5, mT5 and GPT. 


##### 5.9.4.1. T5 and mT5
<a id="markdown-t5-and-mt5" name="t5-and-mt5"></a>
In order to benchmark on your own dataset, you are required to split the original dataset into two files for T5 and mT5, i.e. a txt file corresponding to the 
source (context) data, and txt file corresponding to the target data. Each line of these two files forms a fine-tuning sample.

Custom fine-tuning configuration files can be found in `conf/fine_tuning/t5/custom_task.yaml` for T5 models and `conf/fine_tuning/mt5/custom_task.yaml` for mT5 models. The essential parameters are listed below. You need
to specify the dataset paths and preferred benchmark metrics.
```yaml
  data:
    train_ds:
      src_file_name: ??? # Path to the txt file corresponding to the source data.
      tgt_file_name: ??? # Path to the txt file corresponding to the target data.

    validation_ds:
      src_file_name: ??? # Path to the txt file corresponding to the source data.
      tgt_file_name: ??? # Path to the txt file corresponding to the target data.
      metric:
        name: "exact_string_match" # Name of the evaluation metric to use.
        average: null # Average the metric over the dataset. Options: ['macro', 'micro']. Works only for 'F1', 'accuracy' etc. Refer to torchmetrics for metrics where this is supported.
        num_classes: null
```
You can follow the instructions in T5 and mT5 fine-tuning sections to submit a custom task job.


##### 5.9.4.2. GPT
<a id="markdown-gpt" name="gpt"></a>
To benchmark on your own dataset, you must supply the original data in a .jsonl format. This means providing a .jsonl file that contains the fine-tuning samples, where each sample comprises an `input` (which is represented by both context and query) and an `output` (the target answer). Each line in the file should constitute one fine-tuning sample.

```
{
  "input": "Which NFL team represented the AFC at Super Bowl 50? Super_Bowl_50 Paragraph: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.",
  "output": "Denver Broncos"
  ...
  }
```

Custom supervised fine-tuning configuration files can be found in `conf/fine_tuning/gpt3/custom_task.yaml` for GPT models. The essential parameters are listed below. You need to specify the dataset paths, preferred benchmark metrics and sampling probabilities from each training dataset when `strategy='random'`.

```yaml
  data:
    train_ds:
      # Example of how to specify paths to multiple datasets
      # file_names: 
      #   - /path/to/squad.jsonl
      #   - /path/to/mnli.jsonl
      #   - /path/to/boolq.jsonl
      # Example of how each dataset is formatted
      # {'input': 'John von Neumann\nVon Neumann made fundamental contributions .... Q: What did the math of artificial viscosity do?', 'output': 'smoothed the shock transition without sacrificing basic physics'}
      file_names: ??? # Path to a list of JSONL files corresponding to the source data.
      # Example of how to specify concat_sampling_probabilities
      # concat_sampling_probabilities:
      #   - 0.5
      #   - 0.25
      #   - 0.25
      concat_sampling_probabilities: ??? # When providing a list of datasets, this arg defines the sampling probabilities from each dataset when strategy='random'

    validation_ds:
      file_names: ??? # Path to a list of JSONL files corresponding to the source data. Data format is identical to train_ds.
      metric:
        name: "loss" # Name of the evaluation metric to use. Options: ['exact_string_match', 'loss']
        average: null # Average the metric over the dataset. Options: ['macro', 'micro']. Works only for 'F1', 'accuracy' etc. Refer to torchmetrics for metrics where this is supported.
        num_classes: null
```
You can follow the instructions in GPT supervised fine-tuning sections to submit a custom task job.


### 5.10. Model Prompt Learning
<a id="markdown-model-prompt-learning" name="model-prompt-learning"></a>


Within NeMo Framework we refer to **p-tuning** and **prompt tuning** methods collectively as prompt
learning. Both methods are parameter efficient alternatives to fine-tuning pretrained language
models. Our NeMo implementation makes it possible to use one pretrained GPT, T5 or mT5 models on many downstream
tasks without needing to tune the model's full set of parameters. It also allows for adding new tasks
to your model without overwriting or disrupting previous tasks for which the model has already been
p-tuned/prompt-tuned. Because the original model parameters are frozen and never altered by either
method, p-tuning/prompt-tuning also avoid cartographic forgetting issues often encountered when
fine-tuning models. 

Instead of selecting discrete text prompts in a manual or automated fashion, prompt tuning and p-tuning utilize virtual prompt embeddings that can be optimized via gradient decent. The only difference between prompt tuning and p-tuning within NeMo-Megatron is the architecture used to tune the soft prompt tokens during training.

- Our prompt tuning implementation is based off Lester et. al’s EMNLP 2021 paper "[The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)"
- Our p-tuning implementation is based off Liu et al's paper "[GPT Understands, Too](https://arxiv.org/abs/2103.10385)"

For more details of our implementation, please check [Prompt Learning](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/prompt_learning.html) in NeMo.


#### 5.10.1. GPT Prompt Learning
<a id="markdown-gpt-prompt-learning" name="gpt-prompt-learning"></a>

SQuAD v1.1 benchmark is supported for prompt learning. With default prompt learning config file, 
our scripts will download and preprocess original SQuAD v1.1 dataset to prompt learning dataset format.
You can also bring your own task dataset as long as it has been processed into the prompt learning dataset 
format.

The configuration used for the prompt learning needs to be defined in the
`conf/config.yaml` file by modifying the `prompt_learning` parameter, which specifies the
file to use for prompt learning purposes. The `prompt_learning` parameter must be included
in `stages` to run the prompt learning pipeline. To prompt learning on `squad` task, set
`prompt_learning` parameter to `gpt3/squad`, which can be found in `conf/prompt_learning/gpt3/squad.yaml`. It is possible to use optimizations such as sequence-parallelism from the base GPT model while prompt-learning as well. To enable this, set `model.sequence_sequence_parallel=True`.

##### 5.10.1.1. Common
<a id="markdown-common" name="common"></a>
To specify the configuration for prompt learning, 
use all the `run` parameters to define the job specific config:
```yaml
run:
  name: ${.task_name}_${.model_train_name}
  time_limit: "04:00:00"
  dependency: "singleton"
  convert_name: convert_nemo
  model_train_name: gpt3_5b
  task_name: "squad"
  results_dir: ${base_results_dir}/${.model_train_name}/prompt_learning_${.task_name}
```

To specify which language model checkpoint to load and its definition, use the `model` parameter:

```yaml
model:
  language_model_path: ${base_results_dir}/${prompt_learning.run.model_train_name}/${prompt_learning.run.convert_name}/megatron_gpt.nemo # Restore lanugage model from pre-trained .nemo checkpoint
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
```

##### 5.10.1.2. Slurm
<a id="markdown-slurm" name="slurm"></a>

Set configuration for a Slurm cluster in the `conf/cluster/bcm.yaml` file:

```yaml
partition: null
account: null
exclusive: True
gpus_per_task: 1
gpus_per_node: null
mem: 0
overcommit: False
job_name_prefix: "nemo-megatron-"
```

**Example:**

To run only the prompt learning pipeline and not the data preparation, training, 
conversion or other pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - prompt_learning
```

then run:
```
python3 main.py
```

##### 5.10.1.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the prompt learning script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The evaluation script must be launched in a multi-node job.

To run the prompt learning pipeline to prompt-learn a 5B GPT model converted checkpoint stored in 
`/mount/results/gpt3_5b/convert_nemo`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py prompt_learning=gpt3/squad \
stages=[prompt_learning] cluster_type=bcp \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data base_results_dir=/mount/results \
prompt_learning.run.model_train_name=gpt3_5b \
prompt_learning.model.language_model_path=/mount/results/gpt3_5b/convert_nemo/results/megatron_gpt.nemo \
>> /results/prompt_learning_gpt3_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/prompt_learning_gpt3_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

#### 5.10.2. T5 and mT5 Prompt Learning
<a id="markdown-t5-and-mt5-prompt-learning" name="t5-and-mt5-prompt-learning"></a>

The configuration used for the prompt learning needs to be defined in the
`conf/config.yaml` file by modifying the `prompt_learning` parameter, which specifies the
file to use for prompt learning purposes. The `prompt_learning` parameter must be included
in `stages` to run the prompt learning pipeline. To prompt learning on `squad` task, set
`prompt_learning` parameter to `t5/squad`, which can be found in `conf/prompt_learning/t5/squad.yaml` for T5 models
(or `mt5/squad`, which can be found in `conf/prompt_learning/mt5/squad.yaml` for mT5 models). 

##### 5.10.2.1. Common
<a id="markdown-common" name="common"></a>
To specify the configuration for prompt learning, 
use all the `run` parameters to define the job specific config:
```yaml
run:
  name: ${.task_name}_${.model_train_name}
  time_limit: "04:00:00"
  dependency: "singleton"
  convert_name: convert_nemo
  model_train_name: t5_220m # or mt5_390m
  task_name: "squad"
  results_dir: ${base_results_dir}/${.model_train_name}/prompt_learning_${.task_name}
```

To specify which language model checkpoint to load and its definition, use the `model` parameter:

```yaml
model:
  language_model_path: ${base_results_dir}/${prompt_learning.run.model_train_name}/${prompt_learning.run.convert_name}/megatron_t5.nemo # or megatron_mt5.nemo
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
```

##### 5.10.2.2. Slurm
<a id="markdown-slurm" name="slurm"></a>

Set configuration for a Slurm cluster in the `conf/cluster/bcm.yaml` file:

```yaml
partition: null
account: null
exclusive: True
gpus_per_task: 1
gpus_per_node: null
mem: 0
overcommit: False
job_name_prefix: "nemo-megatron-"
```

**Example:**

To run only the prompt learning pipeline and not the data preparation, training, 
conversion or other pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - prompt_learning
```

then run:
```
python3 main.py
```

##### 5.10.2.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the prompt learning script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The evaluation script must be launched in a multi-node job.

To run the prompt learning pipeline to prompt-learn a 220M T5 model converted checkpoint stored in 
`/mount/results/t5_220m/convert_nemo`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py prompt_learning=t5/squad \
stages=[prompt_learning] cluster_type=bcp \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts  data_dir=/mount/data base_results_dir=/mount/results \
prompt_learning.run.model_train_name=t5_220m \
prompt_learning.model.language_model_path=/mount/results/t5_220m/convert_nemo/results/megatron_t5.nemo \
>> /results/prompt_learning_t5_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/prompt_learning_t5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

To run the prompt learning pipeline to prompt-learn a 390M mT5 model converted checkpoint stored in 
`/mount/results/mt5_390m/convert_nemo`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py prompt_learning=mt5/squad \
stages=[prompt_learning] cluster_type=bcp \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data base_results_dir=/mount/results \
prompt_learning.run.model_train_name=mt5_390m \
prompt_learning.model.language_model_path=/mount/results/t5_220m/convert_nemo/results/megatron_mt5.nemo \
>> /results/prompt_learning_mt5_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/prompt_learning_mt5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.


### 5.11. Model Adapter Learning and IA3 Learning
<a id="markdown-model-adapter-learning" name="model-adapter-learning"></a>


NeMo Framework supports Adapter Learning and Infused Adapter by Inhibiting and Amplifying Inner Activations (IA3) learning. Both methods are parameter-efficient alternatives to fine-tuning pretrained language
models. Our NeMo implementation makes it possible to use one pretrained GPT or T5 models on many downstream
tasks without tuning the model's full set of parameters. Because the original model parameters are frozen and never altered by either
method, these also avoid cartographic forgetting issues often encountered when fine-tuning models. 

Unlike prompt-learning and p-tuning, Adapter learning and IA3 do not insert virtual prompts into the input. Adapter learning introduces feedforward layers within the core transformer architecture which are updated for specific downstream tasks. IA3 adds even fewer parameters that simply scale the hidden representations in the transformer layer, these scaling parameters can be trained for specific downstream tasks.

- Our Adapter learning implementation for GPT3 and T5 is based of "[Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf)"
- Our IA3 implementation is based of "[Few-Shot Parameter-Efficient Fine-Tuning is Better
and Cheaper than In-Context Learning](https://arxiv.org/pdf/2205.05638.pdf)". Note that the paper proposes a recipe called *t-few* which also introduces an unlikelihood loss function and a continued training procedure. Our IA3 implementation does not support these additions and only focuses on the core architectural change.


#### 5.11.1. GPT Adapter Learning and IA3 Learning
<a id="markdown-gpt-adapter-learning" name="gpt-adapter-learning"></a>

SQuAD v1.1 benchmark is supported for Adapter learning and IA3. With default adapter learning and IA3 config file, 
our scripts will download and preprocess original SQuAD v1.1 dataset to adapter learning and IA3 dataset format 
(the same format as prompt learning).
You can also bring your own task dataset as well.

The configuration used for the adapter learning needs to be defined in the
`conf/config.yaml` file by modifying the `adapter_learning` parameter, which specifies the
file to use for adapter learning purposes. The `adapter_learning` parameter must be included
in `stages` to run the adapter learning pipeline. To adapter learning on `squad` task, set
`adapter_learning` parameter to `gpt3/squad`, which can be found in `conf/adapter_learning/gpt3/squad.yaml`.

IA3 learning can be defined in the same way inside
`conf/config.yaml` file by modifying the `ia3_learning` parameter, which specifies the
file to use for IA3 learning purposes. The `ia3_learning` parameter must be included
in `stages` to run the IA3 learning pipeline. To IA3 learning on `squad` task, set
`ia3_learning` parameter to `gpt3/squad`, which can be found in `conf/ia3_learning/gpt3/squad.yaml`.

##### 5.11.1.1. Common
<a id="markdown-common" name="common"></a>
To specify the configuration for adapter learning (or IA3 learning), 
use all the `run` parameters to define the job specific config:
```yaml
run:
  name: ${.task_name}_${.model_train_name}
  time_limit: "04:00:00"
  dependency: "singleton"
  convert_name: convert_nemo
  model_train_name: gpt3_5b
  task_name: "squad"
  results_dir: ${base_results_dir}/${.model_train_name}/adapter_learning_${.task_name} # or ia3_learning
```

To specify which language model checkpoint to load and its definition, use the `model` parameter:

```yaml
model:
  language_model_path: ${base_results_dir}/${adapter_learning.run.model_train_name}/${adapter_learning.run.convert_name}/megatron_gpt.nemo # # or ia3_learning
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
```

##### 5.11.1.2. Slurm
<a id="markdown-slurm" name="slurm"></a>

Set configuration for a Slurm cluster in the `conf/cluster/bcm.yaml` file:

```yaml
partition: null
account: null
exclusive: True
gpus_per_task: 1
gpus_per_node: null
mem: 0
overcommit: False
job_name_prefix: "nemo-megatron-"
```

**Example:**

To run only the adapter learning pipeline and not the data preparation, training, 
conversion or other pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - adapter_learning # or ia3_learning
```

then run:
```
python3 main.py
```

##### 5.11.1.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the adapter learning script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The evaluation script must be launched in a multi-node job.

To run the adapter learning pipeline to adapter-learn a 5B GPT model converted checkpoint stored in 
`/mount/results/gpt3_5b/convert_nemo`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py adapter_learning=gpt3/squad \
stages=[adapter_learning] cluster_type=bcp \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts  data_dir=/mount/data base_results_dir=/mount/results \
adapter_learning.run.model_train_name=gpt3_5b \
adapter_learning.model.language_model_path=/mount/results/gpt3_5b/convert_nemo/results/megatron_gpt.nemo \
>> /results/adapter_learning_gpt3_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/adapter_learning_gpt3_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

To run the IA3 learning pipeline ro IA3-learn a 5B GPT model converted checkpoint stored in 
`/mount/results/gpt3_5b/convert_nemo`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py ia3_learning=gpt3/squad \
stages=[ia3_learning] cluster_type=bcp \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data base_results_dir=/mount/results \
ia3_learning.run.model_train_name=gpt3_5b \
ia3_learning.model.language_model_path=/mount/results/gpt3_5b/convert_nemo/results/megatron_gpt.nemo \
>> /results/ia3_learning_gpt3_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/ia3_learning_gpt3_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

#### 5.11.2. T5 Adapter Learning and IA3 Learning
<a id="markdown-t5-and-mt5-adapter-learning" name="t5-and-mt5-adapter-learning"></a>

The configuration used for the adapter learning needs to be defined in the
`conf/config.yaml` file by modifying the `adapter_learning` parameter, which specifies the
file to use for adapter learning purposes. The `adapter_learning` parameter must be included
in `stages` to run the adapter learning pipeline. To adapter learning on `squad` task, set
`adapter_learning` parameter to `t5/squad`, which can be found in `conf/adapter_learning/t5/squad.yaml` for T5 models.

IA3 learning can be defined in the same way inside
`conf/config.yaml` file by modifying the `ia3_learning` parameter, which specifies the
file to use for IA3 learning purposes. The `ia3_learning` parameter must be included
in `stages` to run the IA3 learning pipeline. To IA3 learning on `squad` task, set
`ia3_learning` parameter to `t5/squad`, which can be found in `conf/adapter_learning/t5/squad.yaml` for T5 models.

##### 5.11.2.1. Common
<a id="markdown-common" name="common"></a>
To specify the configuration for adapter learning (or IA3 learning), 
use all the `run` parameters to define the job specific config:
```yaml
run:
  name: ${.task_name}_${.model_train_name}
  time_limit: "04:00:00"
  dependency: "singleton"
  convert_name: convert_nemo
  model_train_name: t5_220m
  task_name: "squad"
  results_dir: ${base_results_dir}/${.model_train_name}/adapter_learning_${.task_name} # or ia3_learning
```

To specify which language model checkpoint to load and its definition, use the `model` parameter:

```yaml
model:
  language_model_path: ${base_results_dir}/${adapter_learning.run.model_train_name}/${adapter_learning.run.convert_name}/megatron_t5.nemo # or ia3_learning
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
```

##### 5.11.2.2. Slurm
<a id="markdown-slurm" name="slurm"></a>

Set configuration for a Slurm cluster in the `conf/cluster/bcm.yaml` file:

```yaml
partition: null
account: null
exclusive: True
gpus_per_task: 1
gpus_per_node: null
mem: 0
overcommit: False
job_name_prefix: "nemo-megatron-"
```

**Example:**

To run only the adapter learning pipeline and not the data preparation, training, 
conversion or other pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - adapter_learning # or ia3_learning
```

then run:
```
python3 main.py
```

##### 5.11.2.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the adapter learning script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The evaluation script must be launched in a multi-node job.

To run the adapter learning pipeline to adapter-learn a 220M T5 model converted checkpoint stored in 
`/mount/results/t5_220m/convert_nemo`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py adapter_learning=t5/squad \
stages=[adapter_learning] cluster_type=bcp \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data base_results_dir=/mount/results \
adapter_learning.run.model_train_name=t5_220m \
adapter_learning.model.language_model_path=/mount/results/t5_220m/convert_nemo/results/megatron_t5.nemo \
>> /results/adapter_learning_t5_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/adapter_learning_t5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

To run the IA3 learning pipeline to IA3-learn a 220M T5 model converted checkpoint stored in 
`/mount/results/t5_220m/convert_nemo`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py ia3_learning=t5/squad \
stages=[ia3_learning] cluster_type=bcp \
launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data base_results_dir=/mount/results \
ia3_learning.run.model_train_name=t5_220m \
ia3_learning.model.language_model_path=/mount/results/t5_220m/convert_nemo/results/megatron_t5.nemo \
>> /results/ia3_learning_t5_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/ia3_learning_t5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.


### 5.12 LoRA Model and Generalized PEFT Framework
<a id="markdown-peft-model" name="lora-peft-framework"></a>
Many Parameter Efficient Fine-Tuning (PEFT) models have overlapping functionalities. In order to enhance NeMo's codebase, we have worked towards unifying the implementation of all supported PEFT methods, making it more streamlined. Furthermore, we have introduced the Low-rank Adapter PEFT model for GPT-style and mT5/T5-style base models in NeMo.


#### 5.12.1 PEFT Training and Inference for GPT-style Models
The new PEFT framework is built upon the SFT models and datasets, thereby inheriting all the dataset preparation requirements from SFT. For more details, please refer to the SFT section below.

##### 5.12.1.1 PEFT Training and Inference
We offer a training and inference script in NeMo. Below is an example of how to use the training script. The `TRAIN_FILE`s (and `VALIDATION_FILE`s) follow the same format as SFT.

Take note of the `model.peft.peft_scheme` argument. You can train a LoRA, P-tuning, Adapter, or IA3 model by setting this argument to the desired PEFT method.
```bash
python3 /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
  model.restore_from_path=<BASE_GPT_MODEL> \ 
  model.data.train_ds.num_workers=0 \
  model.data.validation_ds.num_workers=0 \
  model.data.train_ds.file_names=[<TRAIN_FILE1>,<TRAIN_FILE2>,...] \ 
  model.data.train_ds.concat_sampling_probabilities=[0.3,0.2,..] \ # should sum to 1 and be of the same length as number of training files
  model.data.validation_ds.file_names=[<VALIDATION_FILE1>, <VALIDATION_FILE2>,...] \ 
  model.data.train_ds.prompt_template='{input} Answer: {output}' \
  model.peft.peft_scheme='lora'  # can be replaced with 'adapter', 'ptuning' or 'ia3'
  model.answer_only_loss=True 
```
At the end of training a '.nemo' model is generated which contains the parameters for the PEFT model.
Similarly, the PEFT framework has a single inference script as well:
```bash
python3 /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval.py \
model.restore_from_path=<BASE_GPT_MODEL> \
model.peft.restore_from_path=<PEFT_MODEL> \
model.data.test_ds.file_names=[<TEST_FILE>] \
model.data.test_ds.names=['my_test_set'] \
model.data.test_ds.tokens_to_generate=30 \
inference.greedy=True \
inference.outfile_path=<OUTPUT_FILE>
```
Additionally, NeMo has a notebook which walks through the steps (which these scripts encapsulate) to train and run inference for PEFT models: https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/lora.ipynb

##### 5.12.2 PEFT Training and Inference for mT5/T5-style Models
We offer training and inference scripts in NeMo for parameter efficient tuning of mT5/T5-style models. You can train a LoRA, P-tuning, Adapter, or IA3 model using its corresponding training and inference script. 

##### 5.12.2.1 PEFT Training and Inference
Below is an example of how to use the training scripts for adapter tuning. The `TRAIN_FILE`s (and `VALIDATION_FILE`s) follow the same format as SFT.

```bash
python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_t5_adapter_tuning.py \
    model.language_model_path=<BASE_T5_MODEL> \
    model.data.train_ds=[<TRAIN_FILE1>,<TRAIN_FILE2>,...] \
    model.data.validation_ds=[<VALIDATION_FILE1>, <VALIDATION_FILE2>,...]
```

At the end of tuning, a '.nemo' model is generated which contains the parameters for the PEFT model.
Similarly, the PEFT framework has an inference script as well:

```bash
python /data/NeMo/examples/nlp/language_modeling/tuning/megatron_t5_adapter_eval.py \
    data.test_ds=[<TEST_FILE>] \
    language_model_path=[BASE_T5_MODEL] \
    adapter_model_file=[PEFT_MODEL] \
    pred_file_path=<OUTPUT_FILE>
```

You can switch to IA3, P-tuning, or LoRA methods by using the same input arguments to a different script. Below is the table including filepaths for each PEFT method:

| PEFT Method    | Filepath   | 
| -------------- | ---------- | 
| Adapter tuning |  ```/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_t5_adapter_tuning.py```   | 
| IA3 tuning     |  ```/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_t5_ia3_tuning.py```   | 
| P-tuning       | ```/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_t5_prompt_learning.py```    | 
| LoRA tuning    | ```/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_t5_lora_tuning.py```    | 

Similarly, the inference script filepaths are provided below:

| PEFT Method    | Filepath   | 
| -------------- | ---------- | 
| Adapter tuning |  ```/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_t5_adapter_eval.py```   | 
| IA3 tuning     |  ```/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_t5_ia3_eval.py```   | 
| P-tuning       | ```/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_t5_prompt_learning_eval.py```    | 
| LoRA tuning    | ```/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_t5_lora_eval.py```    | 


### 5.13. Model Evaluation
<a id="markdown-model-evaluation" name="model-evaluation"></a>

#### 5.13.1. GPT Evaluation
<a id="markdown-gpt-evaluation" name="gpt-evaluation"></a>

We also provide a simple tool to help evaluate the trained checkpoints. You can
evaluate the capabilities of the GPT model on the following ZeroShot
downstream evaluation tasks: `lambada`, `boolq`, `race`, `piqa`, `hellaswag`, `winogrande`,
`wikitext2`, and `wikitext103`.

The model evaluation must be performed using a training checkpoint (.ckpt format), not
a converted checkpoint (`.nemo` format).

The configuration used for the evaluation needs to be specified in the
`conf/config.yaml` file, specifying the `evaluation` parameter, which specifies the
file to use for evaluation purposes. The `evaluation` parameter must be included in `stages`
 to run the evaluation pipeline. The default value is set to
`gpt3/evaluate_all`, which can be found in `conf/evaluation/gpt3/evaluate_all.yaml`. The
parameters can be modified to adapt different evaluation tasks and checkpoints
in evaluation runs. For Base Command Platform, all these parameters should be overridden from the command line.

##### 5.13.1.1. Common
<a id="markdown-common" name="common"></a>
To specify the configuration for what tasks to run for evaluation, use the `run.tasks` parameter. 
And use all the `run` parameters to define the job specific config:
```yaml
run:
    name: ${.eval_name}_${.model_train_name}
    time_limit: "4:00:00"
    nodes: ${divide_ceil:${evaluation.model.model_parallel_size}, 8} # 8 gpus per node
    ntasks_per_node: ${divide_ceil:${evaluation.model.model_parallel_size}, ${.nodes}}
    eval_name: eval_all
    model_train_name: gpt3_5b
    train_dir: ${base_results_dir}/${.model_train_name}
    tasks: all_tasks    # supported: lambada, boolq, race, piqa, hellaswag, winogrande, wikitext2, wikitext103 OR all_tasks
    results_dir: ${base_results_dir}/${.model_train_name}/${.eval_name}
```

To specify which model checkpoint to load and its definition, use the `model` parameter:

```yaml
model:
    model_type: nemo-gpt3
    checkpoint_folder: ${evaluation.run.train_dir}/results/checkpoints
    checkpoint_name: latest # latest OR name pattern of a checkpoint (e.g. megatron_gpt-*last.ckpt)
    hparams_file: ${evaluation.run.train_dir}/results/hparams.yaml
    tensor_model_parallel_size: 2 #1 for 126m, 2 for 5b, 8 for 20b
    pipeline_model_parallel_size: 1
    model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
    precision: bf16 # must match training precision - 32, 16 or bf16
    eval_batch_size: 4
    vocab_file: ${data_dir}/bpe/vocab.json
    merge_file: ${data_dir}/bpe/merges.txt
```

##### 5.13.1.2. Slurm
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

To run only the evaluation pipeline and not the data preparation, training, 
conversion or inference pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - evaluation
```

then run:
```
python3 main.py
```

##### 5.13.1.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the evaluation script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The evaluation script must be launched in a multi-node job.

To run the evaluation pipeline to evaluate a 126M GPT model checkpoint stored in 
`/mount/results/gpt3_126m/checkpoints`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py stages=[evaluation] \
 cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_gpt3 \
base_results_dir=/mount/results evaluation.model.vocab_file=/mount/data/bpe/vocab.json \
evaluation.model.merge_file=/mount/data/bpe/merges.txt evaluation.run.results_dir=/mount/results/gpt3_126m/evaluation \
evaluation.model.checkpoint_folder=/mount/results/gpt3_126m/results/checkpoints evaluation.model.eval_batch_size=16 \
evaluation.model.tensor_model_parallel_size=1 \
>> /results/eval_gpt3_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/eval_gpt3_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

##### 5.13.1.4 Interleaved Pipeline Parallelism
<a id="markdown-interleaved-pipeline-parallelism" name="interleaved-pipeline-parallelism"></a>
If your model was trained with interleaved pipeline parallelism, then the model must converted to a non-interleaved model.
In order to check if your model used interleaved, inspect the training config and verify that
`model.virtual_pipeline_model_parallel_size > 0`.

To convert the model, use the script from the NeMo Toolkit: [examples/nlp/language_modeling/megatron_change_num_partitions.py](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_change_num_partitions.py)

```
CUDA_VISIBLE_DEVICES=0 python3 -u /opt/NeMo/examples/nlp/language_modeling/megatron_change_num_partitions.py  \
  --num_gpu_per_node=1 \
  --model_extracted_dir=${RESULTS_DIR}/checkpoints \
  --target_file=${RESULTS_DIR}/checkpoints/megatron_gpt_converted.nemo \
  --ckpt_name='megatron_gpt--val_loss=2.59-step=9421-consumed_samples=2411520.0-last.ckpt' \
  --tensor_model_parallel_size=1 \
  --target_tensor_model_parallel_size=1 \
  --pipeline_model_parallel_size=4 \
  --target_pipeline_model_parallel_size=4 \
  --virtual_pipeline_model_parallel_size=3 \
  --hparams_file=${RESULTS_DIR}/hparams.yaml \
  --precision=bf16 "
```

Note the conversion script should only be run with a single GPU.

The output of the conversion script is a `.nemo` file. This file should be added to your evaluation config:

```
evaluation.model.nemo_model=/path/to/converted.nemo \
evaluation.model.checkpoint_folder=null \
evaluation.model.checkpoint_name=null \
evaluation.model.hparams_file=null \
```


#### 5.13.2. T5 Evaluation
<a id="markdown-t5-evaluation" name="gpt-evaluation"></a>


On top of fine-tuned checkpoint, you can run the evaluation scripts to
evaluate the capabilities of the finetuned T5 model on SQuAD.
The model evaluation must be performed with a fine-tuned checkpoint in `.nemo` format.

The configuration used for the evaluation needs to be specified in the
`conf/config.yaml` file, specifying the `evaluation` parameter, which specifies the
file to use for evaluation purposes. The `evaluation` parameter must be included in `stages`
 to run the evaluation pipeline. The default value is set to
`t5/squad`, which can be found in `conf/evaluation/t5/squad.yaml`. The
parameters can be modified to adapt different evaluation tasks and checkpoints
in evaluation runs. For Base Command Platform, all these parameters should be overridden from the command line.


##### 5.13.2.1. Common
<a id="markdown-common" name="common"></a>
To specify the configuration for what tasks to run for evaluation, use the `run.task_name` parameter. 
And use all the `run` parameters to define the job specific config: 
```yaml
run:
    name: eval_${.task_name}_${.model_train_name}
    time_limit: "04:00:00"
    dependency: "singleton"
    model_train_name: t5_220m
    task_name: "squad"
    fine_tuning_results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}
    results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}_eval
```

To specify which fine-tuned checkpoint to load and its definition, use the `model` parameter:

```yaml
model:
    restore_from_path: ${evaluation.run.fine_tuning_results_dir}/checkpoints/megatron_t5_glue.nemo # Path to a finetuned T5 .nemo file
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1
```

##### 5.13.2.2. Slurm
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

To run only the evaluation pipeline and not the data preparation, training, 
conversion or inference pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - evaluation
```

then run:
```
python3 main.py
```

##### 5.13.2.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the evaluation script on Base Command Platform for T5 models, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The evaluation script must be launched in a multi-node job.

To run the evaluation pipeline to evaluate a 220M T5 model which has been fine-tuned
on `squad` task and checkpoint stored in `/mount/results/t5_220m/squad/results/checkpoints`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py evaluation=t5/squad \
stages=[evaluation] \
 cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts  data_dir=/mount/data \
base_results_dir=/mount/results evaluation.run.model_train_name=t5_220m \
evaluation.model.restore_from_path=/mount/results/t5_220m/squad/results/checkpoints/megatron_t5_glue.nemo \
>> /results/eval_t5_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/eval_t5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.


#### 5.13.3. mT5 Evaluation
<a id="markdown-mt5-evaluation" name="mt5-evaluation"></a>


On top of fine-tuned checkpoint, you can run the evaluation scripts to
evaluate the capabilities of the finetuned mT5 model on the following 
downstream evaluation tasks: `xquad`. Usually the task of fine-tuning and evaluation
should be the same.

The model evaluation must be performed with a fine-tuned checkpoint in `.nemo` format.

The configuration used for the evaluation needs to be specified in the
`conf/config.yaml` file, specifying the `evaluation` parameter, which specifies the
file to use for evaluation purposes. The `evaluation` parameter must be included in `stages`
 to run the evaluation pipeline. The default value is set to
`mt5/xquad`, which can be found in `conf/evaluation/mt5/xquad.yaml`. The
parameters can be modified to adapt different evaluation tasks and checkpoints
in evaluation runs. For Base Command Platform, all these parameters should be overridden from the command line.


##### 5.13.3.1. Common
<a id="markdown-common" name="common"></a>
To specify the configuration for what tasks to run for evaluation, use the `run.task_name` parameter. 
And use all the `run` parameters to define the job specific config: 
```yaml
run:
    name: eval_${.task_name}_${.model_train_name}
    time_limit: "04:00:00"
    dependency: "singleton"
    model_train_name: mt5_390m
    task_name: "xquad"
    fine_tuning_results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}
    results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}_eval
```

To specify which fine-tuned checkpoint to load and its definition, use the `model` parameter:

```yaml
model:
    restore_from_path: ${evaluation.run.fine_tuning_results_dir}/checkpoints/megatron_mt5_xquad.nemo # Path to a finetuned T5 .nemo file
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1
```

##### 5.13.3.2. Slurm
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

To run only the evaluation pipeline and not the data preparation, training, 
conversion or inference pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - evaluation
```

then run:
```
python3 main.py
```

##### 5.13.3.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the evaluation script on Base Command Platform for mT5 models, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The evaluation script must be launched in a multi-node job.

To run the evaluation pipeline to evaluate a 390M mT5 model which has been fine-tuned
on `xquad` task and checkpoint stored in `/mount/results/mt5_390m/xquad/results/checkpoints`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py evaluation=mt5/xquad \
stages=[evaluation] cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data \
base_results_dir=/mount/results evaluation.run.model_train_name=mt5_390m \
evaluation.model.restore_from_path=/mount/results/mt5_390m/xquad/results/checkpoints/megatron_mt5_xquad.nemo \
>> /results/eval_mt5_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/eval_mt5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.


#### 5.13.4. Prompt Learned GPT Evaluation
<a id="markdown-prompt-learned-gpt-evaluation" name="prompt-learned-gpt-evaluation"></a>

We also provide a simple tool to help evaluate the prompt learned GPT checkpoints. You can
evaluate the capabilities of the prompt learned GPT model on a customized prompt learning test dataset.
We provide an example to evaluate our checkpoint, which went through prompt learning on SQuAD v1.1,
on the SQuAD v1.1 test dataset created in prompt learning step.

The configuration used for the evaluation needs to be defined in the
`conf/config.yaml` file by modifying the `evaluation` parameter, which specifies the
file to be used for evaluation purposes. The `evaluation` parameter must be included in `stages`
 to run the evaluation pipeline. The value should be set to
`prompt_gpt3/squad.yaml`, which can be found in `conf/evaluation/prompt_gpt3/squad.yaml`. The
parameters can be modified to adapt different evaluation tasks and checkpoints
in evaluation runs. For Base Command Platform, all these parameters should be overridden from the command line.

##### 5.13.4.1. Common
<a id="markdown-common" name="common"></a>
To specify the configuration, use all the `run` parameters to define the job specific config. (
`run.tasks` has to be set to `prompt` to run evaluation on prompt learning test tasks):
```yaml
run:
  name: ${.eval_name}_${.model_train_name}
  time_limit: "4:00:00"
  nodes: ${divide_ceil:${evaluation.model.model_parallel_size}, 8} # 8 gpus per node
  ntasks_per_node: ${divide_ceil:${evaluation.model.model_parallel_size}, ${.nodes}}
  eval_name: eval_prompt_squad
  model_train_name: gpt3_5b
  tasks: "prompt" # general prompt task
  prompt_learn_dir: ${base_results_dir}/${.model_train_name}/prompt_learning_squad # assume prompt learning was on squad task
  results_dir: ${base_results_dir}/${.model_train_name}/${.eval_name}
```

To specify which model checkpoint to load and which prompt learning test dataset to evaluate, 
use the `model` parameter:

```yaml
model:
  model_type: nemo-gpt3-prompt
  nemo_model: ${evaluation.run.prompt_learn_dir}/megatron_gpt_prompt.nemo
  tensor_model_parallel_size: 2 #1 for 126m, 2 for 5b, 8 for 20b
  pipeline_model_parallel_size: 1
  model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
  precision: bf16 # must match training precision - 32, 16 or bf16
  eval_batch_size: 4
  prompt_dataset_paths: ${data_dir}/prompt_data/v1.1/squad_test.jsonl
  disable_special_tokens: False # Whether to disable virtual tokens in prompt model evaluation. This is equivalent to evaluate without prompt-/p-tuning.
```

##### 5.13.4.2. Slurm
<a id="markdown-slurm" name="slurm"></a>

Set configuration for a Slurm cluster in the `conf/cluster/bcm.yaml` file:

```yaml
partition: null
account: null
exclusive: True
gpus_per_task: 1
gpus_per_node: null
mem: 0
overcommit: False
job_name_prefix: "nemo-megatron-"
```

**Example:**

To run only the evaluation pipeline and not the data preparation, training, 
conversion or inference pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - evaluation
```

then run:
```
python3 main.py
```

##### 5.13.4.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the evaluation script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The evaluation script must be launched in a multi-node job.

To run the evaluation pipeline to evaluate a prompt learned 5B GPT model checkpoint stored in 
`/mount/results/gpt3_5b/checkpoints`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py stages=[evaluation] evaluation=prompt_gpt3/squad \
 cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data \
base_results_dir=/mount/results evaluation.run.results_dir=/mount/results/gpt3_5b/eval_prompt_squad \
evaluation.model.nemo_model=/mount/results/gpt3_5b/prompt_learning_squad/results/megatron_gpt_prompt.nemo \
evaluation.model.nemo_model=4 evaluation.model.tensor_model_parallel_size=2 \
>> /results/eval_prompt_gpt3_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/eval_prompt_gpt3_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.


#### 5.13.5. Prompt Learned T5 and mT5 Evaluation
<a id="markdown-prompt-learned-t5-and-mt5-evaluation" name="prompt-learned-t5-and-mt5-evaluation"></a>

We also provide a simple tool to help evaluate the prompt learned T5 or mT5 checkpoints. You can
evaluate the capabilities of the prompt learned models on a customized prompt learning test dataset.
We provide an example to evaluate our checkpoint, which went through prompt learning on SQuAD v1.1,
on the SQuAD v1.1 test dataset created in prompt learning step.

The configuration used for the evaluation needs to be defined in the
`conf/config.yaml` file by modifying the `evaluation` parameter, which specifies the
file to use for evaluation purposes. The `evaluation` parameter must be included in `stages`
 to run the evaluation pipeline. The value should be set to
`prompt_t5/squad.yaml`, which can be found in `conf/evaluation/prompt_t5/squad.yaml` for T5 models (or 
`prompt_mt5/squad.yaml`, which can be found in `conf/evaluation/prompt_mt5/squad.yaml` for mT5 models). The
parameters can be modified to adapt different evaluation tasks and checkpoints
in evaluation runs. For Base Command Platform, all these parameters should be overridden from the command line.

##### 5.13.5.1. Common
<a id="markdown-common" name="common"></a>
To specify the configuration, use all the `run` parameters to define the job specific config (
`run.tasks` has to be set to `prompt` to run evaluation on prompt learning test tasks):
```yaml
run:
  name: eval_${.task_name}_${.model_train_name}
  time_limit: "04:00:00"
  dependency: "singleton"
  model_train_name: t5_220m # or mt5_390m
  task_name: "squad"
  prompt_learning_dir: ${base_results_dir}/${.model_train_name}/prompt_learning_squad # assume prompt learning was on squad task
  results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}_eval
```

To specify which model checkpoint to load and which prompt learning test dataset to evaluate, 
use the following parameters:

```yaml
data:
  test_ds:
    - ${data_dir}/prompt_data/v1.1/squad_test.jsonl
  num_workers: 4
  global_batch_size: 16
  micro_batch_size: 16
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
pipeline_model_parallel_split_rank: ${divide_floor:${.pipeline_model_parallel_size}, 2}
model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
language_model_path: ${base_results_dir}/${evaluation.run.model_train_name}/convert_nemo/results/megatron_t5.nemo  # or megatron_mt5.nemo
virtual_prompt_model_file: ${evaluation.run.prompt_learning_dir}/results/megatron_t5_prompt.nemo # or megatron_mt5_prompt.nemo
```

##### 5.13.5.2. Slurm
<a id="markdown-slurm" name="slurm"></a>

Set configuration for a Slurm cluster in the `conf/cluster/bcm.yaml` file:

```yaml
partition: null
account: null
exclusive: True
gpus_per_task: 1
gpus_per_node: null
mem: 0
overcommit: False
job_name_prefix: "nemo-megatron-"
```

**Example:**

To run only the evaluation pipeline and not the data preparation, training, 
conversion or inference pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - evaluation
```

then run:
```
python3 main.py
```

##### 5.13.5.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the evaluation script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The evaluation script must be launched in a multi-node job.

To run the evaluation pipeline to evaluate a prompt learned 220M T5 model checkpoint stored in 
`/mount/results/t5_220m/prompt_learning_squad`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py stages=[evaluation] evaluation=prompt_t5/squad \
 cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data \
base_results_dir=/mount/results evaluation.run.results_dir=/mount/results/t5_220m/eval_prompt_squad \
evaluation.model.virtual_prompt_model_file=/mount/results/t5_220m/prompt_learning_squad/results/megatron_t5_prompt.nemo \
>> /results/eval_prompt_t5_log.txt 2>&1
```
The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/eval_prompt_t5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

To run the evaluation pipeline to evaluate a prompt learned 390M mT5 model checkpoint stored in 
`/mount/results/mt5_390m/prompt_learning_squad`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py stages=[evaluation] evaluation=prompt_mt5/squad \
 cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data \
base_results_dir=/mount/results evaluation.run.results_dir=/mount/results/mt5_390m/eval_prompt_squad \
evaluation.model.virtual_prompt_model_file=/mount/results/mt5_390m/prompt_learning_squad/results/megatron_mt5_prompt.nemo \
>> /results/eval_prompt_mt5_log.txt 2>&1
```
The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/eval_prompt_mt5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

#### 5.13.6. Adapter Learned and IA3 Learned GPT Evaluation
<a id="markdown-prompt-learned-and-ia3-learned-gpt-evaluation" name="prompt-learned-and-ia3-learned-gpt-evaluation"></a>

We also provide a simple tool to help evaluate the adapter and IA3 learned GPT checkpoints. You can
evaluate the capabilities of the adapter learned GPT model on a customized adapter learning test dataset.
We provide an example to evaluate our checkpoint, which went through adapter learning or IA3 learning on SQuAD v1.1.

The configuration used for the evaluation needs to be defined in the
`conf/config.yaml` file by modifying the `evaluation` parameter, which specifies the
file to be used for evaluation purposes. The `evaluation` parameter must be included in `stages`
 to run the evaluation pipeline. The value should be set to
`adapter_gpt3/squad.yaml` for adapter learning, which can be found in `conf/evaluation/adapter_gpt3/squad.yaml`. 
 The value should be set to `ia3_gpt3/squad.yaml` for IA3 learning, which can be found in `conf/evaluation/ia3_gpt3/squad.yaml`.
The parameters can be modified to adapt different evaluation tasks and checkpoints
in evaluation runs. For Base Command Platform, all these parameters should be overridden from the command line.

##### 5.13.6.1. Common
<a id="markdown-common" name="common"></a>
To specify the configuration, use all the `run` parameters to define the job specific config. (
`run.tasks` has to be set to `adapter` to run evaluation on adapter learning test tasks):
```yaml
run:
  name: ${.eval_name}_${.model_train_name}
  time_limit: "4:00:00"
  nodes: ${divide_ceil:${evaluation.model.model_parallel_size}, 8} # 8 gpus per node
  ntasks_per_node: ${divide_ceil:${evaluation.model.model_parallel_size}, ${.nodes}}
  eval_name: eval_adapter_squad # or eval_ia3_squad
  model_train_name: gpt3_5b
  tasks: "adapter" # general adapter task
  adapter_learn_dir: ${base_results_dir}/${.model_train_name}/adapter_learning_squad # or ia3_learning_squad
  results_dir: ${base_results_dir}/${.model_train_name}/${.eval_name}
```

To specify which model checkpoint to load and which adapter learning test dataset to evaluate, 
use the `model` parameter:

```yaml
data:
  test_ds:
    - ${data_dir}/prompt_data/v1.1/squad_test.jsonl
  num_workers: 4
  global_batch_size: 16
  micro_batch_size: 16
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
pipeline_model_parallel_split_rank: ${divide_floor:${.pipeline_model_parallel_size}, 2}
model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
language_model_path: ${base_results_dir}/${evaluation.run.model_train_name}/convert_nemo/results/megatron_gpt.nemo 
adapter_model_file: ${evaluation.run.adapter_learning_dir}/results/megatron_gpt_adapter.nemo # or megatron_gpt_ia3.nemo
```

##### 5.13.6.2. Slurm
<a id="markdown-slurm" name="slurm"></a>

Set configuration for a Slurm cluster in the `conf/cluster/bcm.yaml` file:

```yaml
partition: null
account: null
exclusive: True
gpus_per_task: 1
gpus_per_node: null
mem: 0
overcommit: False
job_name_prefix: "nemo-megatron-"
```

**Example:**

To run only the evaluation pipeline and not the data preparation, training, 
conversion or inference pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - evaluation
```

then run:
```
python3 main.py
```

##### 5.13.6.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the evaluation script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The evaluation script must be launched in a multi-node job.


To run the evaluation pipeline to evaluate an adapter learned 220M T5 model checkpoint stored in 
`/mount/results/gpt3_5b/adapter_learning_squad`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py stages=[evaluation] evaluation=adapter_gpt3/squad \
 cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data \
base_results_dir=/mount/results evaluation.run.results_dir=/mount/results/gpt3_5b/eval_adapter_squad \
evaluation.model.adapter_model_file=/mount/results/gpt3_5b/adapter_learning_squad/results/megatron_gpt3_adapter.nemo \
>> /results/eval_adapter_gpt3_log.txt 2>&1
```
The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/eval_adapter_gpt3_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

To run the evaluation pipeline to evaluate an IA3 learned 220M T5 model checkpoint stored in 
`/mount/results/gpt3_5b/ia3_learning_squad`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py stages=[evaluation] evaluation=ia3_gpt3/squad \
 cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data \
base_results_dir=/mount/results evaluation.run.results_dir=/mount/results/gpt3_5b/eval_ia3_squad \
evaluation.model.adapter_model_file=/mount/results/gpt3_5b/ia3_learning_squad/results/megatron_t5_ia3.nemo \
>> /results/eval_ia3_t5_log.txt 2>&1
```
The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/eval_ia3_t5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.


#### 5.13.7. Adapter Learned and IA3 Learned T5 Evaluation
<a id="markdown-adapter-learned-and-ia3-t5-evaluation" name="adapter-learned-and-ia3-t5-evaluation"></a>

The configuration used for the evaluation needs to be defined in the
`conf/config.yaml` file by modifying the `evaluation` parameter, which specifies the
file to use for evaluation purposes. The `evaluation` parameter must be included in `stages`
 to run the evaluation pipeline. The value should be set to
`adapter_t5/squad.yaml`, which can be found in `conf/evaluation/adapter_t5/squad.yaml` for adapter learned T5 models (or 
`ia3_t5/squad.yaml`, which can be found in `conf/evaluation/ia3_t5/squad.yaml` for IA3 learned models). The
parameters can be modified to adapt different evaluation tasks and checkpoints
in evaluation runs. For Base Command Platform, all these parameters should be overridden from the command line.

##### 5.13.7.1. Common
<a id="markdown-common" name="common"></a>
To specify the configuration, use all the `run` parameters to define the job specific config:
```yaml
run:
  name: eval_${.task_name}_${.model_train_name}
  time_limit: "04:00:00"
  dependency: "singleton"
  model_train_name: t5_220m
  task_name: "squad"
  adapter_learning_dir: ${base_results_dir}/${.model_train_name}/adapter_learning_squad # or ia3_learning_squad
  results_dir: ${base_results_dir}/${.model_train_name}/${.task_name}_eval
```

To specify which model checkpoint to load and which test dataset to evaluate, 
use the following parameters:

```yaml
data:
  test_ds:
    - ${data_dir}/prompt_data/v1.1/squad_test.jsonl
  num_workers: 4
  global_batch_size: 16
  micro_batch_size: 16
tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
pipeline_model_parallel_split_rank: ${divide_floor:${.pipeline_model_parallel_size}, 2}
model_parallel_size: ${multiply:${.tensor_model_parallel_size}, ${.pipeline_model_parallel_size}}
language_model_path: ${base_results_dir}/${evaluation.run.model_train_name}/convert_nemo/results/megatron_t5.nemo 
adapter_model_file: ${evaluation.run.adapter_learning_dir}/results/megatron_t5_adapter.nemo # or megatron_t5_ia3.nemo
```

##### 5.13.7.2. Slurm
<a id="markdown-slurm" name="slurm"></a>

Set configuration for a Slurm cluster in the `conf/cluster/bcm.yaml` file:

```yaml
partition: null
account: null
exclusive: True
gpus_per_task: 1
gpus_per_node: null
mem: 0
overcommit: False
job_name_prefix: "nemo-megatron-"
```

**Example:**

To run only the evaluation pipeline and not the data preparation, training, 
conversion or inference pipelines set the `conf/config.yaml` file to:

```yaml
stages:
  - evaluation
```

then run:
```
python3 main.py
```

##### 5.13.7.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the evaluation script on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The evaluation script must be launched in a multi-node job.

To run the evaluation pipeline to evaluate an adapter learned 220M T5 model checkpoint stored in 
`/mount/results/t5_220m/adapter_learning_squad`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py stages=[evaluation] evaluation=adapter_t5/squad \
 cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data \
base_results_dir=/mount/results evaluation.run.results_dir=/mount/results/t5_220m/eval_adapter_squad \
evaluation.model.adapter_model_file=/mount/results/t5_220m/adapter_learning_squad/results/megatron_t5_adapter.nemo \
>> /results/eval_adapter_t5_log.txt 2>&1
```
The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/eval_adapter_t5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

To run the evaluation pipeline to evaluate an IA3 learned 220M T5 model checkpoint stored in 
`/mount/results/t5_220m/ia3_learning_squad`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py stages=[evaluation] evaluation=ia3_t5/squad \
 cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data \
base_results_dir=/mount/results evaluation.run.results_dir=/mount/results/t5_220m/eval_ia3_squad \
evaluation.model.adapter_model_file=/mount/results/t5_220m/ia3_learning_squad/results/megatron_t5_ia3.nemo \
>> /results/eval_ia3_t5_log.txt 2>&1
```
The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/eval_ia3_t5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

### 5.14. Model Export
<a id="markdown-model-export" name="model-export"></a>

We also provide a tool to enable deployment of the NeMo Framework model on the NVIDIA Triton
Inference Server with FasterTransformer Backend.

The export supports only GPT. You can checkout T5 and mT5 support
in FasterTransformer repository but it is limited to older versions of
NeMo and Megatron-LM.

#### 5.14.1. GPT Export
<a id="markdown-gpt-export" name="gpt-export"></a>

GPT model is evaluated with `lambada` task which results can be compared with results from evaluation stage.

The configuration used for the export needs to be specified in the
`conf/config.yaml` file, specifying the `export` parameter, which specifies the
file to use for export purposes. The `export` parameter must be inclueded in `stages`
to run the training pipeline export stage. The default value is set to
`gpt3/export_gpt3`, which can be found in `conf/export/gpt3/export_gpt3.yaml`. The
parameters can be modified to adapt different export and set of tests run on prepared Triton Model Repository.
For Base Command Platform, all these parameters should be overridden from the command line.

##### 5.14.1.1. Common
<a id="markdown-common" name="common"></a>
Also the other `run` parameters might be used to define the job specific config:
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


##### 5.14.1.2. Slurm
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

##### 5.14.1.3. Base Command Platform
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

#### 5.14.2. T5 Export
<a id="markdown-t5-export" name="t5-export"></a>

T5 models are evaluated with `lambada` task which results can be compared with results from evaluation stage.

The configuration used for the export needs to be specified in the
`conf/config.yaml` file, specifying the `export` parameter, which specifies the
file to use for export purposes. The `export` parameter must be inclueded in `stages`
to run the training pipeline export stage. The value can be set to `t5/export_t5`, which can be found in `conf/export/t5/export_t5.yaml`. The parameters can be modified to adapt different export and set of tests run on prepared Triton Model Repository.
For Base Command Platform, all these parameters should be overridden from the command line.

##### 5.14.2.1. Common
<a id="markdown-common" name="common"></a>
Also the other `run` parameters might be used to define the job specific config:
```yaml
run:
  name: export_${.model_train_name}
  time_limit: "2:00:00"
  model_train_name: "t5_220m"
  training_dir: ${base_results_dir}/${.model_train_name}
  config_summary: tp${export.model.tensor_model_parallel_size}_pp${export.triton_deployment.pipeline_model_parallel_size}_${export.model.weight_data_type}_${export.triton_deployment.data_type}
  results_dir: ${base_results_dir}/${.model_train_name}/export_${.config_summary}
  model_type: "t5"
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

##### 5.14.2.2. Slurm
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

##### 5.14.2.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the export stage on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The export scripts must be launched in a multi-node job.

To run the export pipeline to evaluate a 220M T5 model checkpoint stored in 
`/mount/results/t5_220m/checkpoints`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py \
stages=[export] \
cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_t5 \
base_results_dir=/mount/results \
export.run.model_train_name=t5_220m \
export.model.tensor_model_parallel_size=1 \
export.triton_deployment.pipeline_model_parallel_size=1 \
>> /results/export_t5_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/export_t5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.



#### 5.14.3. mT5 Export
<a id="markdown-mt5-export" name="mt5-export"></a>

T5 models are evaluated with `lambada` task which results can be compared with results from evaluation stage.

The configuration used for the export needs to be specified in the
`conf/config.yaml` file, specifying the `export` parameter, which specifies the
file to use for export purposes. The `export` parameter must be inclueded in `stages`
to run the training pipeline export stage. The value can be set to `mt5/export_mt5`, which can be found in `conf/export/mt5/export_mt5.yaml`. The parameters can be modified to adapt different export and set of tests run on prepared Triton Model Repository.
For Base Command Platform, all these parameters should be overridden from the command line.

##### 5.14.3.1. Common
<a id="markdown-common" name="common"></a>
Also the other `run` parameters might be used to define the job specific config:
```yaml
run:
  name: export_${.model_train_name}
  time_limit: "2:00:00"
  model_train_name: "mt5_125m"
  training_dir: ${base_results_dir}/${.model_train_name}
  config_summary: tp${export.model.tensor_model_parallel_size}_pp${export.triton_deployment.pipeline_model_parallel_size}_${export.model.weight_data_type}_${export.triton_deployment.data_type}
  results_dir: ${base_results_dir}/${.model_train_name}/export_${.config_summary}
  model_type: "mt5"
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


##### 5.14.3.2. Slurm
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

##### 5.14.3.3. Base Command Platform
<a id="markdown-base-command-platform" name="base-command-platform"></a>
In order to run the export stage on Base Command Platform, set the
`cluster_type` parameter in `conf/config.yaml` to `bcp`. This can also be overridden
from the command line, using hydra. The export scripts must be launched in a multi-node job.

To run the export pipeline to evaluate a 125M mT5 model checkpoint stored in 
`/mount/results/mt5_125m/checkpoints`, run:
```
python3 /opt/NeMo-Megatron-Launcher/launcher_scripts/main.py \
stages=[export] \
cluster_type=bcp launcher_scripts_path=/opt/NeMo-Megatron-Launcher/launcher_scripts data_dir=/mount/data/the_pile_mt5 \
base_results_dir=/mount/results \
export.run.model_train_name=mt5_125m \
export.model.tensor_model_parallel_size=1 \
export.triton_deployment.pipeline_model_parallel_size=1 \
>> /results/export_mt5_log.txt 2>&1
```

The command above assumes you mounted the data workspace in `/mount/data`, and the results workspace in `/mount/results`. 
The stdout and stderr outputs will also be redirected to the `/results/export_mt5_log.txt` file, to be able to download the logs from NGC.
Any other parameter can also be added to the command to modify its behavior.

### 5.15 Instruction Following via Supervised Finetuning (SFT)
<a id="markdown-instruction-following-via-supervised-finetuning-(sft)" name="instruction-following-via-supervised-finetuning-(sft)"></a>
SFT is the process of finetuning all of the model's parameters on supervised data of inputs and outputs that teaches the model how to follow user specified instructions. It is typically done after model pre-training. This section describes the steps involved in finetuning a GPT model for instruction following. In the subsequent sections, we will describe how to format your data and run training.

#### 5.15.1 SFT Data Formatting
<a id="markdown-data-formatting" name="data-formatting"></a>
To demonstrate how to format your SFT data, we'll take the Dolly dataset (https://github.com/databrickslabs/dolly) as an example, which consists of 15k instruction-context-response triples.

First, to download the data, run `launcher_scripts/nemo_launcher/collections/dataprep_scripts/dolly_dataprep/download.py --path_to_save /path/to/save/data.jsonl`

The downloaded data `/path/to/save/data.jsonl` is formattated as a JSONL file with each line formatted as:

```
{
    "instruction": "When did Virgin Australia start operating?",

    "context": "Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It is the largest airline by fleet size to use the Virgin brand. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.[3] It suddenly found itself as a major airline in Australia's domestic market after the collapse of Ansett Australia in September 2001. The airline has since grown to directly serve 32 cities in Australia, from hubs in Brisbane, Melbourne and Sydney.[4]",

    "response": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.",

    "category": "closed_qa"
}
```

From the above example, there is no clear "input" and "output" field that SFT requires. An example of how to process the above data format into a JSONL file that contains "input" and "output" fields is at `launcher_scripts/nemo_launcher/collections/dataprep_scripts/dolly_datapreep/preprocess.py`. The script converts the "Instruction", "Context" and "Response" fields into "Input" and "Output". The script also concatenates the "Instruction" and "Context" fields with a \n\n separator and randomizes the order in which they appear in the input to generate a new JSONL file.

`python launcher_scripts/nemo_launcher/collections/dataprep_scripts/dolly_datapreep/preprocess.py --input /path/to/save/data.jsonl` generates a file `/path/to/save/data-output.jsonl` that can provided to SFT training described below.

For dialogue dataset, it is formatted as a JSONL file with each line formatted as:
```
{
  "mask": "User",
  "system": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n",
  "conversations": [
    {
      "from": "User",
      "value": "Who are you?"
    },
    {
      "from": "Assistant",
      "value": "I am NV Assistant, a language model trained by researchers from NVIDIA NeMo team."
    },
    {
      "from": "User",
      "value": "What can you do?"
    },
    {
      "from": "Assistant",
      "value": "I can chat with you."
    }
  ]
}, 
```
where the field `system` is used to define the system prompt for the conversation. The `conversations` is a list of multiple turn conversations. `from` is the name of the person and `value` is the actual conversation text. The `mask` field indicates which person's conversation is going to be masked during the SFT, so it is not used to compute the cross-entropy loss. 

It is important to ensure that the dialogue length is within the model's maximum sequence length. Otherwise, the entire dialogue may be masked out because it is truncated inside the dataset. In this case, you will see a 'NaN' error during training. To avoid this issue, you can split long dialogues into shorter segments, or use a model that can handle longer sequences

#### 5.15.2 SFT Training
<a id="markdown-sft-training" name="sft-training"></a>

Once you have one or more dataset you would like to finetune on, you can run the finetuning script from NeMo as follows:

```bash
TRAIN="[/path/to/dataset_1.jsonl,/path/to/dataset_2.jsonl]"

VALID="[/path/to/validation_data.jsonl]"

VALID_NAMES="[your-validation-dataset-name]"

CONCAT_SAMPLING_PROBS="[0.3,0.7]"

TP_SIZE=2

PP_SIZE=1

python /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_sft.py \
  trainer.precision=bf16 \
  trainer.max_steps=1000 \
  trainer.devices=8 \
  trainer.val_check_interval=200 \
  model.megatron_amp_O2=True \
  model.restore_from_path=/path/to/your/gpt.nemo \
  model.tensor_model_parallel_size=${TP_SIZE} \
  model.pipeline_model_parallel_size=${PP_SIZE} \
  model.optim.lr=5e-6 \
  model.answer_only_loss=True \
  model.data.train_ds.micro_batch_size=1 \
  model.data.train_ds.global_batch_size=128 \
  model.data.train_ds.file_names=${TRAIN} \
  model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
  model.data.validation_ds.micro_batch_size=1 \
  model.data.validation_ds.global_batch_size=128 \
  model.data.validation_ds.file_names=${VALID} \
  model.data.validation_ds.names=${VALID_NAMES} \
  model.data.test_ds.micro_batch_size=1 \
  model.data.test_ds.global_batch_size=128 \
  model.data.train_ds.num_workers=0 \
  model.data.validation_ds.num_workers=0 \
  model.data.test_ds.num_workers=0 \
  model.data.validation_ds.metric.name=loss \
  model.data.test_ds.metric.name=loss \
  exp_manager.create_wandb_logger=True \
  exp_manager.explicit_log_dir=/results \
  exp_manager.resume_if_exists=True \
  exp_manager.resume_ignore_no_checkpoint=True \
  exp_manager.create_checkpoint_callback=True \
  exp_manager.checkpoint_callback_params.monitor=validation_loss
```

The `${TP_SIZE}` and `${PP_SIZE}` above should correspond to the Tensor and Pipeline model parallel sizes the `/path/to/your/gpt.nemo` model was saved with.

For finetuning dialogue dataset, we just need to add one extra configuration line to indicate the dataset type is dialogue.  
```bash
  model.data.chat=True
```

### 5.16. Reinforcement Learning from Human Feedback
<a id="markdown-reinforcement-learning-from-human-feedback" name="reinforcement-learning-from-human-feedback"></a>

NeMo-RLHF is a library to fine-tune LLMs using Reinforcement Learning from Human Feedback (RLHF) in a scalable and fully distributed manner.

NeMo-RLHF supports only GPT models and implements the Proximal Policy Optimization (PPO) algorithm. Support for other models and RL algorithms will be added in future releases. Furthermore, NeMo-RLHF is not currently integrated into NeMo-Megatron-Launcher, so the RLHF jobs must be launched directly from the NeMo-RLHF repository in `/opt/nemo-rlhf`, which should be copied to the local file system on the login node.

We provide configurations to try RLHF on the newly released 2B GPT model with 4096 sequence length [available on HuggingFace](https://huggingface.co/nvidia/GPT-2B-001). We recommend using the [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) or the [Stack Exchange Preferences](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences) datasets to get started.

#### 5.16.1. Reward Model Training
<a id="markdown-reward-model-training" name="reward-model-training"></a>

NeMo-RLHF can be used to train your own reward model. The reward model is trained using a pairwise comparison loss and therefore needs a dataset with response pairs, where one response in the pair is ranked better than the other. A good reward model is crucial for the success of the PPO training in the next stage.

##### 5.16.1.1 Data preprocessing
<a id="markdown-data-preprocessing" name="data-preprocessing"></a>

With your own or publicly available data, start by processing them into a jsonl format.
This is where you should format the prompt based on your specific needs and model. For instance, if your original data looks like
```
Human: Give me a tasty apple pie recipe
AI: Sure! Here's how my grandma used to cook an awesome apple pie: (...)
```
then you may for instance turn it into
```
Setting:
You are a helpful assistant that responds concisely.

User:
Give me a tasty apple pie recipe

Assistant:
Sure! Here's how my grandma used to cook an awesome apple pie: (...)
```

Format your pairwise comparison dataset with the following structure:

```
{"text": prompt1+good_response_1}
{"text": prompt1+bad_response_1}
{"text": prompt2+good_response_2}
{"text": prompt2+bad_response_2}
...
```

where 1 and 2 are different prompts. Note that for the same prompt, prompt+good_response must come before prompt+bad_response in the dataset you generate.
If you have prompts with more than two responses, you currently need to convert them into pairwise preferences (i.e., generate multiple pairs sharing the same prompt).

Then use the `preprocess_data_for_megatron.py` script to convert this jsonl format into the NeMo format. 
For reference we used the following command for preprocessing the dataset using the SentencePiece tokenizer:

```bash
python3 /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input "test.jsonl" \
    --output-prefix "./output" \
    --tokenizer-model sp_tokenizer_256k.model \
    --tokenizer-library=sentencepiece \
    --json-keys text \
    --dataset-impl mmap \
    --workers 30 \
    --chunk_size=100 \
    --append-eod
```
This generates files `output_text_document.bin` and `output_text_document.idx` to use for reward model training, described below.

##### 5.16.1.2 Training a Reward Model
<a id="markdown-training-a-reward-model" name="training-a-reward-model"></a>

To launch reward model training we first need to start with a pre-trained or fine-tuned NeMo checkpoint. Our `training_rm.yaml` file has default settings for the 2B model but feel free to use any other model (adjusting the config accordingly). An example command to begin training is:

```bash
cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& python -u examples/nlp/gpt/train_reward_model.py \
    --config-path=examples/nlp/gpt/conf/ \
    --config-name=training_rm \
    exp_manager.explicit_log_dir=/path/to/rm_output_dir \
    model.pretrained_checkpoint.restore_from_path=/path/to/init_model.nemo \
    "model.data.data_prefix={train: [/path/to/rm_train], validation: [/path/to/rm_val], test: [/path/to/rm_test]}"
```

The data files should point to the names of datasets generated as described in the previous section, but without the ".bin" or ".idx" suffix.
Note that if you are using the command above with your own pre-trained model, you will need to modify `training_rm.yaml` (or the command line) to provide correct values for `tokenizer.model` and `tokenizer.tokenizer_model`.
You can use `tar tvf /path/to/init_model.nemo` to inspect the model and obtain the name of its tokenizer files: typically, both files are identical and you may thus use the same name for both options, e.g. with
```bash
model.tokenizer.model=nemo:2b164b2c1dd74bd691ff90a0db3d39b8_xyz_256k.model \
model.tokenizer.tokenizer_model=nemo:2b164b2c1dd74bd691ff90a0db3d39b8_xyz_256k.model \
```

_Remark: currently, the example training script does not automatically run evaluation on the provided test set. This may change in a future release._

##### 5.16.1.3 Reward Model Evaluation
<a id="markdown-reward-model-evaluation" name="reward-model-evaluation"></a>

Once trained, a reward model may be served for evaluation purpose, as described in the section "Launching the Reward Model Inference Server" below.
This can also useful to compute the mean / std of reward predictions before doing PPO training, to be able to normalize them: documentation and scripts to perform such normalization will be provided soon.

#### 5.16.2. PPO Training
<a id="markdown-ppo-training" name="ppo-training"></a>

After fine-tuning a GPT model using Supervised Finetuning (SFT) and training a Reward Model as explained in the previous sections, NeMo-RLHF can be used to launch PPO jobs to fine-tune the SFT model using RLHF. During PPO training, four different models will be interacting with each other:

1. The PPO Actor Network (also known as the Policy Network): This is the model we are training, and it should start from an SFT model trained as explained in the SFT section.
2. The Reward Model (RM) Network (also known as a Preference Model (PM)): This model takes a prompt concatenated with a response as input, and outputs a single scalar value: the reward, which the PPO algorithm will try to maximize. The RM should be a model trained as described in the RM Training section.
3. The PPO Critic Network (also known as the Value Network): Since PPO is an Actor-Critic algorithm, we need a Critic to guide the Actor during training. The Critic will provide value estimates for each token in the responses provided by the Actor. These values can be seen as an estimate of the total reward the Actor will receive after generating all the remaining tokens. The Critic should be initialized from the RM so as to provide useful feedback in the early stages of training. Note: The RM generates a single reward for the entire sequence, whereas the Critic generates a value for each token.
4. The Initial Policy Network (also known as the Reference Model): We use this model to compute a KL Divergence penalty term that ensures that the PPO Actor does not diverge too much from the Initial Policy. This way, we prevent the PPO Actor from overfitting to the rewards given by the RM, and ensure it does not forget the knowledge it acquired during pretraining and SFT. This model should be the one used to initialize the PPO Actor Network.

To launch a full PPO training job, we need to launch the RM and the Initial Policy as inference servers. These two models are not trained, so they only need to perform inference and share their results with the PPO Actor. However, both the PPO Actor and Critic need to be trained.

Our architecture is designed to launch all four models completely separately. Therefore, we will launch two inference servers (one for the RM and one for the initial policy), one server that can do inference and training (the PPO Critic), and one master job to control the training (the PPO Actor). Next we will look at how to launch each of those four jobs.

##### 5.16.2.1 Launching the Reward Model Inference Server
<a id="markdown-launching-the-reward-model-inference-server" name="launching-the-reward-model-inference-server"></a>

To launch the Reward Model inference server, this command can be run inside the container:

```bash
cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& python -u examples/nlp/gpt/serve_reward_model.py \
    --config-path=examples/nlp/gpt/conf/ \
    --config-name=inference_rm \
    gpt_rm_model_file=/path/to/trained_rm.nemo \
    port=5555
```

This command will launch the RM inference server on the local computer, using port 5555. All the configuration parameters can be modified in the `inference_rm.yaml` file, or by overriding them through the CLI command. Ensure `server=True` is set in the configuration of this job to correctly launch the inference server.

##### 5.16.2.2 Launching the Initial Policy Inference Server
<a id="markdown-launching-the-initial-policy-inference-server" name="launching-the-initial-policy-inference-server"></a>

To launch the Initial Policy inference server, this command can be run inside the container:

```bash
cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& python -u examples/nlp/gpt/serve_initial_policy.py \
    --config-path=examples/nlp/gpt/conf/ \
    --config-name=inference_initial_policy \
    gpt_model_file=/path/to/sft_model.nemo \
    port=5556
```

This command will launch the Initial Policy inference server on the local computer, using port 5556. All the configuration parameters can be modified in the `inference_initial_policy.yaml` file, or by overriding them through the CLI command. Ensure `server=True` is set in the configuration of this job to correctly launch the inference server.

##### 5.16.2.3 Launching the PPO Critic Training and Inference Server
<a id="markdown-launching-the-ppo-critic-training-and-inference-server" name="launching-the-ppo-critic-training-and-inference-server"></a>

The PPO Critic has to perform both inference *and* training.
To launch the PPO Critic server, which provides both functionalities, this command can be run inside the container:

```bash
cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& python -u examples/nlp/gpt/serve_ppo_critic.py \
    --config-path=examples/nlp/gpt/conf/ \
    --config-name=gpt_ppo_critic \
    exp_manager.explicit_log_dir=/path/to/critic_output_dir \
    model.pretrained_checkpoint.restore_from_path=/path/to/trained_rm.nemo \
    inference.port=5557
```

This command will launch the PPO Critic server on the local computer, using port 5557. All the configuration parameters can be modified in the `gpt_ppo_critic.yaml` file, or by overriding them through the CLI command: in particular, the Critic's model config should match the one used to train the RM, and you may need to provide the correct name of the tokenizer files as described in the RM training section above.
Ensure `inference.server=True` is set in the configuration of this job to correctly launch the server.

##### 5.16.2.4 Launching the PPO Actor Training
<a id="markdown-launching-the-ppo-actor-training" name="launching-the-ppo-actor-training"></a>
The PPO Actor training job contains the master controller that makes the HTTP calls to all three servers when needed. To launch the PPO Actor server, this command can be run inside the container:

```bash
cd /opt/nemo-rlhf \
&& export PYTHONPATH="/opt/nemo-rlhf:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& python -u examples/nlp/gpt/train_gpt_ppo_actor.py \
    --config-path=examples/nlp/gpt/conf/ \
    --config-name=gpt_ppo_actor \
    exp_manager.explicit_log_dir=/path/to/actor_output_dir \
    "model.data.data_prefix={train: [/path/to/actor_train], validation: [/path/to/actor_val], test: [/path/to/actor_test]}" \
    model.pretrained_checkpoint.restore_from_path=/path/to/sft_model.nemo
```

This command will launch the PPO Actor job on the local computer. All the configuration parameters can be modified in the `gpt_ppo_actor.yaml` file, or by overriding them through the CLI command: in particular, the Actor's model config should match the one used to train the SFT model, and you may need to provide the correct name of the tokenizer files as described in the RM training section above.

The data files should point to the names of datasets (without the ".bin" or ".idx" suffix) generated in a manner similar to what is described in the RM training section, but with an important difference: they should only contain prompts.
This means that the raw .jsonl from which the datasets are built should follow the following format:
```
{"text": prompt1}
{"text": prompt2}
{"text": prompt3}
...
```

_Remark: currently, the example training script does not automatically run evaluation on the provided test set. This may change in a future release._

##### 5.16.2.5 Launching all jobs at once with SLURM
<a id="markdown-launching-all-jobs-at-once-with-slurm" name="launching-all-jobs-at-once-with-slurm"></a>
Heterogeneous jobs can be used to launch all four jobs simultaneously on different nodes, using a script like:

```bash
#!/bin/bash
#SBATCH -N 1 --ntasks-per-node 8 -t 4:00:00 --exclusive
#SBATCH hetjob
#SBATCH -N 1 --ntasks-per-node 8 -t 4:00:00 --exclusive
#SBATCH hetjob
#SBATCH -N 1 --ntasks-per-node 8 -t 4:00:00 --exclusive
#SBATCH hetjob
#SBATCH -N 8 --ntasks-per-node 8 -t 4:00:00 --exclusive

RM_MODEL=/path/to/trained_rm.nemo
ACTOR_MODEL=/path/to/sft_model.nemo
OUTPUT_DIR=/path/to/output_dir
TRAIN_DATA_PATH=/path/to/train_actor
VALID_DATA_PATH=/path/to/val_actor
TEST_DATA_PATH=/path/to/test_actor

NEMO_RLHF_DIR=/opt/nemo-rlhf
CONTAINER="nvcr.io/ea-bignlp/nemofw-training:23.07-py3"

mkdir -p $OUTPUT_DIR

# START HETEROGENEUS JOB 0

mkdir -p ${OUTPUT_DIR}/rm
RM_OUT=${OUTPUT_DIR}/rm/rm-%j.log
RM_ERR=${OUTPUT_DIR}/rm/rm-%j.err
read -r -d '' cmd_rm_inference <<EOF
cd ${NEMO_RLHF_DIR} \
&& export PYTHONPATH="${NEMO_RLHF_DIR}:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& python -u examples/nlp/gpt/serve_reward_model.py \
    --config-path=examples/nlp/gpt/conf/ \
    --config-name=inference_rm \
    gpt_rm_model_file=${RM_MODEL} \
    port=${RM_PORT=5555}
EOF

srun -o $RM_OUT -e $RM_ERR --het-group=0 --container-image=${CONTAINER} bash -c "${cmd_rm_inference}" & pids[0]=$!

# END HETEROGENEUS JOB 0

####################################################

# START HETEROGENEUS JOB 1

mkdir -p ${OUTPUT_DIR}/init_policy
IP_OUT=${OUTPUT_DIR}/init_policy/init_policy-%j.log
IP_ERR=${OUTPUT_DIR}/init_policy/init_policy-%j.err
read -r -d '' cmd_init_policy_inference <<EOF
cd ${NEMO_RLHF_DIR} \
&& export PYTHONPATH="${NEMO_RLHF_DIR}:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& python -u examples/nlp/gpt/serve_initial_policy.py \
    --config-path=examples/nlp/gpt/conf/ \
    --config-name=inference_initial_policy \
    gpt_model_file=${ACTOR_MODEL} \
    port=${INIT_POLICY_PORT=5556}
EOF

srun -o $IP_OUT -e $IP_ERR --het-group=1 --container-image=${CONTAINER} bash -c "${cmd_init_policy_inference}" & pids[1]=$!

# END HETEROGENEUS JOB 1

######################################################

# START HETEROGENEUS JOB 2

mkdir -p ${OUTPUT_DIR}/critic
CRIT_OUT=${OUTPUT_DIR}/critic/critic-%j.log
CRIT_ERR=${OUTPUT_DIR}/critic/critic-%j.err
read -r -d '' cmd_critic_inference <<EOF
cd ${NEMO_RLHF_DIR} \
&& export PYTHONPATH="${NEMO_RLHF_DIR}:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& python -u examples/nlp/gpt/serve_ppo_critic.py \
    --config-path=examples/nlp/gpt/conf/ \
    --config-name=gpt_ppo_critic \
    exp_manager.explicit_log_dir=${OUTPUT_DIR}/critic \
    model.pretrained_checkpoint.restore_from_path=${RM_MODEL} \
    inference.port=${CRITIC_PORT=5557}
EOF

srun -o $CRIT_OUT -e $CRIT_ERR --het-group=2 --container-image=${CONTAINER} bash -c "${cmd_critic_inference}" & pids[2]=$!

# END HETEROGENEUS JOB 2

sleep 30
####################################################

# START HETEROGENEUS JOB 3

host_rm="$(scontrol show hostnames=$SLURM_JOB_NODELIST_HET_GROUP_0 | head -n1)"
host_init_policy="$(scontrol show hostnames=$SLURM_JOB_NODELIST_HET_GROUP_1 | head -n1)"
host_critic="$(scontrol show hostnames=$SLURM_JOB_NODELIST_HET_GROUP_2 | head -n1)"

mkdir -p ${OUTPUT_DIR}/actor
ACT_OUT=${OUTPUT_DIR}/actor/actor-%j.log
ACT_ERR=${OUTPUT_DIR}/actor/actor-%j.err
read -r -d '' cmd_ppo <<EOF
cd ${NEMO_RLHF_DIR} \
&& export PYTHONPATH="${NEMO_RLHF_DIR}:${PYTHONPATH}" \
&& export HYDRA_FULL_ERROR=1 \
&& python -u examples/nlp/gpt/train_gpt_ppo_actor.py \
    --config-path=examples/nlp/gpt/conf/ \
    --config-name=gpt_ppo_actor \
    exp_manager.explicit_log_dir=${OUTPUT_DIR}/actor
    trainer.num_nodes=8 \
    "model.data.data_prefix={train: [${TRAIN_DATA_PATH}], validation: [${VALID_DATA_PATH}], test: [${TEST_DATA_PATH}]}" \
    model.pretrained_checkpoint.restore_from_path=${ACTOR_MODEL} \
    model.rlhf.reward_model.ip=${host_rm} \
    model.rlhf.reward_model.port=${RM_PORT=5555} \
    model.rlhf.initial_policy.ip=${host_init_policy} \
    model.rlhf.initial_policy.port=${INIT_POLICY_PORT=5556} \
    model.rlhf.critic.ip=${host_critic} \
    model.rlhf.critic.port=${CRITIC_PORT=5557}
EOF

srun -o $ACT_OUT -e $ACT_ERR --het-group=3 --container-image=${CONTAINER} bash -c "${cmd_ppo}" & pids[3]=$!

# END HETEROGENEUS JOB 3

# The code below monitors the four SLURM jobs to ensure they are all stopped when one of them finishes.
# (otherwise some jobs may remain pending until they reach the cluster time limit).
all_done=false
while ! $all_done; do
    all_done=true
    for pid in "${pids[@]}"; do
        if ps -p "$pid" > /dev/null; then
            # Process is still running.
            all_done=false
        else
            # Process is no longer running => check its exit status.
            wait "$pid"
            exit_code=$?
            echo "Process $pid exited with code $exit_code at $(date '+%Y-%m-%d %H:%M:%S')"
            # Wait a bit (to get a clean stack trace in case there is one being generated), then kill the
            # remaining processes if needed.
            sleep 60
            for other_pid in "${pids[@]}"; do
                if ps -p "$other_pid" > /dev/null; then
                    echo "Killing processs $other_pid"
                    kill -9 "$other_pid"
                fi
            done
            exit $exit_code
        fi
    done

    # Sleep for a while before checking again.
    sleep 60
done
```

It is important to launch all jobs with `&` after the `srun` command, to ensure they do not block each other.

Note: all four scripts support data parallelism. Therefore, the SLURM `–ntasks-per-node` value may be set to the number of GPUs on each node, and `trainer.devices` should also be set to that same value.

##### 5.16.2.6 Ensuring consistency between jobs
<a id="markdown-ensuring-consistency-between-jobs" name="ensuring-consistency-between-jobs"></a>

Since there are four independent jobs, each with their own config, one must be careful to ensure that the various configs are compatible with each other by following the guidelines below:

- `critic.exp_manager.checkpoint_callback_params.every_n_train_steps` should be set to `actor.trainer.val_check_interval * actor.model.global_batch_size / critic.model.global_batch_size` so that the Critic is saved at the same frequency as the Actor.
- `critic.inference.micro_batch_size` should be set to `actor.model.rlhf.ppo.rollout_micro_batch_size` divided by the Critic's data parallel size (which is obtained by the total number of GPUs the Critic is running on, i.e., `trainer.devices * trainer.num_nodes`, divided by the product of `model.tensor_model_parallel_size * model.pipeline_model_parallel_size`), rounded up.
This ensures that the Critic can process the Actor's requests as efficiently as possible.
- Similarly, `rm.inference_micro_batch_size` and `init_policy.inference_micro_batch_size` should be set to `actor.model.rlhf.ppo.rollout_micro_batch_size` divided by the RM and Initial Policy's data parallel size, rounded up.
- `critic.model.ppo_epochs` should be equal to `actor.model.rlhf.ppo.epochs` so that the Critic performs the same number of updates as the Actor on the rollout buffer data.

##### 5.16.2.7 PPO Hyper-parameters
<a id="markdown-ppo-hyper-parameters" name="ppo-hyper-parameters"></a>

All the model parameters can be controlled the same way as in other NeMo training jobs. However, we also provide full control of the behavior of PPO during training, with a section in the Actor config yaml file inside `model.rlhf`. These are the available hyper-parameters:

- `rlhf.{reward_model,critic,initial_policy}.{ip,port}`: Provide the ip address and the port where the Reward Model, PPO Critic and Initial Policy will be running, to enable communication with them.
- `rlhf.ppo.entropy_bonus`: Weight of the entropy term in the PPO loss.
- `rlhf.ppo.inital_pollicy_kl_penalty`: Weight of the KL Divergence w.r.t. the Initial Policy in the PPO loss.
- `rlhf.ppo.use_absolute_kl`: Whether or not to use the absolute value of the KL Divergence w.r.t. the Initial Policy.
- `rlhf.ppo.epochs`: Number of training epochs the Actor will perform on the samples stored in the rollout buffer before generating new samples.
- `rlhf.ppo.num_rollout_samples`: Number of samples that will be generated during the rollout stage before moving to the training stage.
- `rlhf.ppo.rollout_micro_batch_size`: Micro batch size for the rollout phase. Each GPU will load this many prompts at once and generate responses for them.
- `rlhf.ppo.ratio_eps`: Epsilon value for clipping the PPO ratio during training.
- `rlhf.ppo.discount`: Discount factor for calculating the returns and advantages.
- `rlhf.ppo.gae_lambda`: Lambda value for the Generalized Advantage Estimation (GAE) calculation.
- `rlhf.ppo.normalize_advantage`: Whether or not to normalize the advantages to have a mean of zero and standard deviation of one within each global batch.

Note that although the sampling parameters during the rollout phase can also be modified (through `model.sampling_params.*`), it is not recommended to do so because the implementation currently does not account for these changes when computing the log probabilities of the generated responses.

The Critic's config also contains a `model.rlhf` section with the following hyper-parameter:
- `rlhf.ppo.critic_loss_clip_value`: Used in the Critic loss term that clamps the difference between the current Critic value predictions and those that were predicted during rollout generation (disabled when set to zero).

#### 5.16.3. Future Work
<a id="markdown-future-work" name="future-work"></a>

- The throughput of PPO will be increased in future releases.
- We will continue improving the stability of the PPO learning phase.
- We will add more learning algorithms beyond PPO.

### 5.17 Curating pretraining datasets with the NeMo Data Curator

The NeMo Data Curator is a Python library that consists of a collection of scalable data-mining modules for curating NLP data for training LLMs. The modules within the NeMo Data Curator enable NLP researchers to mine high-quality text at scale from massive uncurated web corpora.

Currently, within the NeMo Data Curator, we support the following data-curation modules:
 - Configurable data download and text extraction:
   - Default implementations of download and extraction of Common Crawl, Wikipedia, and ArXiv data
   - Users can easily customize the download and extraction and extend to other datasets (see NeMo Data Curator internal documentation available in the container for more information)
 - Text reformatting and cleaning via [ftfy](https://ftfy.readthedocs.io/en/latest/)
 - Quality filtering:
   - Multilingual heuristic-based filtering
   - Classifier-based filtering via [fastText](https://fasttext.cc/)
 - Document-level deduplication
   - Exact deduplication
   - Fuzzy deduplication. Our implementation of fuzzy deduplication builds off of the following existing libraries:
     - For computing MinHash signatures we use a modified version of the MinHasher class provided in [pyLSH](https://github.com/mattilyra/LSH)
     - For the locality sensitive hashing, we extended the Redis-based implementation found in [datasketch](https://github.com/ekzhu/datasketch) beyond a single Redis server to a Redis Cluster. This enables this module to efficiently deduplicate large datasets that do not fit in memory of a single node (e.g., several TB of text)

The modules are implemented in a scalable manner using [Message Passing Interface (MPI) for Python (mpi4py)](https://mpi4py.readthedocs.io/en/stable/) and we use [Dask](https://dask.org) for creating balanced input jsonl files. With the scalable modules within the NeMo Data Curator, we have been have been able to fully process a [Common Crawl Snapshot](https://commoncrawl.org/2020/12/nov-dec-2020-crawl-archive-now-available/) (consisting of 60 TB of compressed WARC files) in approximately two days using 30 CPU nodes (with hardware similar to the `c5.24xlarge` [Amazon AWS C5 instance](https://aws.amazon.com/ec2/instance-types/c5/)). Please note that the core functions used within the NeMo Data Curator (e.g., html extraction, text cleaning, heuristic filtering, etc.) have not been fully optimized. The main goal of the NeMo Data Curator is to provide users the capability to apply these functions to their large datasets using many compute nodes.

If users to desire to use the NeMo Data Curator in order to curate their own pretraining datasets, they should copy it out of the container using the
command provided in the [environment preparation section of the quick start guide](#5111-slurm). Within the `nemo-data-curator` directory, they
can use the example SLURM scripts and additional documentation provided in the docs sub-directory and README of that directory.

## 6. Deploying the NeMo Megatron Model

This section describes the deployment of the NeMo Megatron model on the NVIDIA Triton
Inference Server with FasterTransformer Backend on both single and multiple
node environments.    NVIDIA Triton Inference Server supports many inference
scenarios, of which two most important are:
* Offline inference scenario - with a goal to maximize throughput regardless
    of the latency, usually achieved with increasing batch size and using server
    static batching feature.
* Online inference scenario - with a goal to maximize throughput within a given
    latency budget, usually achieved with small batch sizes and increasing
    concurrency requests to the server, using dynamic batching feature.


### 6.1. Run NVIDIA Triton Server with Generated Model Repository
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

## 6.2. GPT Text Generation with Ensemble

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



## 6.3. UL2 Checkpoint Deployment

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




## 7. Performance
<a id="markdown-performance" name="performance"></a>

### 7.1. GPT Results
<a id="markdown-gpt-results" name="gpt-results"></a>

#### 7.1.1. Training Accuracy Results
Training Accuracy: NVIDIA DGX SuperPOD (8 x 8 x A100 80GB for 126M GPT Model; 16 x 8 x A100 80GB for 5B GPT Model)

We evaluated the 126M parameter and 5B parameter models on 8 different language
tasks. The results can be found in the table below. All the tasks are provided
as part of the evaluation harness, so the user can evaluate any `.nemo`
checkpoint file on all these tasks.

|Task                            |Metric                        | 126M                         | 5B                             |
| ---------------- | ---------------- | ---------------- | ---------------- |
|Lambada                     |Accuracy                    | 38.70%                     | 68.93%                     |
|                                    |PPL                             | 25.8                         | 4.22                         |
|Boolq                         |Accuracy                    | 56.94%                     | 65.29%                     |
|Race                            |Accuracy                    | 28.71%                     | 38.66%                     |
|                                    |Accuracy Norm         | 34.74%                     | 41.62%                     |
|Piqa                            |Accuracy                    | 61.21%                     | 73.88%                     |
|                                    |Accuracy Norm         | 61.97%                     | 75.40%                     |
|Hellaswag                 |Accuracy                    | 28.48%                     | 46.45%                     |
|                                    |Accuracy Norm         | 29.54%                     | 60.85%                     |
|Winogrande                |Accuracy                    | 50.43%                     | 60.77%                     |
|Wikitext2                 |Word PPL                    | 31.35                        | 12.36                        |
|                                    |Byte PPL                    | 1.9                            | 1.6                            |
|                                    |Bits per Byte PPL | 0.64                         | 0.47                         |
|Wikitext103             |Word PPL                    | 31.35                        | 12.36                        |
|                                    |Byte PPL                    | 1.9                            | 1.6                            |
|                                    |Bits per Byte PPL | 0.64                         | 0.47                         |

Training the 5B GPT model to convergence takes 6.5 days, and the loss curve can be seen in the figure below:

<img src="img/5B_GPT_3_loss_final.svg"/>

The table below shows the converged training loss, the throughput, and the
total time to train for the 5B GPT model, using a given number of GPUs and a
given Global Batch Size (GBS).

| \#GPUs | GBS    | Seq Length | \#Tokens | Loss    | Throughput (Tokens/sec) | Time to Train (days) |
| ------ | ---- | ---------- | -------- | ----- | ----------------------- | -------------------- |
| 160    | 1440 | 2048       | 300B     | 1.685 | 726,384                 | 4.8                  |


#### 7.1.2. Training Performance Results
<a id="markdown-training-performance-results" name="training-performance-results"></a>
Training performance: 
 - NVIDIA DGX SuperPOD (16 x 8 x A100 80GB for 5B GPT model)
 - NVIDIA DGX SuperPODs (128 x 8 x A100 80GB for 175B GPT model)

We measured the throughput of training 5B and 175B parameter GPT models on 
different numbers of DGX nodes, and we achieved near-linear
scaling. For example, when scaling from 1 node to 32 nodes with a 5B model, we achieve a 28.73x
speed-up. When scaling from 8 nodes to 128 (16x more nodes) nodes with a 175B model, we achieve 14.62x speed-up.
The tables and charts below show the performance results.

|      |                                 |        |        |        | Nodes  |        |         |
| ---- | ------------------------------- | ------ | ------ | ------ | ------ | ------ | ------- |
|      |                                 | 1      | 2      | 4      | 8      | 16     | 32      |
|      | Tokens per Second               | 40345  | 79815  | 161754 | 312774 | 659481 | 1159288 |
| 5B   | Perfect Linear Scaling (Tokens) | 40345  | 80690  | 161380 | 322760 | 645520 | 1291040 |
|      | Speed-up                        | 1x     | 1.98x  | 4.01x  | 7.75x  | 16.35x | 28.73x  |

<img src="img/5B_GPT_3_throughput.svg"/>

|      |                                 |        | Nodes  |        |       |        |
| ---- | ------------------------------- | ------ | ------ | ------ | ----- | ------ |
|      |                                 | 8      | 16     | 32     | 64    | 128    |
|      | Tokens per Second               | 7500   | 14950  | 29537  | 58211 | 109684 |
| 175B | Perfect Linear Scaling (Tokens) | 7500   | 15000  | 30000  | 60000 | 120000 |
|      | Speed-up                        | 1x     | 1.99x  | 3.94x  | 7.76x | 14.62x |

<img src="img/175B_GPT_3_throughput.svg"/>

#### 7.1.3. Inference Performance
<a id="markdown-inference-performance" name="inference-performance"></a>

Inference performance was measured for NVIDIA DGX SuperPOD (1 x 8 x A100 80GB).

Inference parameters:
* batch size: 1
* input tokens length: 60
* output tokens length: 20

<img src="img/infer_model_size_gpt3.svg"/>

| GPT Model size | Average latency [ms]           | TP | PP | GPUs |
|----------------|--------------------------------|----|----|------|
| 5B             |                             87 |  8 |  4 |   32 |
| 20B            |                            202 |  8 |  4 |   32 |
| 175B           |                            893 |  8 |  4 |   32 |
| 530B           |                            977 | 32 |  1 |   32 |

### 7.2. T5 Results
<a id="markdown-t5-results" name="t5-results"></a>

#### 7.2.1. Training Accuracy Results

The user can also prompt-learn on top of any `.nemo` trained checkpoint file on `SQuAD` task mentioned in T5 prompt-learning section.
The results can be found in the table below.

| Task   | Metric      | 220M  | 3B    |
|--------|-------------|-------|-------|
| SQuAD  | Exact Match | 74.20 | 78.52 |
| SQuAD  | F1          | 84.54 | 87.17 |

Training the 220M T5 model to convergence takes 4 days, and the loss curve can be seen in the figure below:

<img src="img/220M_T5_loss_final.svg"/>

The table below shows the converged training loss, the throughput, and the
total time to train for the 220M T5 model, using a given number of GPUs and a
given Global Batch Size (GBS).

| \#GPUs | GBS    | Seq Length | \#Tokens | Loss    | Throughput (Tokens/sec) | Time to Train (days) |
|--------|------|------------|----------|-------|-------------------------|----------------------|
| 32         | 2048 | 512                | 1T             | 1.501 | 3,273,728                             | 4                                        |


Training the 3B T5 model to convergence takes 11 days, and the loss curve of a fully trained model can be seen in the figure below:

<img src="img/3B_T5_loss_100percent.svg"/>

The table below shows the converged training loss, the throughput, and the
total time to train for the 3B T5 model, using a given number of GPUs and a
given Global Batch Size (GBS).

| \#GPUs | GBS    | Seq Length | \#Tokens | Loss  | Throughput (Tokens/sec) | Time to Train (days) |
|--------|------|------------|----------|--------------------|-------------------------|----------------------|
| 160        | 2160 | 512                | 1T             | 1.147                            | 1,395,131                             | 11                                     |



#### 7.2.2. Training Performance Results
<a id="markdown-training-performance-results" name="training-performance-results"></a>
Training Performance: NVIDIA DGX SuperPOD (20 x 8 x A100 80GB for 3B T5 Model)

We measured the throughput of training a 3B parameter T5 model on NVIDIA DGX
SuperPOD using a different number of nodes. When scaling from 1 node to 20 nodes, we achieve 16.38x
speed-up. We are actively working on improving the scaling performance for T5 models. The table and chart below show the performance results.


|        |                                    |           |          |          | Nodes    |          |          |
|--------|------------------------------------|-----------|----------|----------|----------|----------|----------|
|        |                                    | 1         | 2        | 4        | 5        | 10       | 20       |
|        | Tokens per Second                  | 110769    | 215579   | 417644   | 515100   | 957506   |  1626353 |
| 3B     | Perfect Linear Scaling (Tokens)    | 110769    | 221538   | 443077   | 553846   | 1107692  | 2215385  |
|        | Speed-up                           | 1x        | 1.95x    | 3.77x    | 4.65x    | 8.64x    | 14.68x   |

<img src="img/3B_T5_throughput_2208.svg"/>

#### 7.2.3. Inference Performance
Inference performance was measured for NVIDIA DGX SuperPOD (1 x 8 x A100 80GB).

Inference parameters:
* batch size: 1
* input tokens length: 60
* output tokens length: 20

<img src="img/infer_model_size_t5.svg"/>

| T5 Model size | Average latency [ms] | TP | PP | GPUs |
|---------------|----------------------|----|----|------|
| 3B            |                   94 |  2 |  1 |    2 |
| 11B           |                  123 |  4 |  1 |    4 |
| 23B           |                  213 |  4 |  1 |    4 |
| 41B           |                  332 |  8 |  1 |    8 |


### 7.3. mT5 Results
<a id="markdown-t5-results" name="t5-results"></a>

#### 7.3.1. Training Accuracy Results
Training Accuracy: NVIDIA DGX SuperPOD (4 x 8 x A100 80GB for 170M mT5 Model; 8 x 8 x A100 80GB for 390M mT5 Model; 20 x 8 x A100 80GB for 3B mT5 Model)

We evaluated our mT5 models on XQuAD task. The results can be found in the table below. The user can 
fine-tune on top of any `.nemo` trained checkpoint file on `XQuAD` task mentioned in mT5 fine-tuning section.

| Task-Language | Metric      | 170M  | 390M  |
|---------------|-------------|-------|-------|
| XQuAD-de      | Exact Match | 43.0  | 54.7  |
| XQuAD-en      | Exact Match | 63.8  | 68.8  |
| XQuAD-es      | Exact Match | 47.0  | 55.3  |
| XQuAD-hi      | Exact Match | 34.5  | 47.1  |
| XQuAD-zh      | Exact Match | 46.8  | 56.1  |

The user can also prompt-learn on top of any `.nemo` trained checkpoint file on `SQuAD` task mentioned in mT5 prompt-learning section.
The results can be found in the table below.

| Task   | Metric      | 390M  | 3B    |
|--------|-------------|-------|-------|
| SQuAD  | Exact Match | 76.86 | 81.55 |
| SQuAD  | F1          | 84.67 | 89.34 |



Training the 170M mT5 model to convergence takes 4 days, and the loss curve can be seen in the figure below:

<img src="img/170M_mT5_loss_final.svg"/>

The table below shows the converged training loss, the throughput, and the
total time to train for the 170M mT5 model, using a given number of GPUs and a
given Global Batch Size (GBS).

| \#GPUs | GBS    | Seq Length | \#Tokens | Loss  | Throughput (Tokens/sec) | Time to Train (days) |
|--------|------|------------|----------|-------|-------------------------|----------------------|
| 32         | 2048 | 512                | 1T             | 1.980 | 4,112,062               | 4                                        |




Training the 390M mT5 model to convergence takes 4 days, and the loss curve can be seen in the figure below:

<img src="img/390M_mT5_loss_final.svg"/>

The table below shows the converged training loss, the throughput, and the
total time to train for the 390M mT5 model, using a given number of GPUs and a
given Global Batch Size (GBS).

| \#GPUs | GBS    | Seq Length | \#Tokens | Loss  | Throughput (Tokens/sec) | Time to Train (days) |
|--------|------|------------|----------|-------|-------------------------|----------------------|
| 64     | 2048 | 512                | 1T             | 1.584 | 3,744,914               | 4                                        |


Training the 3B mT5 model to convergence takes 14 days, and the loss curve of a fully trained model can be seen in the figure below:

<img src="img/3B_mT5_loss_final.svg"/>

The table below shows the converged training loss, the throughput, and the
total time to train for the 3B T5 model, using a given number of GPUs and a
given Global Batch Size (GBS).

| \#GPUs | GBS  | Seq Length | \#Tokens | Loss   | Throughput (Tokens/sec) | Time to Train (days) |
|--------|------|------------|----------|--------|-------------------------|----------------------|
| 160        | 1920 | 512                | 1T             | 1.134  | 911,065                 | 14                   |


#### 7.3.2. Training Performance Results
<a id="markdown-training-performance-results" name="training-performance-results"></a>
Training Performance: NVIDIA DGX SuperPOD (20 x 8 x A100 80GB for 3B mT5 Model)

We measured the throughput of training a 3B parameter mT5 model on NVIDIA DGX
SuperPOD using a different number of nodes. When scaling from 1 node to 20 nodes, we achieve 14.87x
speed-up. We are actively working on improving the scaling performance for mT5 models. 
The table and chart below show the performance results.


|         |                                    |        |         |         | Nodes   |         |          |
|---------|------------------------------------|--------|---------|---------|---------|---------|----------|
|         |                                    | 1      | 2       | 4       | 5       | 10      | 20       |
|         | Tokens per Second                  | 91166  | 179583  | 346263  | 429088  | 798570  | 1303767  |
| 3B      | Perfect Linear Scaling (Tokens)    | 91166  | 182331  | 364663  | 455829  | 911657  | 1823314  |
|         | Speed-up                           | 1x     | 1.97x   | 3.8x    | 4.71x   | 8.76x   | 14.3x    |


<img src="img/3B_mT5_throughput_2208.svg"/>

#### 7.3.3. Inference Performance
Inference performance was measured for NVIDIA DGX SuperPOD (1 x 8 x A100 80GB).

Inference parameters:
* batch size: 1
* input tokens length: 60
* output tokens length: 20

<img src="img/infer_model_size_mt5.svg"/>

| mT5 Model size | Average latency [ms] | TP | PP | GPUs |
|----------------|----------------------|----|----|------|
| 380M           |                   35 |  1 |  1 |    1 |
| 3B             |                  102 |  2 |  1 |    2 |
| 11B            |                  134 |  4 |  1 |    4 |
| 23B            |                  230 |  4 |  1 |    4 |

### 7.4. BERT Results
<a id="markdown-bert-results" name="bert-results"></a>

#### 7.4.1. Training Accuracy Results
Training Accuracy: NVIDIA DGX SuperPOD (16 x 8 x A100 80GB for 4b Bert Model)

Training the 4B BERT model for 95 Billion takes 1.5 days, and the loss curve can be seen in the figure below:

<img src="img/4b_bert_loss_final.png"/>

The table below shows the converged training loss, the throughput, and the
total time to train for the 4B BERT model, using a given number of GPUs and a
given Global Batch Size (GBS).

| \#GPUs | GBS    | Seq Length | \#Tokens | Loss  | Throughput (Tokens/sec) | Time to Train (days) |
|--------|------|------------|----------|-------|-------------------------|----------------------|
| 16         | 2048 | 512                | 217B             | 1.44 | 728178               | 1.5                                        |


#### 7.4.2. Training Performance Results
<a id="markdown-training-performance-results" name="training-performance-results"></a>
Training Performance: NVIDIA DGX SuperPOD (20 x 8 x A100 80GB for 4B BERT Model)

We measured the throughput of training a 4B parameter BERT model on NVIDIA DGX
SuperPOD using a different number of nodes. When scaling from 1 node to 16 nodes, we achieve 12.71x
speed-up. 
The table and chart below show the performance results.


|         |                                    |        |         |         | Nodes   |         |
|---------|------------------------------------|--------|---------|---------|---------|---------|
|         |                                    | 1      | 2       | 4       | 8       | 16      | 
|         | Tokens per Second                  | 57287  | 108695  | 215358  | 393167  | 728178  | 
| 4B      | Perfect Linear Scaling (Tokens)    | 57287  | 114574  | 229148  | 458296  | 916592  | 
|         | Speed-up                           | 1x     | 1.89x   | 3.75x    | 6.86x   | 12.71x   | 


<img src="img/4B_bert_throughput_2211.png"/>

#### 7.4.3. Training Performance Results (LDDL)
<a id="markdown-training-performance-results-lddl" name="training-performance-results-lddl"></a>
We measured the performance of different Bert configurations with and without LDDL and saw an average 25% reduction in training time. 
The table and chart below show the performance results.

| Bert Config | Train time without LDDL | Trian time with LDDL | MODEL SPEC                 | TFLOPS w/o LDDL | TFLOPS(LDDL) | Speedup (%) |
| ----------- | ----------------------- | -------------------- | -------------------------- | --------------- | ------------ | ----------- |
| 110m        | 0.078                   | 0.076                | 8 Nodes TP1 PP1 GBS 256    | 18.280          | 18.900       | 2.63%       |
| 4b          | 1.794                   | 1.393                | 16 Nodes TP 1 PP1 GBS 2048 | 108.900         | 140.400      | 28.79%      |
| 20b         | 7.956                   | 6.79                 | 32 Nodes TP4 PP4 GBS 4096  | 137.300         | 160.870      | 17.17%      |
| 100b        | 9.743                   | 7.54                 | 128Nodes TP4 PP16 GBS 4096 | 124.88          | 162.83       | 29.22%      |

## 8. Changelog
<a id="markdown-changelog" name="changelog"></a>
**NeMo Framework 23.07**
* Add Low-Rank Adaptation (LoRA) Support for T5 and mT5
* Add Batch Size Ramp-up Support for GPT

**NeMo Framework 23.05**
* Low-Rank Adaptation (LoRA) Support for GPT
* LDDL (Language Datasets and Data Loaders) for BERT on 100B model resulting in a 30% performance speedup
* Unify dataset and model classes for all PEFT (p-tuning, adapters, IA3) with SFT model class as parent for GPT
* Converter from Interleaved PP to non-Interleaved PP
* Dialog dataset guidance for SFT to help create better chat models
* Support Dynamic Sequence Length Batches with GPT SFT
* Data parallelism enabled for RLHF servers, providing a 2x end-to-end speedup in most jobs

**NeMo Framework 23.04.1**
* Addressed issue in RLHF which prevented some jobs from running in Slurm clusters
* Corrections related to the renaming of NeMo Megatron to NeMo Framework
* Modified run.name in the *_improved configuration files to match the correct parameter count

**NeMo Framework 23.04**
* NeMo Data Curator - a scalable Python library for curating large-scale datasets required for training large language foundation models
* Enable Continued Training for P-Tuning
* Switch to Megatron Core for Model Parallelism in NeMo Framework
* Extend the Data Validation Tool to provide P-Tuning GPU Runtime Estimates
* Tensor and Pipeline Parallelism Conversion Support for GPT and T5
* Supervised Fine-Tuning Support for GPT
* RLHF (Reinforcement Learning from Human Feedback) for GPT
* New GPT model sizes - 400M_improved, 1B_improved, 7B_improved, 40B_improved based on new and improved model configurations
* List of GPT model configuration changes

| Configuration    | Previous | New |
| -------- | ------- | ------- |
| Activation  | GeLU    | Fast-SwiGLU |
| Position Embedding | Learned Absolute     | RoPE |
| Dropout    | 0.1    | 0 |
| Embeddings and Output Layer | Tied | Untied |
| Bias terms | Yes | No |
| Normalization | LayerNorm | LayerNorm1p |

**NeMo Framework 23.03**
* Per micro-batch data loader for GPT and BERT
* SquaredReLU and SwiGLU activation function support for GPT and T5
* Rotary Position Embedding (RoPE) for GPT and RETRO
* Early stopping support when P-Tuning/Prompt Tuning GPT, T5, and mT5
* Refactored Adapter learning implementation to mimic the Parameter-Efficient Transfer Learning for NLP approach
* Flash Attention for GPT models in Transformer Engine

**Announcement**

**Coming Soon!**  The data curation module, Prospector-LM, which is a scalable Python library for curating large-scale datasets and can be leveraged for training large language foundation models.

**NeMo Framework 23.01**
* BERT with tensor parallelism support (training only)
* BERT with pipeline parallelism support (training only)
* Sequence Parallelism and Selective Activation Checkpointing for BERT (training only)
* Interleaved Pipeline Scheduling for BERT
* Distributed Adam Optimizer for BERT
* AutoConfigurator for BERT
* 110M, 4B, 20B, and 100B BERT training configurations
* Support for the Mixture of Experts for T5 (no expert parallelism, training only)
* Performance improvement for GPT P-Tuning (20% - 25% speed-up)
* ALiBi Position Embeddings for T5 and mT5 (training only)
* Log total model size (across modal parallel ranks) for GPT, T5, mT5, and BERT

**NeMo Framework 22.11**
* Interleaved Pipeline Scheduling for GPT (training only)
* FP8 support using Transformer Engine (training only)
* Distributed Adam Optimizer for T5 and mT5
* P-Tuning and Prompt Tuning for GPT with Sequence Parallelism
* Training configurations improved throughput by 7.9% (5B GPT), 9.6% (3B T5), 4.3% (11B T5), 52.4% (23B T5), and 26.6% (41B T5) 

**NeMo Framework 22.09**
* NeMo Framework supports training and inference containers on OCI. For more details about orchestration scripts, reach out to [oci_nm@nvidia.com](mailto:oci_nm@nvidia.com)
* P-Tuning and Prompt Tuning for T5 and mT5 with pipeline parallelism (training only)
* Adapter learning for GPT and T5 with tensor parallelism and pipeline parallelism (training only)
* IA3 learning for GPT and T5 with tensor parallelism and pipeline parallelism (training only)
* AutoConfigurator to find the highest throughput configs for training on Base Command Platform
* AutoConfigurator: parallel inference hyperparameter search for GPT on Base Command Manager

**NeMo Framework 22.08.01**
* Cloud service providers: support for Amazon Web Services (performance validated up to 20 `p4d.24xlarge` instances)
* Cloud service providers: switched orchestration from Azure CycleCloud to NVIDIA Nephele for Microsoft Azure

**NeMo Framework 22.08**
* Distributed Adam Optimizer for GPT
* Asymmetric encoder-decoder configuration for T5 and mT5
* Support for untying embeddings from the classifier layer for T5 and mT5
* Relative Position Embeddings for T5 and mT5 (pipeline parallelism>=3)
* P-Tuning and Prompt Tuning for T5 and mT5 with tensor parallelism (training only)
* Code refactor - improved consistency and readability of configurations and logs
* SQuAD fine-tuning and evaluation support for T5 with pipeline parallelism =<2
* XQuAD fine-tuning and evaluation support for mT5 with pipeline parallelism =<2

**NeMo Framework 22.06-hotfix.01**
* Fix: AutoConfigurator for T5 and mT5
* Fix: Evaluation harness in GPT
* Fix: Prompt learning in GPT
* Fix: Out of memory when pretraining GPT with Sequence Parallelism

**NeMo Framework 22.06**
* Sequence Parallelism and Selective Activation Checkpointing for GPT
* Relative Position Embeddings for T5
  * We used mC4 dataset (24 Languages) for pretraining the mT5 and verified our results on KNLI, KorQuAD, KLUE-STS, and XNLI tasks
* AutoConfigurator update with Sequence Parallelism and Selective Activation Checkpointing for GPT
* AutoConfigurator: support for DGX A100 40GB configurations for GPT, T5, and mT5
* P-Tuning and Prompt Tuning for GPT with pipeline parallelism (training only)
* Operation fusions for higher training throughput (2%-7% speed-up)
* Default GPT configurations changed to include Sequence Parallelism and Selective Activation Checkpointing: 20B (speed-up: 14%), 40B (speed-up: 9%), 175B (speed-up: 15%) 

**NeMo Framework 22.05.01**
* Cloud service providers: support for Microsoft Azure (performance validated up to 36 `Standard_ND96amsr_A100_v4` instances)
* Cluster validation tools (DGMI, NCCL)
* 20B GPT training configuration improved by 2.7% for higher throughput

**NeMo Framework 22.05**
* Asynchronous gradient all-reduce for GPT, T5, mT5 models with pipeline parallel size equal to 1
* P-Tuning and Prompt Tuning for GPT with tensor parallelism (training only)
* AutoConfigurator to find the highest throughput configs for training and inference on Base Command Manager
* Custom tokenizer support (training only)
* GPT with pipeline parallelism support on Base Command Manager (inference)
* Hyperparameters for text generation: top-p, top-k, and temperature

**NeMo Framework 22.04**
* T5 with pipeline parallelism support (training only)
* Switched from GeLU to GeGLU as activation function for T5
* mT5 with tensor parallelism and pipeline parallelism support (training only)
* 11B, 23B, and 41B T5 training configurations
* 170M, 390M, and 3B mT5 training configurations
* Automatic and configurable Non-Uniform Memory Access (NUMA) mapping

**NeMo Framework 22.03**
* T5 with tensor parallelism support (optimized for <20B parameters, training only)
* 220M and 3B T5 training configurations
* GLUE fine-tuning and evaluation support for T5

**NeMo Framework 22.02**
* GPT with pipeline parallelism support (training only)
* 40B and 175B GPT training configurations

**NeMo Framework 22.01**
* GPT with tensor parallelism support on Base Command Platform
* O2-style AMP (accelerated training of larger models)
* Chatbot sample application using your trained GPT model
* Training metric monitoring and visualization with Weights & Biases

## 9. Known Issues
<a id="markdown-known-issues" name="known-issues"></a>
Fixes for the following issues will be released shortly:
* The inference hyperparameter search is not available in this release for T5 and mT5.
* Accuracy and performance measurement for GPT is currently not supported. Please use the NeMo Megatron 22.05 inference container to use this feature.
* The fine-tuning SQuAD results for T5 are lower than expected.
* Evaluation for GPT has been tested for PP <=2 and may have issues for PP >2. It is recommended to convert to TP only for Evaluation.
* Transformer Engine (TE)-based GPT models are currently not supported for any Parameter Efficient Fine Tuning (PEFT) techniques - this will be added soon.
* TE-based GPT Eval will take more memory than non-TE-based GPT Eval.
* Iteration per second metric is incorrectly displayed on the Logs progress bar - this will be addressed in the next release. Instead, please use train step timing in Weights & Biases or TensorBoard.
* Please note that Batch Size Ramp-up Support for GPT is currently working only with FusedAdam optimizer.
