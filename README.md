# NeMo Framework Launcher

The NeMo Framework Launcher is a cloud-native tool for launching end-to-end NeMo Framework training jobs.

See the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/index.html) for
the most up-to-date information and how to get started quickly.

The NeMo Framework focuses on foundation model training for generative AI models. 
Large language model (LLM) pretraining typically requires a lot of compute and model parallelism to efficiently scale training.
NeMo Framework includes the latest in large-scale training techniques including:

- Model parallelism
  * Tensor
  * Pipeline
  * Sequence
- Distributed Optimizer
- Mixed precision training
  * FP8
  * BF16
- Distributed Checkpointing
- Community Models
  * LLAMA-2

NeMo Framework model training scales to 1000's of GPUs and can be used for training LLMs on trillions of tokens.

The Launcher is designed to be a simple and easy to use tool for launching NeMo FW training jobs
on CSPs or on-prem clusters. The launcher is typically used from a head node and only requires
a minimal python installation.

The Launcher will generate and launch submission scripts for the cluster scheduler and will also organize 
and store jobs results. Tested configuration files are included with the launcher but anything
in a configuration file can be easily modified by the user.

The NeMo FW Launcher is tested with the [NeMo FW Container](https://registry.ngc.nvidia.com/orgs/ea-bignlp/teams/ga-participants/containers/nemofw-training) which can be applied for [here](https://developer.nvidia.com/nemo-framework).
Access is automatic. 
Users may also easily configure the launcher to use any container image that they want to provide.

The NeMo FW launcher supports:
- Cluster setup and configuration
- Data downloading, curating, and processing
- Model parallel configuration
- Model training
- Model fine-tuning (SFT and PEFT)
- Model evaluation
- Model export and deployment


Some of the models that we support include:
- GPT
  * Pretraining, Fine-tuning, SFT, PEFT
- BERT
- T5/MT5
  * PEFT, MoE (non-expert)

See the [Feature Matrix](https://docs.nvidia.com/nemo-framework/user-guide/latest/featurematrix.html#gpt-models) for more details.


## Installation

The NeMo Framework Launcher should be installed on a head node or a local machine in a virtual python environment.

```bash
git clone https://github.com/NVIDIA/NeMo-Megatron-Launcher.git
cd NeMo-Megatron-Launcher
pip install -r requirements.txt
```

## Usage

The best way to get started with the NeMo Framework Launcher is go through 
the [NeMo Framework Playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html)

After everything is configured in the `.yaml` files, the Launcher can be run with:

```bash
python main.py
```

Since the Launcher uses [Hydra](https://hydra.cc/docs/intro/), 
any configuration can be overridden directly in the `.yaml` file or via the command line.
See Hydra's [override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/) for more information. 

## Contributing

Contributions are welcome!

To contribute to the NeMo Framework Launcher, simply create a pull request with the changes on GitHub.
After the pull request is reviewed by a NeMo FW Developer, approved, and passes the unit and CI tests, 
then it will be merged.

## License

The NeMo Framework Launcher is licensed under the [Apache 2.0 License](https://github.com/NVIDIA/NeMo-Megatron-Launcher/blob/master/LICENSE)