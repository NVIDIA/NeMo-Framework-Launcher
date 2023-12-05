from omegaconf import OmegaConf


class TestConfig:
    def test_config(self):
        conf = OmegaConf.load("conf/config.yaml")
        s = """
        defaults:
          - _self_
          - cluster: bcm  # Set to bcm for BCM and BCP clusters. Set to k8s for a k8s cluster.
          - data_preparation: gpt3/download_gpt3_pile
          - quality_filtering: heuristic/english
          - training: gpt3/5b
          - conversion: gpt3/convert_gpt3
          - fine_tuning: null
          - peft: null
          - prompt_learning: null
          - adapter_learning: null
          - ia3_learning: null
          - evaluation: gpt3/evaluate_all
          - export: gpt3/export_gpt3
          - rlhf_rm: gpt3/2b_rm
          - rlhf_ppo: gpt3/2b_ppo
          - override hydra/job_logging: stdout
        
        hydra:
          run:
            dir: .
          output_subdir: null
        
        debug: False
        
        stages:
          #- data_preparation
          #- training
          - conversion
          #- prompt_learning
          #- adapter_learning
          #- ia3_learning
          #- evaluation
          #- export
        
        cluster_type: bcm  # bcm or bcp. If bcm, it must match - cluster above.
        launcher_scripts_path: ???  # Path to NeMo Megatron Launch scripts, should ends with /launcher_scripts
        data_dir: ${launcher_scripts_path}/data  # Location to store and read the data.
        base_results_dir: ${launcher_scripts_path}/results  # Location to store the results, checkpoints and logs.
        container_mounts: # List of additional paths to mount to container. They will be mounted to same path.
            - null
        container: nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.11
        
        wandb_api_key_file: null  # File where the w&B api key is stored. Key must be on the first line.
        wandb_api_bcp_secret_key: null  # For BCP clusters, read the W&B api key directly from the environment variable set as a secret from BCP. The value must match the name of the environment variable in BCP, such as WANDB_TOKEN.

        bcp_no_redirect: True  # If True, all stdout and stderr will not be redirected and appear in the standard logs. If False, all stdout and stderr output will be redirected to individual files on a per-rank basis. Ignored for non-BCP clusters.
        
        env_vars:
          NCCL_TOPO_FILE: null # Should be a path to an XML file describing the topology
          UCX_IB_PCI_RELAXED_ORDERING: null # Needed to improve Azure performance
          NCCL_IB_PCI_RELAXED_ORDERING: null # Needed to improve Azure performance
          NCCL_IB_TIMEOUT: null # InfiniBand Verbs Timeout. Set to 22 for Azure
          NCCL_DEBUG: null # Logging level for NCCL. Set to "INFO" for debug information
          NCCL_PROTO: null # Protocol NCCL will use. Set to "simple" for AWS
          TRANSFORMERS_OFFLINE: 1
          TORCH_NCCL_AVOID_RECORD_STREAMS: 1
          NCCL_NVLS_ENABLE: 0
        
        # GPU Mapping
        numa_mapping:
          enable: True  # Set to False to disable all mapping (performance will suffer).
          mode: unique_contiguous  # One of: all, single, single_unique, unique_interleaved or unique_contiguous.
          scope: node  # Either node or socket.
          cores: all_logical  # Either all_logical or single_logical.
          balanced: True  # Whether to assing an equal number of physical cores to each process.
          min_cores: 1  # Minimum number of physical cores per process.
          max_cores: 8  # Maximum number of physical cores per process. Can be null to use all available cores.
        
        # Do not modify below, use the values above instead.
        data_preparation_config: ${hydra:runtime.choices.data_preparation}
        quality_filtering_config: ${hydra:runtime.choices.quality_filtering}
        training_config: ${hydra:runtime.choices.training}
        fine_tuning_config: ${hydra:runtime.choices.fine_tuning}
        peft_config: ${hydra:runtime.choices.peft}
        prompt_learning_config: ${hydra:runtime.choices.prompt_learning}
        adapter_learning_config: ${hydra:runtime.choices.adapter_learning}
        ia3_learning_config: ${hydra:runtime.choices.ia3_learning}
        evaluation_config: ${hydra:runtime.choices.evaluation}
        conversion_config: ${hydra:runtime.choices.conversion}
        export_config: ${hydra:runtime.choices.export}
        rlhf_rm_config: ${hydra:runtime.choices.rlhf_rm}
        rlhf_ppo_config: ${hydra:runtime.choices.rlhf_ppo}
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"conf/config.yaml must be set to {expected} but it currently is {conf}."
