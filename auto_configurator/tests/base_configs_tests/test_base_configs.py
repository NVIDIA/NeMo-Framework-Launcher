from omegaconf import OmegaConf


class TestBaseConfigs:
    def test_gpt3_base_config(self):
        conf = OmegaConf.load("base_configs/gpt3.yaml")
        s = """
        run:
          name: gpt3_126m
          results_dir: ${base_results_dir}/${.name}
          time_limit: "1-00:00:00"
          dependency: "singleton"

        trainer:
          num_nodes: 8
          devices: 8
          accelerator: gpu
          precision: bf16
          logger: False
          enable_checkpointing: False
          use_distributed_sampler: False
          max_epochs: null
          max_steps: 600000
          max_time: "00:23:30:00"
          log_every_n_steps: 1
          val_check_interval: 50
          limit_val_batches: 1
          limit_test_batches: 1
          accumulate_grad_batches: 1
          gradient_clip_val: 1.0


        exp_manager:
          explicit_log_dir: ${training.run.results_dir}/results
          exp_dir: null
          name: megatron_gpt
          create_wandb_logger: False
          wandb_logger_kwargs:
            project: nemo_gpt3
            name: ${training.run.name}
          resume_if_exists: True
          resume_ignore_no_checkpoint: True
          create_checkpoint_callback: True
          checkpoint_callback_params:
            monitor: val_loss
            save_top_k: 10
            mode: min
            always_save_nemo: False
            save_nemo_on_train_end: False
            filename: 'megatron_gpt--{val_loss:.2f}-{step}-{consumed_samples}'
            model_parallel_size: ${multiply:${training.model.tensor_model_parallel_size}, ${training.model.pipeline_model_parallel_size}}
          log_step_timing: True
          step_timing_kwargs:
            sync_cuda: True
            buffer_size: 5

        model:
          # model parallelism
          micro_batch_size: 4
          global_batch_size: 256
          tensor_model_parallel_size: 1
          pipeline_model_parallel_size: 1
          virtual_pipeline_model_parallel_size: null
          resume_from_checkpoint: null

          # model architecture
          encoder_seq_length: 2048
          max_position_embeddings: 2048
          num_layers: 12
          hidden_size: 768
          ffn_hidden_size: ${multiply:4, ${.hidden_size}}
          num_attention_heads: 12
          init_method_std: 0.023
          hidden_dropout: 0.1
          kv_channels: null
          apply_query_key_layer_scaling: True
          layernorm_epsilon: 1e-5
          make_vocab_size_divisible_by: 128
          pre_process: True
          post_process: True
          persist_layer_norm: True
          gradient_as_bucket_view: True
          sync_batch_comm: False
  
          # Fusion
          grad_div_ar_fusion: True # Fuse grad division into torch.distributed.all_reduce
          gradient_accumulation_fusion: True # Fuse weight gradient accumulation to GEMMs
          bias_activation_fusion: True # Use a kernel that fuses the bias addition from weight matrices with the subsequent activation function.
          bias_dropout_add_fusion: True # Use a kernel that fuses the bias addition, dropout and residual connection addition.
          masked_softmax_fusion: True # Use a kernel that fuses the attention softmax with it's mask.

          activations_checkpoint_granularity: null
          activations_checkpoint_method: null
          activations_checkpoint_num_layers: null
          num_micro_batches_with_partial_activation_checkpoints: null
          activations_checkpoint_layers_per_pipeline: null

          sequence_parallel: True

          tokenizer:
            library: 'megatron'
            type: 'GPT2BPETokenizer'
            model: null
            delimiter: null
            vocab_file: ${data_dir}/bpe/vocab.json
            merge_file: ${data_dir}/bpe/merges.txt

          # precision
          native_amp_init_scale: 4294967296
          native_amp_growth_interval: 1000
          hysteresis: 2
          fp32_residual_connection: False
          fp16_lm_cross_entropy: False

          # Megatron O2-style half-precision
          megatron_amp_O2: True
          grad_allreduce_chunk_size_mb: 125

          ## Using Megatron Core
          mcore_gpt: True

          ## Transformer Engine
          transformer_engine: False
          fp8: False # enables fp8 in TransformerLayer forward
          fp8_e4m3: False # sets fp8_format = recipe.Format.E4M3
          fp8_hybrid: True # sets fp8_format = recipe.Format.HYBRID
          fp8_margin: 0 # scaling margin
          fp8_interval: 1 # scaling update interval
          fp8_amax_history_len: 1024 # Number of steps for which amax history is recorded per tensor
          fp8_amax_compute_algo: max # 'most_recent' or 'max'. Algorithm for computing amax from history
          fp8_wgrad: True
          ub_tp_comm_overlap: False

          # miscellaneous
          seed: 1234
          use_cpu_initialization: False
          onnx_safe: False
          apex_transformer_log_level: 30

          # Nsys profiling options
          nsys_profile:
            enabled: False
            trace: [nvtx,cuda]
            start_step: 10  # Global batch to start profiling
            end_step: 10 # Global batch to end profiling
            ranks: [0] # Global rank IDs to profile
            gen_shape: False # Generate model and kernel details including input shapes

          optim:
            name: distributed_fused_adam
            overlap_grad_sync: False
            bucket_cap_mb: ${training.model.grad_allreduce_chunk_size_mb}
            lr: 6e-4
            weight_decay: 0.1
            betas:
            - 0.9
            - 0.95
            sched:
              name: CosineAnnealing
              warmup_steps: 636
              constant_steps: 100000
              min_lr: 6e-5

          data:
            data_impl: mmap
            splits_string: "99990,8,2"
            seq_length: 2048
            skip_warmup: True
            num_workers: 2
            dataloader_type: single
            reset_position_ids: False
            reset_attention_mask: False
            eod_mask_loss: False
            index_mapping_dir: null
            data_prefix:
              - 1.0
              - ${data_dir}/my-gpt3_00_text_document
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"base_configs/gpt3.yaml must be set to {expected} but it currently is {conf}."

    def test_t5_base_config(self):
        conf = OmegaConf.load("base_configs/t5.yaml")
        s = """
        run:
          name: t5_220m
          results_dir: ${base_results_dir}/${.name}
          time_limit: "7-00:00:00"
          dependency: "singleton"

        name: megatron_t5
        restore_from_path: null # used when starting from a .nemo file

        trainer:
          num_nodes: 4
          devices: 8
          accelerator: gpu
          precision: bf16
          logger: False # logger provided by exp_manager
          enable_checkpointing: False
          use_distributed_sampler: False
          max_epochs: null
          max_steps: 1000000 # consumed_samples = global_step * global_batch_size
          max_time: "06:23:30:00"
          log_every_n_steps: 1
          val_check_interval: 50
          limit_val_batches: 1
          limit_test_batches: 1
          accumulate_grad_batches: 1
          gradient_clip_val: 1.0


        exp_manager:
          explicit_log_dir: ${training.run.results_dir}/results
          exp_dir: null
          name: megatron_t5
          create_wandb_logger: False
          wandb_logger_kwargs:
            project: nemo_t5
            name: ${training.run.name}
          resume_if_exists: True
          resume_ignore_no_checkpoint: True
          create_checkpoint_callback: True
          checkpoint_callback_params:
            monitor: val_loss
            save_top_k: 10
            mode: min
            always_save_nemo: False # saves nemo file during validation, not implemented for model parallel
            save_nemo_on_train_end: False # not recommended when training large models on clusters with short time limits
            filename: 'megatron_t5--{val_loss:.2f}-{step}-{consumed_samples}'
            model_parallel_size: ${multiply:${training.model.tensor_model_parallel_size}, ${training.model.pipeline_model_parallel_size}}
          log_step_timing: True
          step_timing_kwargs:
            sync_cuda: True
            buffer_size: 5

        model:
          # model parallelism
          micro_batch_size: 64
          global_batch_size: 2048 # will use more micro batches to reach global batch size
          tensor_model_parallel_size: 1
          pipeline_model_parallel_size: 1
          resume_from_checkpoint: null # manually set the checkpoint file to load from
          pipeline_model_parallel_split_rank: ${divide_floor:${.pipeline_model_parallel_size}, 2}

          # model architecture
          make_vocab_size_divisible_by: 128 # Pad the vocab size to be divisible by this value for computation efficiency.
          pre_process: True # add embedding
          post_process: True # add pooler

          megatron_amp_O2: True # use AMP with O2 style mixed precision instead of native amp on-the-fly weight autocasting.
          grad_allreduce_chunk_size_mb: 125
          gradient_as_bucket_view: True # Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
          sync_batch_comm: False

          seq_length: 512
          max_position_embeddings: ${.seq_length}

          encoder:
            num_layers: 12
            hidden_size: 768
            ffn_hidden_size: 2048  # Transformer FFN hidden size. 4 * hidden_size.
            num_attention_heads: 12
            kv_channels: 64  # Projection weights dimension in multi-head attention. Set to hidden_size // num_attention_heads if null
            init_method_std: 0.015  # Standard deviation of the zero mean normal distribution used for weight initialization.')
            hidden_dropout: 0.1  # Dropout probability for hidden state transformer.
            attention_dropout: 0.1 # Dropout probability in the attention layer.
            position_embedding_type: 'learned_absolute' # Position embedding type. Options ['learned_absolute', 'relative']
            relative_attention_num_buckets: 32 # Relative position number of buckets for computing the bias
            relative_attention_max_distance: 128 # max_distance to keep relative distance in the attention_num_buckets.
            relative_position_bias_self_attention_only: True # Whether to only use relative position bias for self attention only.
            apply_query_key_layer_scaling: True # scale Q * K^T by 1 / layer-number.
            layernorm_epsilon: 1e-5
            persist_layer_norm: True # Use of persistent fused layer norm kernel.
            bias_activation_fusion: True # Use a kernel that fuses the bias addition from weight matrices with the subsequent activation function.
            grad_div_ar_fusion: True # Fuse grad division into torch.distributed.all_reduce
            masked_softmax_fusion: True # Use a kernel that fuses the attention softmax with it's mask.
            bias_dropout_add_fusion: True # Use a kernel that fuses the bias addition, dropout and residual connection addition.
            bias: True # Whether to use bias terms in all weight matrices.
            normalization: 'layernorm' # Normalization layer to use. Options are 'layernorm', 'rmsnorm'
            arch: 'transformer' # Options: ['transformer', 'perceiver']
            activation: 'geglu' # Options ['gelu', 'geglu', 'swiglu', 'reglu']
            headscale: False # Whether to learn extra parameters that scale the output of the each self-attention head.
            transformer_block_type: 'pre_ln' # Options ['pre_ln', 'post_ln', 'normformer']
            openai_gelu: False # Use OpenAI's GELU instead of the default GeLU
            # miscellaneous
            onnx_safe: False # Use work-arounds for known problems with Torch ONNX exporter.
            fp32_residual_connection: False # Use FP32 for residual connections.
            # activations checkpointing
            activations_checkpoint_granularity: full
            activations_checkpoint_method: block # 'uniform', 'block'
            activations_checkpoint_num_layers: 0

          decoder:
            num_layers: 12
            hidden_size: 768
            ffn_hidden_size: 2048  # Transformer FFN hidden size. 4 * hidden_size.
            num_attention_heads: 12
            kv_channels: 64  # Projection weights dimension in multi-head attention. Set to hidden_size // num_attention_heads if null
            init_method_std: 0.015 # Standard deviation of the zero mean normal distribution used for weight initialization.')
            hidden_dropout: 0.1 # Dropout probability for hidden state transformer.
            attention_dropout: 0.1 # Dropout probability in the attention layer.
            position_embedding_type: 'learned_absolute' # Position embedding type. Options ['learned_absolute', 'relative']
            relative_attention_num_buckets: 32 # Relative position number of buckets for computing the bias
            relative_attention_max_distance: 128 # max_distance to keep relative distance in the attention_num_buckets.
            relative_position_bias_self_attention_only: True # Whether to only use relative position bias for self attention only.
            apply_query_key_layer_scaling: True # scale Q * K^T by 1 / layer-number.
            layernorm_epsilon: 1e-5
            persist_layer_norm: True # Use of persistent fused layer norm kernel.
            bias_activation_fusion: True # Use a kernel that fuses the bias addition from weight matrices with the subsequent activation function.
            grad_div_ar_fusion: True # Fuse grad division into torch.distributed.all_reduce
            masked_softmax_fusion: True # Use a kernel that fuses the attention softmax with it's mask.
            bias_dropout_add_fusion: True # Use a kernel that fuses the bias addition, dropout and residual connection addition.
            bias: True # Whether to use bias terms in all weight matrices.
            normalization: 'layernorm' # Normalization layer to use. Options are 'layernorm', 'rmsnorm'
            arch: 'transformer' # Options: ['transformer', 'perceiver']
            activation: 'geglu' # Options ['gelu', 'geglu', 'swiglu', 'reglu']
            headscale: False # Whether to learn extra parameters that scale the output of the each self-attention head.
            transformer_block_type: 'pre_ln' # Options ['pre_ln', 'post_ln', 'normformer']
            openai_gelu: False # Use OpenAI's GELU instead of the default GeLU
            # miscellaneous
            onnx_safe: False # Use work-arounds for known problems with Torch ONNX exporter.
            fp32_residual_connection: False # Use FP32 for residual connections.
            # activations checkpointing
            activations_checkpoint_granularity: full
            activations_checkpoint_method: block # 'uniform', 'block'
            activations_checkpoint_num_layers: 0

          tokenizer:
            library: 'megatron'
            type: 'BertWordPieceCase'
            model: null
            vocab_file: ${data_dir}/bpe/vocab.txt
            merge_file: null
            num_sentinel_tokens: 100

          # precision
          native_amp_init_scale: 4294967296 # 2 ** 32
          native_amp_growth_interval: 1000
          fp16_lm_cross_entropy: False # Move the cross entropy unreduced loss calculation for lm head to fp16

          # miscellaneous
          seed: 1234
          use_cpu_initialization: False # Init weights on the CPU (slow for large models)
          apex_transformer_log_level: 30 # Python logging level displays logs with severity greater than or equal to this


          # embedding sharing
          share_token_embeddings: True # If True share encoder/decoder embeddings
          share_decoder_tokens_head_embeddings: True # If True share decoder embeddings and decoder projection to logits

          nsys_profile:
            enabled: False
            trace: [nvtx,cuda]
            start_step: 10  # Global batch to start profiling
            end_step: 10 # Global batch to end profiling
            ranks: [0] # Global rank IDs to profile
            gen_shape: False # Generate model and kernel details including input shapes

          optim:
            name: distributed_fused_adam
            overlap_grad_sync: False
            bucket_cap_mb: ${training.model.grad_allreduce_chunk_size_mb}
            lr: 0.0001
            betas:
              - 0.9
              - 0.999
            eps: 1e-8
            weight_decay: 0.01
            sched:
              name: WarmupAnnealing
              min_lr: 0.00001
              last_epoch: -1
              warmup_ratio: 0.01


          data:
            data_impl: mmap
            splits_string: "90,5,5"
            seq_length: 512
            seq_length_dec: 128
            skip_warmup: True
            num_workers: 4
            dataloader_type: single # cyclic
            masked_lm_prob: 0.15
            dataset_type: 't5'
            short_seq_prob: 0.0
            max_ngram_size: 10
            mean_ngram_size: null
            geometric_dist: True
            permutation: False
            whole_word_masking: True
            favor_longer_ngrams: False
            respect_document_boundaries: True # If true, a single training exampl cannot cross document boundaries, increasing the fraction of <pad> tokens within a batch.
            index_mapping_dir: null # path to save index mapping .npy files, by default will save in the same location as data_prefix
            data_prefix: # Should be weight path weight path... for a blended dataset
              - 1.0
              - ${data_dir}/my-t5_00_text_document
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"base_configs/t5.yaml must be set to {expected} but it currently is {conf}."

    def test_mt5_base_config(self):
        conf = OmegaConf.load("base_configs/mt5.yaml")
        s = """
        run:
          name: mt5_170m
          results_dir: ${base_results_dir}/${.name}
          time_limit: "7-00:00:00"
          dependency: "singleton"
          preprocessed_dir: ${data_dir}/mc4/preprocessed # used for auto data blending
          blending_alpha: 0.7 # blending ratio across different languages; language sampling ratio ~L^alpha

        name: megatron_mt5
        restore_from_path: null # used when starting from a .nemo file

        trainer:
          num_nodes: 4
          devices: 8
          accelerator: gpu
          precision: bf16
          logger: False # logger provided by exp_manager
          enable_checkpointing: False
          use_distributed_sampler: False
          max_epochs: null
          max_steps: 1000000 # consumed_samples = global_step * global_batch_size
          max_time: "06:23:30:00"
          log_every_n_steps: 1
          val_check_interval: 50
          limit_val_batches: 1
          limit_test_batches: 1
          accumulate_grad_batches: 1
          gradient_clip_val: 1.0


        exp_manager:
          explicit_log_dir: ${training.run.results_dir}/results
          exp_dir: null
          name: megatron_mt5
          create_wandb_logger: False
          wandb_logger_kwargs:
            project: nemo_mt5
            name: ${training.run.name}
          resume_if_exists: True
          resume_ignore_no_checkpoint: True
          create_checkpoint_callback: True
          checkpoint_callback_params:
            monitor: val_loss
            save_top_k: 10
            mode: min
            always_save_nemo: False # saves nemo file during validation, not implemented for model parallel
            save_nemo_on_train_end: False # not recommended when training large models on clusters with short time limits
            filename: 'megatron_mt5--{val_loss:.2f}-{step}-{consumed_samples}'
            model_parallel_size: ${multiply:${training.model.tensor_model_parallel_size}, ${training.model.pipeline_model_parallel_size}}
          log_step_timing: True
          step_timing_kwargs:
            sync_cuda: True
            buffer_size: 5

        model:
          # model parallelism
          micro_batch_size: 64
          global_batch_size: 2048 # will use more micro batches to reach global batch size
          tensor_model_parallel_size: 1
          pipeline_model_parallel_size: 1
          resume_from_checkpoint: null # manually set the checkpoint file to load from
          pipeline_model_parallel_split_rank: ${divide_floor:${.pipeline_model_parallel_size}, 2}

          # model architecture
          make_vocab_size_divisible_by: 128 # Pad the vocab size to be divisible by this value for computation efficiency.
          pre_process: True # add embedding
          post_process: True # add pooler

          megatron_amp_O2: True # use AMP with O2 style mixed precision instead of native amp on-the-fly weight autocasting.
          grad_allreduce_chunk_size_mb: 125
          gradient_as_bucket_view: True # Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
          sync_batch_comm: False

          seq_length: 512
          max_position_embeddings: ${.seq_length}

          encoder:
            num_layers: 8
            hidden_size: 512
            ffn_hidden_size: 1024  # Transformer FFN hidden size. 4 * hidden_size.
            num_attention_heads: 6
            kv_channels: 64  # Projection weights dimension in multi-head attention. Set to hidden_size // num_attention_heads if null
            init_method_std: 0.015  # Standard deviation of the zero mean normal distribution used for weight initialization.')
            hidden_dropout: 0.1  # Dropout probability for hidden state transformer.
            attention_dropout: 0.1 # Dropout probability in the attention layer.
            position_embedding_type: 'learned_absolute' # Position embedding type. Options ['learned_absolute', 'relative']
            relative_attention_num_buckets: 32 # Relative position number of buckets for computing the bias
            relative_attention_max_distance: 128 # max_distance to keep relative distance in the attention_num_buckets.
            relative_position_bias_self_attention_only: True # Whether to only use relative position bias for self attention only.
            apply_query_key_layer_scaling: True # scale Q * K^T by 1 / layer-number.
            layernorm_epsilon: 1e-5
            persist_layer_norm: True # Use of persistent fused layer norm kernel.
            bias_activation_fusion: True # Use a kernel that fuses the bias addition from weight matrices with the subsequent activation function.
            grad_div_ar_fusion: True # Fuse grad division into torch.distributed.all_reduce
            masked_softmax_fusion: True # Use a kernel that fuses the attention softmax with it's mask.
            bias_dropout_add_fusion: True # Use a kernel that fuses the bias addition, dropout and residual connection addition.
            bias: True # Whether to use bias terms in all weight matrices.
            normalization: 'layernorm' # Normalization layer to use. Options are 'layernorm', 'rmsnorm'
            arch: 'transformer' # Options: ['transformer', 'perceiver']
            activation: 'geglu' # Options ['gelu', 'geglu', 'swiglu', 'reglu']
            headscale: False # Whether to learn extra parameters that scale the output of the each self-attention head.
            transformer_block_type: 'pre_ln' # Options ['pre_ln', 'post_ln', 'normformer']
            openai_gelu: False # Use OpenAI's GELU instead of the default GeLU
            # miscellaneous
            onnx_safe: False # Use work-arounds for known problems with Torch ONNX exporter.
            fp32_residual_connection: False # Use FP32 for residual connections.
            # activations checkpointing
            activations_checkpoint_granularity: full
            activations_checkpoint_method: block # 'uniform', 'block'
            activations_checkpoint_num_layers: 0

          decoder:
            num_layers: 8
            hidden_size: 512
            ffn_hidden_size: 1024  # Transformer FFN hidden size. 4 * hidden_size.
            num_attention_heads: 6
            kv_channels: 64  # Projection weights dimension in multi-head attention. Set to hidden_size // num_attention_heads if null
            init_method_std: 0.015 # Standard deviation of the zero mean normal distribution used for weight initialization.')
            hidden_dropout: 0.1 # Dropout probability for hidden state transformer.
            attention_dropout: 0.1 # Dropout probability in the attention layer.
            position_embedding_type: 'learned_absolute' # Position embedding type. Options ['learned_absolute', 'relative']
            relative_attention_num_buckets: 32 # Relative position number of buckets for computing the bias
            relative_attention_max_distance: 128 # max_distance to keep relative distance in the attention_num_buckets.
            relative_position_bias_self_attention_only: True # Whether to only use relative position bias for self attention only.
            apply_query_key_layer_scaling: True # scale Q * K^T by 1 / layer-number.
            layernorm_epsilon: 1e-5
            persist_layer_norm: True # Use of persistent fused layer norm kernel.
            bias_activation_fusion: True # Use a kernel that fuses the bias addition from weight matrices with the subsequent activation function.
            grad_div_ar_fusion: True # Fuse grad division into torch.distributed.all_reduce
            masked_softmax_fusion: True # Use a kernel that fuses the attention softmax with it's mask.
            bias_dropout_add_fusion: True # Use a kernel that fuses the bias addition, dropout and residual connection addition.
            bias: True # Whether to use bias terms in all weight matrices.
            normalization: 'layernorm' # Normalization layer to use. Options are 'layernorm', 'rmsnorm'
            arch: 'transformer' # Options: ['transformer', 'perceiver']
            activation: 'geglu' # Options ['gelu', 'geglu', 'swiglu', 'reglu']
            headscale: False # Whether to learn extra parameters that scale the output of the each self-attention head.
            transformer_block_type: 'pre_ln' # Options ['pre_ln', 'post_ln', 'normformer']
            openai_gelu: False # Use OpenAI's GELU instead of the default GeLU
            # miscellaneous
            onnx_safe: False # Use work-arounds for known problems with Torch ONNX exporter.
            fp32_residual_connection: False # Use FP32 for residual connections.
            # activations checkpointing
            activations_checkpoint_granularity: full
            activations_checkpoint_method: block # 'uniform', 'block'
            activations_checkpoint_num_layers: 0

          tokenizer:
            library: 'sentencepiece'
            type: null
            model: ${data_dir}/mc4/bpe/mt5_tokenizer.model
            vocab_file: null
            merge_file: null
            num_sentinel_tokens: 100

          # precision
          native_amp_init_scale: 4294967296 # 2 ** 32
          native_amp_growth_interval: 1000
          fp16_lm_cross_entropy: False # Move the cross entropy unreduced loss calculation for lm head to fp16

          # miscellaneous
          seed: 1234
          use_cpu_initialization: False # Init weights on the CPU (slow for large models)
          apex_transformer_log_level: 30 # Python logging level displays logs with severity greater than or equal to this


          # embedding sharing
          share_token_embeddings: True # If True share encoder/decoder embeddings
          share_decoder_tokens_head_embeddings: True # If True share decoder embeddings and decoder projection to logits

          nsys_profile:
            enabled: False
            trace: [nvtx,cuda]
            start_step: 10  # Global batch to start profiling
            end_step: 10 # Global batch to end profiling
            ranks: [0] # Global rank IDs to profile
            gen_shape: False # Generate model and kernel details including input shapes

          optim:
            name: distributed_fused_adam
            overlap_grad_sync: False
            bucket_cap_mb: ${training.model.grad_allreduce_chunk_size_mb}
            lr: 0.0001
            betas:
              - 0.9
              - 0.999
            eps: 1e-8
            weight_decay: 0.01
            sched:
              name: WarmupAnnealing
              min_lr: 0.00001
              last_epoch: -1
              warmup_ratio: 0.01


          data:
            data_impl: mmap
            splits_string: "90,5,5"
            seq_length: 512
            seq_length_dec: 128
            skip_warmup: True
            num_workers: 8
            dataloader_type: single # cyclic
            masked_lm_prob: 0.15
            dataset_type: 't5'
            short_seq_prob: 0.0
            max_ngram_size: 10
            mean_ngram_size: null
            geometric_dist: True
            permutation: False
            whole_word_masking: False
            favor_longer_ngrams: False
            respect_document_boundaries: True # If true, a single training exampl cannot cross document boundaries, increasing the fraction of <pad> tokens within a batch.
            index_mapping_dir: null # path to save index mapping .npy files, by default will save in the same location as data_prefix
            data_prefix: null # Should be weight path weight path... for a blended dataset. If null will automatically blend all language files in mC4_dir.
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"base_configs/mt5.yaml must be set to {expected} but it currently is {conf}."

    def test_bert_base_config(self):
        conf = OmegaConf.load("base_configs/bert.yaml")
        s = """
        run:
          name: bert_110m
          results_dir: ${base_results_dir}/${.name}
          time_limit:  "1-00:00:00"
          dependency: "singleton"

        name: megatron_bert
        restore_from_path: null # used when starting from a .nemo file

        trainer:
          devices: 8
          num_nodes: 8
          accelerator: gpu
          precision: bf16
          logger: False # logger provided by exp_manager
          enable_checkpointing: False
          use_distributed_sampler: False
          max_epochs: -1 # PTL default. In practice we don't usually train for more than 1 epoch.
          max_steps: 100000 # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches
          max_time: "00:23:30:00"
          log_every_n_steps: 1
          val_check_interval: 50
          limit_val_batches: 1
          limit_test_batches: 1
          accumulate_grad_batches: 1
          gradient_clip_val: 1.0

        exp_manager:
          explicit_log_dir: ${training.run.results_dir}/results
          exp_dir: null
          name: megatron_bert
          create_wandb_logger: False
          wandb_logger_kwargs:
            project: nemo_bert
            name: ${training.run.name}
          resume_if_exists: True
          resume_ignore_no_checkpoint: True
          create_checkpoint_callback: True
          checkpoint_callback_params:
            monitor: val_loss
            save_top_k: 10
            mode: min
            always_save_nemo: False # saves nemo file during validation, not implemented for model parallel
            filename: 'megatron_bert--{val_loss:.2f}-{step}-{consumed_samples}'
            model_parallel_size: ${multiply:${training.model.tensor_model_parallel_size}, ${training.model.pipeline_model_parallel_size}}
          log_step_timing: True
          step_timing_kwargs:
            sync_cuda: True
            buffer_size: 5

        model:
          # model parallelism 
          global_batch_size: 256
          micro_batch_size: 4
          tensor_model_parallel_size: 1
          pipeline_model_parallel_size: 1
          virtual_pipeline_model_parallel_size: null # interleaved pipeline

          # model architecture
          encoder_seq_length: 512
          max_position_embeddings: ${.encoder_seq_length}
          num_layers: 12
          hidden_size: 768
          ffn_hidden_size: ${multiply:4, ${.hidden_size}} #3072 # Transformer FFN hidden size. Usually 4 * hidden_size.
          num_attention_heads: 12
          init_method_std: 0.02 # Standard deviation of the zero mean normal distribution used for weight initialization.')
          hidden_dropout: 0.1 # Dropout probability for hidden state transformer.
          kv_channels: null # Projection weights dimension in multi-head attention. Set to hidden_size // num_attention_heads if null
          apply_query_key_layer_scaling: True # scale Q * K^T by 1 / layer-number.
          layernorm_epsilon: 1e-5
          make_vocab_size_divisible_by: 128 # Pad the vocab size to be divisible by this value for computation efficiency.
          pre_process: True # add embedding
          post_process: True # add pooler
          bert_binary_head: False # BERT binary head

          tokenizer:
            library: 'megatron'
            type: 'BertWordPieceLowerCase'
            model: null
            vocab_file: ${data_dir}/vocab.txt  
            merge_file: null 

          # precision
          native_amp_init_scale: 4294967296 # 2 ** 32
          native_amp_growth_interval: 1000
          fp32_residual_connection: False # Move residual connections to fp32
          fp16_lm_cross_entropy: False # Move the cross entropy unreduced loss calculation for lm head to fp16

          # Megatron O2-style half-precision
          megatron_amp_O2: True # Enable O2-level automatic mixed precision using main parameters
          grad_allreduce_chunk_size_mb: 125
          grad_div_ar_fusion: False

          # miscellaneous
          seed: 1234
          use_cpu_initialization: False # Init weights on the CPU (slow for large models)
          onnx_safe: False # Use work-arounds for known problems with Torch ONNX exporter.
          gradient_as_bucket_view: True # PyTorch DDP argument. Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)

          # Activations checkpointing
          activations_checkpoint_granularity: selective
          activations_checkpoint_method: block # 'uniform', 'block'
          activations_checkpoint_num_layers: 1
          num_micro_batches_with_partial_activation_checkpoints: null
          activations_checkpoint_layers_per_pipeline: null
          sequence_parallel: True

          data:
            data_prefix:
              - 1.0
              - ${data_dir}/my-t5_00_bert_tokenizer_text_document
            index_mapping_dir: null # path to save index mapping .npy files, by default will save in the same location as data_prefix
            data_impl: mmap
            splits_string: 900,50,50
            seq_length: 512 #${model.encoder_seq_length}
            skip_warmup: True
            num_workers: 2
            dataloader_type: single # cyclic
            reset_position_ids: False # Reset position ids after end-of-document token
            reset_attention_mask: False # Reset attention mask after end-of-document token
            eod_mask_loss: False # Mask loss for the end of document tokens
            masked_lm_prob: 0.15 # Probability of replacing a token with mask.
            short_seq_prob: 0.1 # Probability of producing a short sequence.
  
          optim:
            name: distributed_fused_adam
            overlap_grad_sync: False
            bucket_cap_mb: ${training.model.grad_allreduce_chunk_size_mb}
            lr: 2e-4
            weight_decay: 0.01 
            betas: 
              - 0.9
              - 0.98
            sched:
              name: CosineAnnealing
              warmup_steps: 500
              constant_steps: 50000
              min_lr: 2e-5
        """
        expected = OmegaConf.create(s)
        assert (
            expected == conf
        ), f"base_configs/bert.yaml must be set to {expected} but it currently is {conf}."
