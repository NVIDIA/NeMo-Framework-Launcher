

class Training(NeMoStage):
    """Stage class of pretraining with NeMo scripts"""

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "training"
        self.stage_cfg = cfg.get("training")

    def _make_hydra_override(self) -> List:
        """
        Override some existing hydra configurations if necessary.
        Example use cases are:
            1. For bcp cluster, `+rank=\${RANK}` is required running some NeMo scripts.
                Existing hydra config doesn't have `rank` field, so we overwrite on the fly.
            2. Auto blend training dataset by overwriting empty `model.data.data_prefix` as
                `model.data.data_prefix=\$({auto_blend_command})`. Existing `model.data.data_prefix`
                could be None in cfg, so we overwrite it in this function.

        :return: hydra override string added in nemo script calling
        :rtype: str
        """
        hydra_override = []
        choice_model_type, choice_name = self.get_stage_config_choice()
        if self.cluster == "bcp":
            hydra_override += ["+rank=\${RANK}"]
        if self.stage_cfg.model.data.get("data_prefix", None) is None:
            preprocessed_dir = self.stage_cfg.run.get("preprocessed_dir")
            blending_alpha = self.stage_cfg.run.get("blending_alpha")
            auto_blend_command = (
                f"python3 {self._launcher_scripts_path / 'nemo_launcher/collections/auto_blend.py'} "
                f"model_type={choice_model_type} "
                f"preprocessed_dir={preprocessed_dir} "
                f"blending_alpha={blending_alpha}"
            )
            hydra_override += [f"model.data.data_prefix=\$({auto_blend_command})"]
        if self.stage_cfg.model.get("ub_tp_comm_overlap", False):
            get_ub_cfg_file_command = self._get_ub_cfg_file()
            hydra_override += [f"+model.ub_tp_comm_overlap_cfg=\$({get_ub_cfg_file_command})"]
        if self.stage_cfg.model.get("gc_interval", 0) > 1:
            gc_interval = min(self.stage_cfg.model.get("gc_interval"), self.cfg.training.trainer.get("val_check_interval"))
            hydra_override += [f"model.gc_interval={gc_interval}"]
        return hydra_override

    def _get_nemo_code_path(self, model_type: str) -> Path:
        """
        Provide the essential nemo code path for running the stage, usually different model types use different nemo scripts.
        For example, `megatron_t5_pretraining.py` for t5 and `megatron_gpt_pretraining.py` for gpt3.

        :param str model_type: i.e. `gpt3`, `t5`, `mt5`, etc.
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """
        model_type_to_code_path = {
            "t5": self._nemo_code_path / "examples/nlp/language_modeling/megatron_t5_pretraining.py",
            "mt5": self._nemo_code_path / "examples/nlp/language_modeling/megatron_t5_pretraining.py",
            "gpt3": self._nemo_code_path / "examples/nlp/language_modeling/megatron_gpt_pretraining.py",
            "bert": self._nemo_code_path / "examples/nlp/language_modeling/megatron_bert_pretraining.py",
        }
        return model_type_to_code_path[model_type]

    def _get_ub_cfg_file(self) -> str:
        """
        Spawn the script to search UB configuration file
        """
        tp_size = self.stage_cfg.model.get("tensor_model_parallel_size")
        hidden_size = self.stage_cfg.model.get("hidden_size")
        mb_size = self.stage_cfg.model.get("micro_batch_size")
        seqlen = self.stage_cfg.model.get("encoder_seq_length")
        ub_cfg_path = os.path.join(self._launcher_scripts_path, "launcher_scripts/conf/training/gpt3/ub-confs")

        get_ub_cfg_file_command = (
            f"python3 {self._launcher_scripts_path / 'nemo_launcher/collections/conditional_cfgs.py'} "
            f"name=get_ub_cfg_file "
            f"ub_cfg_path={ub_cfg_path} "
            f"tp_size={tp_size} "
            f"hidden_size={hidden_size} "
            f"mb_size={mb_size} "
            f"seqlen={seqlen}"
        )
        return get_ub_cfg_file_command
