#!/bin/bash
# Qwen3-MoE-30B-A3B EP 验证脚本
# 用于验证 EP=8 是否是最优配置

set -euo pipefail

# 设置路径
export CHECKPOINT_PATH="/mnt/pre-train/james/share_ckpts/Qwen/Qwen3-30B-A3B-Base"
export DATA_PATH="/mnt/pre-train/james/tokenized_data_megatron/huggingface_preview/Qwen3-30B-A3B-Base/HuggingFaceTB/issues-kaggle-notebooks/kaggle"
export TOKENIZER_ARG="/mnt/pre-train/james/share_ckpts/Qwen/Qwen3-30B-A3B-Base"

# 设置 auto_configurator 路径
export AUTO_CONFIGURATOR_PATH="/mnt/pre-train/wangbinluo/NeMo-Framework-Launcher/auto_configurator"

# 进入 auto_configurator 目录
cd ${AUTO_CONFIGURATOR_PATH}

echo "=========================================="
echo "Qwen3-MoE-30B-A3B EP 验证"
echo "=========================================="
echo "目标: 验证 EP=8 是否是最优配置"
echo "搜索空间: EP = [1, 2, 4, 8, 16]"
echo "=========================================="
echo ""

# 运行自动化并行配置工具
# 注意：checkpoint、tokenizer 和 data_prefix 路径已在 base_configs/qwen3_moe_30b.yaml 中配置
python main.py \
    auto_configurator_path=${AUTO_CONFIGURATOR_PATH} \
    data_dir=${DATA_PATH} \
    search_config=qwen3_moe/30b \
    cluster_type=bcm \
    skip_job_submission=True  # 设置为 False 来运行完整搜索

echo ""
echo "=========================================="
echo "配置生成完成！"
echo "=========================================="
echo "查看生成的配置:"
echo "  ls -la results/qwen3_moe/30b_80gb/candidate_configs/"
echo ""
echo "查看最优配置（运行完成后）:"
echo "  cat results/qwen3_moe/30b_80gb/final_result/optimal_config_*.yaml"
echo ""
echo "查看性能对比（运行完成后）:"
echo "  cat results/qwen3_moe/30b_80gb/final_result/throughput_comparison.csv"
echo "=========================================="
