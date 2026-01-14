#!/bin/bash
# 快速测试 - 只测试5个代表性配置
# 用法: bash run_quick_test.sh [MAX_STEPS]

set -e

MAX_STEPS=${1:-50}

# 路径配置
RESULTS_DIR="/mnt/pre-train/wangbinluo/NeMo-Framework-Launcher/auto_configurator/results/qwen3_moe/30b_140gb"
TRAIN_SCRIPT="/mnt/pre-train/wangbinluo/Megatron-LM/examples/miromind_v0_1/run_qwen3_moe_parallel_search.sh"
MEGATRON_DIR="/mnt/pre-train/wangbinluo/Megatron-LM"

# 激活环境
source /opt/megatron/.venv/bin/activate

# 创建日志目录
mkdir -p "$RESULTS_DIR/training_logs"
mkdir -p "$RESULTS_DIR/logs"

echo "=============================================="
echo "快速测试 - 5个代表性配置"
echo "每配置步数: $MAX_STEPS"
echo "=============================================="

# 定义测试配置: TP PP EP MBS
declare -a CONFIGS=(
    "1 1 8 3"   # 你当前使用的配置
    "1 1 4 3"   # EP=4 对比
    "1 1 2 3"   # EP=2 对比
    "2 1 4 3"   # 加入 TP=2
    "1 2 4 3"   # 加入 PP=2
)

# 创建结果摘要文件
SUMMARY_FILE="$RESULTS_DIR/quick_test_summary.csv"
echo "config_name,tp,pp,ep,mbs,status,duration_s" > "$SUMMARY_FILE"

TOTAL_START_TIME=$(date +%s)
COUNT=0
SUCCESS=0

for config in "${CONFIGS[@]}"; do
    COUNT=$((COUNT + 1))
    
    # 解析参数
    read tp pp ep mbs <<< "$config"
    config_name="quick_test_tp${tp}_pp${pp}_ep${ep}_mbs${mbs}"
    
    echo ""
    echo "=============================================="
    echo "[$COUNT/${#CONFIGS[@]}] 测试配置: $config_name"
    echo "TP=$tp, PP=$pp, EP=$ep, MBS=$mbs"
    echo "=============================================="
    
    START_TIME=$(date +%s)
    
    cd "$MEGATRON_DIR"
    if bash "$TRAIN_SCRIPT" "$tp" "$pp" "$ep" "$mbs" "$config_name" "$MAX_STEPS"; then
        STATUS="success"
        SUCCESS=$((SUCCESS + 1))
    else
        STATUS="failed"
        echo "警告: 配置 $config_name 运行失败"
    fi
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "$config_name,$tp,$pp,$ep,$mbs,$STATUS,$DURATION" >> "$SUMMARY_FILE"
    echo "完成: $config_name (耗时 ${DURATION}s)"
done

TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))

echo ""
echo "=============================================="
echo "快速测试 - 完成"
echo "测试配置数: $COUNT"
echo "成功: $SUCCESS"
echo "总耗时: ${TOTAL_DURATION}s"
echo "=============================================="

# 显示摘要
echo ""
echo "测试结果摘要:"
cat "$SUMMARY_FILE"

echo ""
echo "TensorBoard 日志目录: $RESULTS_DIR/training_logs/"
echo "运行以下命令查看 TensorBoard:"
echo "  tensorboard --logdir=$RESULTS_DIR/training_logs/"
