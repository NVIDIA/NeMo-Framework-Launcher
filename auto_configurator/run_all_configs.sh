#!/bin/bash
# 自动化并行搜索 - 串行测试所有配置
# 用法: bash run_all_configs.sh [MAX_CONFIGS] [MAX_STEPS]
#
# 参数说明:
#   MAX_CONFIGS: 最多测试多少个配置（默认全部）
#   MAX_STEPS: 每个配置运行多少步（默认100）

set -e

MAX_CONFIGS=${1:-999999}
MAX_STEPS=${2:-100}

# 路径配置
CONFIG_DIR="/mnt/pre-train/wangbinluo/NeMo-Framework-Launcher/auto_configurator/results/qwen3_moe/30b_140gb/candidate_configs"
RESULTS_DIR="/mnt/pre-train/wangbinluo/NeMo-Framework-Launcher/auto_configurator/results/qwen3_moe/30b_140gb"
TRAIN_SCRIPT="/mnt/pre-train/wangbinluo/Megatron-LM/examples/miromind_v0_1/run_qwen3_moe_parallel_search.sh"
MEGATRON_DIR="/mnt/pre-train/wangbinluo/Megatron-LM"

# 激活环境
source /opt/megatron/.venv/bin/activate

# 创建日志目录
mkdir -p "$RESULTS_DIR/training_logs"
mkdir -p "$RESULTS_DIR/logs"

echo "=============================================="
echo "自动化并行搜索 - 开始"
echo "配置目录: $CONFIG_DIR"
echo "最大配置数: $MAX_CONFIGS"
echo "每配置步数: $MAX_STEPS"
echo "=============================================="

# 统计
TOTAL_CONFIGS=$(ls -1 "$CONFIG_DIR"/*.yaml 2>/dev/null | wc -l)
echo "共发现 $TOTAL_CONFIGS 个配置文件"

# 记录总开始时间
TOTAL_START_TIME=$(date +%s)

# 计数器
COUNT=0
SUCCESS=0
FAILED=0

# 创建结果摘要文件
SUMMARY_FILE="$RESULTS_DIR/run_summary.csv"
echo "config_name,tp,pp,ep,mbs,status,duration_s" > "$SUMMARY_FILE"

# 遍历所有配置文件
for config_file in "$CONFIG_DIR"/*.yaml; do
    if [ ! -f "$config_file" ]; then
        continue
    fi
    
    COUNT=$((COUNT + 1))
    
    if [ $COUNT -gt $MAX_CONFIGS ]; then
        echo "已达到最大配置数限制 ($MAX_CONFIGS)，停止"
        break
    fi
    
    config_name=$(basename "$config_file" .yaml)
    
    # 解析配置参数
    # 使用 grep 和 awk 从 YAML 中提取值
    tp=$(grep "tensor_model_parallel_size:" "$config_file" | grep -v "expert" | head -1 | awk '{print $2}')
    pp=$(grep "pipeline_model_parallel_size:" "$config_file" | head -1 | awk '{print $2}')
    ep=$(grep "expert_model_parallel_size:" "$config_file" | head -1 | awk '{print $2}')
    mbs=$(grep "micro_batch_size:" "$config_file" | head -1 | awk '{print $2}')
    
    echo ""
    echo "=============================================="
    echo "[$COUNT/$TOTAL_CONFIGS] 运行配置: $config_name"
    echo "TP=$tp, PP=$pp, EP=$ep, MBS=$mbs"
    echo "=============================================="
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 运行训练
    cd "$MEGATRON_DIR"
    if bash "$TRAIN_SCRIPT" "$tp" "$pp" "$ep" "$mbs" "$config_name" "$MAX_STEPS"; then
        STATUS="success"
        SUCCESS=$((SUCCESS + 1))
    else
        STATUS="failed"
        FAILED=$((FAILED + 1))
        echo "警告: 配置 $config_name 运行失败"
    fi
    
    # 记录结束时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # 写入摘要
    echo "$config_name,$tp,$pp,$ep,$mbs,$STATUS,$DURATION" >> "$SUMMARY_FILE"
    
    echo "完成: $config_name (耗时 ${DURATION}s, 状态: $STATUS)"
done

# 记录总结束时间
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))

echo ""
echo "=============================================="
echo "自动化并行搜索 - 完成"
echo "总配置数: $COUNT"
echo "成功: $SUCCESS"
echo "失败: $FAILED"
echo "总耗时: ${TOTAL_DURATION}s"
echo "摘要文件: $SUMMARY_FILE"
echo "=============================================="

# 分析结果
echo ""
echo "正在分析结果..."

cd /mnt/pre-train/wangbinluo/NeMo-Framework-Launcher/auto_configurator
python autoconfig/scripts/compare_throughput.py \
    auto_configurator_path=/mnt/pre-train/wangbinluo/NeMo-Framework-Launcher/auto_configurator \
    search_config=qwen3_moe/30b

echo ""
echo "分析完成！查看结果:"
echo "  - 最优配置: $RESULTS_DIR/final_result/optimal_config_30b_1nodes.yaml"
echo "  - 完整排名: $RESULTS_DIR/final_result/final_summary_1nodes.csv"
