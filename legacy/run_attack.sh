#!/bin/bash

# ERa Attack 快速启动脚本
# 使用方法: ./run_attack.sh [mode] [options]

echo "========================================"
echo "   ERa Attack - EMG对抗攻击系统"
echo "========================================"

# 默认参数
MODE=${1:-"demo"}
ATTACK_TYPE=${2:-"fc_pgd"}
NUM_SAMPLES=${3:-50}

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Python环境
check_environment() {
    echo -e "${YELLOW}检查环境...${NC}"
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}错误: 未找到Python3${NC}"
        exit 1
    fi
    
    # 检查依赖
    python3 -c "import torch" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}警告: PyTorch未安装，正在安装依赖...${NC}"
        pip install -r requirements.txt
    fi
    
    echo -e "${GREEN}环境检查完成!${NC}"
}

# 运行演示
run_demo() {
    echo -e "\n${GREEN}运行演示示例...${NC}"
    python3 example_usage.py
}

# 运行数字攻击
run_digital_attack() {
    echo -e "\n${GREEN}执行数字域攻击...${NC}"
    echo "攻击类型: $ATTACK_TYPE"
    echo "样本数量: $NUM_SAMPLES"
    
    python3 main.py \
        --mode attack \
        --attack-type $ATTACK_TYPE \
        --num-samples $NUM_SAMPLES \
        --device cuda
}

# 运行完整实验
run_experiment() {
    echo -e "\n${GREEN}运行完整实验...${NC}"
    echo "测试所有三种攻击算法"
    
    python3 main.py \
        --mode experiment \
        --num-samples $NUM_SAMPLES \
        --output-dir ./results/experiment_$(date +%Y%m%d_%H%M%S)
}

# 运行物理攻击（需要HackRF）
run_physical_attack() {
    echo -e "\n${YELLOW}准备物理射频攻击...${NC}"
    
    # 检查HackRF
    if ! command -v hackrf_info &> /dev/null; then
        echo -e "${RED}错误: HackRF工具未安装${NC}"
        echo "请安装: sudo apt-get install hackrf"
        exit 1
    fi
    
    # 检查设备连接
    hackrf_info &> /dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: HackRF设备未连接${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}HackRF就绪，开始攻击...${NC}"
    
    python3 main.py \
        --mode attack \
        --attack-type fc_pgd \
        --num-samples 10 \
        --physical \
        --carrier-freq 433e6 \
        --tx-gain 20
}

# 分析结果
analyze_results() {
    echo -e "\n${GREEN}分析实验结果...${NC}"
    
    # 查找最新的结果文件
    LATEST_RESULT=$(ls -t ./results/experiment_results_*.json 2>/dev/null | head -1)
    
    if [ -z "$LATEST_RESULT" ]; then
        echo -e "${YELLOW}未找到实验结果文件${NC}"
        return
    fi
    
    echo "分析文件: $LATEST_RESULT"
    
    # 使用Python分析
    python3 -c "
import json
import sys

with open('$LATEST_RESULT', 'r') as f:
    data = json.load(f)

print('\n实验结果摘要:')
print('-' * 40)

for attack, results in data.get('attack_results', {}).items():
    print(f\"{attack.upper()}:\")
    print(f\"  成功率: {results.get('success_rate', 0):.1f}%\")
    print(f\"  耗时: {results.get('elapsed_time', 0):.2f}秒\")
    print()
"
}

# 清理临时文件
cleanup() {
    echo -e "\n${YELLOW}清理临时文件...${NC}"
    
    # 删除GNU Radio生成的临时文件
    rm -f *.complex64 2>/dev/null
    rm -f flowgraph.py 2>/dev/null
    
    # 删除Python缓存
    rm -rf __pycache__ 2>/dev/null
    rm -rf ./*.pyc 2>/dev/null
    
    echo -e "${GREEN}清理完成${NC}"
}

# 显示帮助
show_help() {
    echo "使用方法: $0 [mode] [options]"
    echo ""
    echo "模式:"
    echo "  demo        - 运行演示示例（默认）"
    echo "  attack      - 执行数字域攻击"
    echo "  experiment  - 运行完整实验"
    echo "  physical    - 执行物理射频攻击（需要HackRF）"
    echo "  analyze     - 分析实验结果"
    echo "  cleanup     - 清理临时文件"
    echo "  help        - 显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 demo                    # 运行演示"
    echo "  $0 attack fc_pgd 100       # FC-PGD攻击100个样本"
    echo "  $0 experiment              # 运行完整实验"
    echo "  $0 physical                # 执行物理攻击"
}

# 主程序
main() {
    # 检查环境
    check_environment
    
    # 根据模式执行
    case $MODE in
        demo)
            run_demo
            ;;
        attack)
            run_digital_attack
            ;;
        experiment)
            run_experiment
            ;;
        physical)
            run_physical_attack
            ;;
        analyze)
            analyze_results
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}未知模式: $MODE${NC}"
            show_help
            exit 1
            ;;
    esac
    
    echo -e "\n${GREEN}完成!${NC}"
}

# 运行主程序
main