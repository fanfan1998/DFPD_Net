#!/bin/bash
# DDP多卡训练启动脚本 - 使用 torchrun

# 设置日志文件路径（带时间戳）
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo "=========================================="
echo "训练开始时间: $(date)"
echo "日志文件: $LOG_FILE"
echo "=========================================="

export CONDA_SOLVER=classic
if [[ -f /bigMemory/anaconda/etc/profile.d/conda.sh ]]; then
  source /bigMemory/anaconda/etc/profile.d/conda.sh 2>/dev/null || true
  conda activate dfclip 2>/dev/null || true
elif [[ -f /disk/disk1/conda/etc/profile.d/conda.sh ]]; then
  source /disk/disk1/conda/etc/profile.d/conda.sh 2>/dev/null || true
  conda activate dfclip 2>/dev/null || true
fi

# 设置可见的GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 从 CUDA_VISIBLE_DEVICES 推断 GPU 数量
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}

# 模型名称参数（默认为 resnet50，可选: resnet50 或 xception）
MODEL_NAME=${1:-xception}

# 验证模型名称
if [[ "$MODEL_NAME" != "resnet50" && "$MODEL_NAME" != "xception" ]]; then
    echo "错误: 不支持的模型名称 '$MODEL_NAME'"
    echo "支持的模型: resnet50, xception"
    exit 1
fi

echo "使用模型: $MODEL_NAME"

# 使用 nohup + torchrun 启动（后台运行）
nohup torchrun --nproc_per_node=$NUM_GPUS \
         --master_addr=localhost \
         --master_port=29500 \
         train.py \
         --train_dataset "FaceForensics++" \
         --test_dataset "FaceForensics++,Celeb-DF-v2" \
         --model_name "$MODEL_NAME" \
         > "$LOG_FILE" 2>&1 &

# 保存进程ID
TRAIN_PID=$!
echo "训练进程已启动，PID: $TRAIN_PID"
echo "使用以下命令查看日志: tail -f $LOG_FILE"
echo "使用以下命令停止训练: kill $TRAIN_PID"

# FaceForensics++, FaceShifter, DeepFakeDetection, Celeb-DF-v1, Celeb-DF-v2, DeeperForensics-1.0, UADFV

