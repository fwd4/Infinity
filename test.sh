#!/bin/bash  

检查参数  
if [ $# -lt 1 ]; then  
    echo "用法: $0 <GPU_IDS> [NUM_CLASSES] [IMAGES_PER_CLASS]"  
    echo "例如: $0 0,1,3,5   # 使用GPU 0,1,3,5四张卡"  
    echo "      $0 0,2       # 使用GPU 0,2两张卡"  
    exit 1  
fi  

#执行命令: bash test.sh 0,1,2,3,,4
# 参数解析  
GPU_IDS=$(echo $1 | tr ',' ' ')  # 将逗号分隔的GPU ID转为空格分隔  
NUM_GPUS=$(echo $GPU_IDS | wc -w)  # 计算GPU数量  

# 可选参数  
NUM_CLASSES=${2:-1000}     # 默认1000类  
IMAGES_PER_CLASS=${3:-50}  # 默认每类50张  
TOTAL_IMAGES=$((NUM_CLASSES * IMAGES_PER_CLASS))  

# 输出文件夹  
OUTPUT_FOLDER="outputs/fid_samples_8_80"  
mkdir -p $OUTPUT_FOLDER  

# Python 脚本路径  
PYTHON_SCRIPT="/home/Infinity/test.py"  

echo "使用 $NUM_GPUS 张GPU: $GPU_IDS"  
echo "处理 $NUM_CLASSES 个类别，每类 $IMAGES_PER_CLASS 张图片"  

# 计算每个GPU处理的类别数  
CLASSES_PER_GPU=$((NUM_CLASSES / NUM_GPUS))  
REMAINDER=$((NUM_CLASSES % NUM_GPUS))  # 处理不能整除的情况  

# 处理索引  
GPU_INDEX=0  
START_CLASS=0  

# 为每个GPU分配任务  
for GPU_ID in $GPU_IDS; do  
    if [ $GPU_INDEX -lt $REMAINDER ]; then     
        THIS_GPU_CLASSES=$((CLASSES_PER_GPU + 1))  
    else  
        THIS_GPU_CLASSES=$CLASSES_PER_GPU  
    fi  
    
    END_CLASS=$((START_CLASS + THIS_GPU_CLASSES - 1))  
    
    echo "在GPU $GPU_ID 上处理类别 $START_CLASS 到 $END_CLASS (共 $THIS_GPU_CLASSES 个类别)..."  
    
    # 启动任务  
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 $PYTHON_SCRIPT \
        --start_class $START_CLASS \
        --end_class $((END_CLASS + 1)) \
        --images_per_class $IMAGES_PER_CLASS \
        --output_folder $OUTPUT_FOLDER \
        --gpu_id 0  &  
    
    START_CLASS=$((END_CLASS + 1))  
    GPU_INDEX=$((GPU_INDEX + 1))  
done  

# 等待所有任务完成  
wait  
echo "所有任务完成！可以在 $OUTPUT_FOLDER 找到生成的图片"  

# 创建NPZ文件（如果需要）  
echo "正在创建NPZ文件..."  
python3 -c "from utils.misc import create_npz_from_sample_folder; create_npz_from_sample_folder('$OUTPUT_FOLDER')"  
echo "FID样本已保存到 outputs/fid_samples.npz"