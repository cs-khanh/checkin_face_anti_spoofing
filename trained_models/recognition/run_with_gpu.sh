#!/bin/bash
# Script để chạy train.py với GPU

ENV_PATH="/home/coder/trong/computervision/checkin_face_anti_spoofing/.env_cv"

export LD_LIBRARY_PATH="$ENV_PATH/lib/python3.9/site-packages/nvidia/cublas/lib:$ENV_PATH/lib/python3.9/site-packages/nvidia/cudnn/lib:$ENV_PATH/lib/python3.9/site-packages/nvidia/cufft/lib:$ENV_PATH/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

echo "✅ CUDA Libraries Path set from: $ENV_PATH"
echo ""

# Sử dụng conda run với path tuyệt đối
conda run -p "$ENV_PATH" --no-capture-output python train.py "$@"
