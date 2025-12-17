#!/bin/bash
# Script để chạy gunicorn với GPU support

ENV_PATH="/home/coder/trong/computervision/checkin_face_anti_spoofing/.env_cv"

export LD_LIBRARY_PATH="$ENV_PATH/lib/python3.9/site-packages/nvidia/cublas/lib:$ENV_PATH/lib/python3.9/site-packages/nvidia/cudnn/lib:$ENV_PATH/lib/python3.9/site-packages/nvidia/cufft/lib:$ENV_PATH/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

echo "✅ Starting Flask app with GPU support"
echo "   CUDA libs: $ENV_PATH/lib/python3.9/site-packages/nvidia/*/lib"
echo ""

# Chạy gunicorn với config
gunicorn -c gunicorn_config.py app:app
