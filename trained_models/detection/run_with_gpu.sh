#!/bin/bash
# Script wrapper để chạy với GPU - tự động set CUDA libraries path

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cufft/lib:$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

echo "✅ CUDA Libraries Path set:"
echo "   - cuBLAS: $CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cublas/lib"
echo "   - cuDNN:  $CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cudnn/lib"
echo "   - cuFFT:  $CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cufft/lib"
echo "   - CUDART: $CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cuda_runtime/lib"
echo ""

python detect_face_1_folder.py "$@"
