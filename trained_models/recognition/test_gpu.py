#!/usr/bin/env python
import os
import sys

# Set CUDA paths
conda_prefix = os.environ.get('CONDA_PREFIX', '')
if conda_prefix:
    cuda_libs = [
        f"{conda_prefix}/lib/python3.9/site-packages/nvidia/cublas/lib",
        f"{conda_prefix}/lib/python3.9/site-packages/nvidia/cudnn/lib",
        f"{conda_prefix}/lib/python3.9/site-packages/nvidia/cufft/lib",
        f"{conda_prefix}/lib/python3.9/site-packages/nvidia/cuda_runtime/lib",
    ]
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = ':'.join(cuda_libs) + ':' + ld_path
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    print(f"Set LD_LIBRARY_PATH: {new_ld_path[:200]}...")

import onnxruntime as ort

print("\n=== ONNX Runtime Info ===")
print(f"Version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")

# Try to create session with CUDA
print("\n=== Testing CUDA Provider ===")
try:
    session_options = ort.SessionOptions()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # Dummy model path - replace with actual
    model_path = "/home/coder/trong/computervision/checkin_face_anti_spoofing/trained_models/recognition/w600k_r50.onnx"
    
    sess = ort.InferenceSession(model_path, session_options, providers=providers)
    actual_providers = sess.get_providers()
    
    if 'CUDAExecutionProvider' in actual_providers:
        print(f"✅ SUCCESS! Using GPU - providers: {actual_providers}")
    else:
        print(f"⚠️ Fallback to CPU - providers: {actual_providers}")
except Exception as e:
    print(f"❌ Error: {e}")
