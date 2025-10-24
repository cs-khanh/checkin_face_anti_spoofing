# Optimized Face Recognition Model Structure

This directory contains an optimized set of models for the face authentication system, organized by functionality.

## Directory Structure

- **detection/**: Face detection models
  - `det_10g.onnx`: Primary SCRFD model (medium size, good accuracy-speed balance)
  - `det_500m_light.onnx`: Lightweight model for constrained environments

- **recognition/**: Face recognition models
  - `w600k_r50.onnx`: Primary ArcFace model based on ResNet50
  - `w600k_mbf_light.onnx`: Lightweight MobileFaceNet model for constrained environments

- **landmark/**: Facial landmark models
  - `2d106det.onnx`: 106-point facial landmark detector

- **anti_spoofing/**: Anti-spoofing models
  - `silent_face/`: Silent Face anti-spoofing implementation
    - MiniFASNetV1SE and MiniFASNetV2 models for different spoof detection techniques

## Configuration

The system is configured to use the primary models by default, with lightweight options available for constrained environments.

To switch to lightweight models for low-resource devices, edit the model_config.json file and replace the model_path with light_model_path for both detection and recognition.
