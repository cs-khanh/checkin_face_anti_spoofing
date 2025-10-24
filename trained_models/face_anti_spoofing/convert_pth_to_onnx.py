from src.anti_spoof_predict import AntiSpoofPredict
import torch
import os
anti = AntiSpoofPredict(device_id=0)

model_path = '/home/coder/trong/computer_vision/face_auth_system/version2/trained_models/face-anti-spoofing/weights/2.7_80x80_MiniFASNetV2.pth'
model = anti._load_model(model_path)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "/home/coder/trong/computer_vision/face_auth_system/version2/trained_models/face-anti-spoofing/weights/2.7_80x80_MiniFASNetV2.pth"
onnx_out   = "/home/coder/trong/computer_vision/face_auth_system/version2/trained_models/face-anti-spoofing/weights/antispoof_80x80.onnx"

# 1) nạp model đúng kernel từ ckpt (repo HF)
anti  = AntiSpoofPredict(device_id=0)
model = anti._load_model(model_path)  # sau patch, trả về nn.Module
model = model.to(device).eval()

# 2) dummy input đúng kích thước & dtype (NCHW, float32, RGB [0,1])
h, w = 80, 80   # hoặc lấy từ parse_model_name nếu cần động
dummy = torch.randn(1, 3, h, w, device=device, dtype=torch.float32)

# 3) export
torch.onnx.export(
    model, dummy, onnx_out,
    input_names=["input"], output_names=["logits"],
    opset_version=12, do_constant_folding=True,
    dynamic_axes=None  # cố định 1x3x80x80 -> nhanh & đơn giản
)
print("✅ Exported:", onnx_out)
