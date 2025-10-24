import cv2, numpy as np, onnxruntime as ort
import torch
onnx_path = "/home/coder/trong/computer_vision/face_auth_system/version2/trained_models/face_anti_spoofing/weights/antispoof_80x80.onnx"
sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])


def preprocess_bgr_to_nchw01(img_bgr, size=(80,80)):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # repo dùng RGB
    img_rgb = cv2.resize(img_rgb, size)
    x = img_rgb.astype(np.float32)            # [0,1]
    x = np.transpose(x, (2,0,1))                        # CHW
    x = np.ascontiguousarray(x)[None, ...]    # [1,3,H,W] float32
    return x

img = cv2.imread('/home/coder/trong/computer_vision/face_auth_system/version2/employees/data_face/NV02_phamgiakhanh/phamgiakhanh_59.jpg')
x = preprocess_bgr_to_nchw01(img, (80,80))

logits = sess.run(["logits"], {"input": x})[0]         # [1,C]
# softmax
e = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = (e / e.sum(axis=1, keepdims=True))[0]
print("Probabilities:", probs)
# # sau khi tính probs
REAL_IDX = 1   # với ONNX của bạn, lớp 2 mới là LIVE/REAL
real_prob = float(probs[REAL_IDX])
printt = float(probs[0])
replay = float(probs[2])
print("Real prob: ", real_prob)
print("Print prob: ", printt)
print("Replay prob: ", replay)

