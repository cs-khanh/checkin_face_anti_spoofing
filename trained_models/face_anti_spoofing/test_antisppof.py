import os, sys, cv2, torch
import torch.nn.functional as F
import numpy as np

# ---- import đúng kiến trúc của Space ----
from src.model_lib.MiniFASNet import MiniFASNetV2
from src.utility import parse_model_name, get_kernel
from src.data_io.transform import ToTensor
from src.anti_spoof_predict import AntiSpoofPredict

anti = AntiSpoofPredict(device_id=0)
model_path = '/home/coder/trong/computer_vision/face_auth_system/version2/trained_models/face_anti_spoofing/weights/2.7_80x80_MiniFASNetV2.pth'
img = cv2.imread('/home/coder/trong/computer_vision/face_auth_system/version2/employees/data_face/NV02_phamgiakhanh/phamgiakhanh_59.jpg')
img = cv2.resize(img, (80,80))
score = anti.predict(img, model_path)[0]
print("Score: ", score)
real_prob = float(score[1])
printt = float(score[0])
replay = float(score[2])
print("Real prob: ", real_prob)
print("Print prob: ", printt)
print("Replay prob: ", replay)