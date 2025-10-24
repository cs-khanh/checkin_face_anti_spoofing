import torch, cv2
from anti_spoof_utils import AntiSpoofDetector
img = cv2.imread("/home/coder/trong/computer_vision/face_auth_system/version2/employees/data_face/NV02_phamgiakhanh/phamgiakhanh_59.jpg")
detector = AntiSpoofDetector(device='cpu', real_index=1)
is_real, conf = detector.predict(img)
print(is_real, conf)
