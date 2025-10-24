import numpy as np
from insightface.model_zoo import get_model
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from insightface.model_zoo import get_model
from insightface.utils import face_align
from tqdm import tqdm
import csv

DRAW_LAND = True
DET_PATH = "/home/coder/trong/computer_vision/face_auth_system/version2/trained_models/detection/det_10g.onnx"

providers =  ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Tạo model với providers thích hợp
scrfd = get_model(DET_PATH, providers=providers)  # auto-detect SCRFD

# Đặt ctx_id phù hợp với provider đã áp dụng
ctx_id = 0
print(f"Using context id: {ctx_id} providers: {providers[ctx_id]}")
scrfd.prepare(ctx_id=ctx_id, input_size=(640,640), det_thresh=0.4, nms=0.4)
PADDING_FRAC = 0.05  # Mở rộng khung mặt thêm 5% mỗi bên
def detect_faces(img):
    bboxes, kpss = scrfd.detect(img,  max_num=1)

    if bboxes is not None and len(bboxes) > 0:
        for i in range(len(bboxes)):
            x1, y1, x2, y2, score = bboxes[i]
            x1 = max(0, int(x1) - PADDING_FRAC * (x2 - x1))
            y1 = max(0, int(y1) - PADDING_FRAC * (y2 - y1))
            x2 = min(img.shape[1], int(x2) + PADDING_FRAC * (x2 - x1))
            y2 = min(img.shape[0], int(y2) + PADDING_FRAC * (y2 - y1))
            bboxes[i][0] = x1
            bboxes[i][1] = y1
            bboxes[i][2] = x2
            bboxes[i][3] = y2
    return bboxes, kpss

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)

# ------------------------ Data iterator ------------------------
def iter_employee_images(root: Path) -> List[Tuple[str, Path]]:
    items = []
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if len(subdirs) > 0:
        # Kiểu A: folder theo tên người
        for d in sorted(subdirs):
            for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
                for p in d.glob(ext):
                    items.append((d.name, p))
    return items

# ------------------------ Main processing ------------------------

data_path = Path("/home/coder/trong/computer_vision/face_auth_system/version2/employees/data_collect")
out_root = Path("/home/coder/trong/computer_vision/face_auth_system/version2/employees/data_face")
log_path = Path("/home/coder/trong/computer_vision/face_auth_system/version2/employees/data_face_log.csv")

log_rows = []

for emp_id, img_path in tqdm(iter_employee_images(data_path)):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] Cannot read: {img_path}")
        continue
    bboxes, kpss = detect_faces(img)
    if bboxes is not None and len(bboxes) > 0:
        box = bboxes[0]
        kps = kpss[0] if kpss is not None and len(kpss) > 0 else None
        x1, y1, x2, y2, score = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        face_crop = img[y1:y2, x1:x2]
        save_dir = out_root / emp_id
        ensure_dir(save_dir)
        save_path = save_dir / img_path.name
        #cv2.imwrite(str(save_path), to_uint8(face_crop))
        # Lưu aligned nếu có đủ 5 điểm landmark
        if kps is not None and np.array(kps).shape == (5,2):
            from insightface.utils import face_align
            aligned = face_align.norm_crop(img, landmark=kps, image_size=112)
            #aligned_path = str(save_path).replace('.jpg', '_aligned112.jpg')
            cv2.imwrite(str(save_path), to_uint8(aligned))
        else:
            log_rows.append({
                'emp_id': emp_id,
                'img': str(img_path),
                'score': '',
                'status': 'not_enough_landmark',
                'reasons': 'not_enough_landmark'
            })
            print(f"[NO FACE] {img_path}")
        # log_rows.append({
        #     'emp_id': emp_id,
        #     'img': str(img_path),
        #     'crop': str(save_path),
        #     'score': score,
        #     'status': 'OK',
        #     'reasons': ''
        # })
        # print(f"[OK] {img_path} -> {save_path}")
    else:
        log_rows.append({
            'emp_id': emp_id,
            'img': str(img_path),
            'score': '',
            'status': 'NO_FACE',
            'reasons': 'no_face'
        })
        print(f"[NO FACE] {img_path}")

# Lưu log ra file CSV
if log_rows:
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Log saved to {log_path}")