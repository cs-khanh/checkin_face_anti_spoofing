import os
import sys

# Set CUDA library path BEFORE importing onnxruntime/insightface
conda_prefix = os.environ.get('CONDA_PREFIX', '')
if not conda_prefix:
    # Fallback to hardcoded path if CONDA_PREFIX not set
    conda_prefix = "/home/coder/trong/computervision/checkin_face_anti_spoofing/.env_cv"

cuda_libs = [
    f"{conda_prefix}/lib/python3.9/site-packages/nvidia/cublas/lib",
    f"{conda_prefix}/lib/python3.9/site-packages/nvidia/cudnn/lib",
    f"{conda_prefix}/lib/python3.9/site-packages/nvidia/cufft/lib",
    f"{conda_prefix}/lib/python3.9/site-packages/nvidia/cuda_runtime/lib",
]
ld_path = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = ':'.join(cuda_libs) + ':' + ld_path
print(f"🔧 Set CUDA libs from: {conda_prefix}")

from pathlib import Path
import numpy as np
from insightface.model_zoo import get_model
import cv2
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
import onnxruntime
import glob

EMB_PATH  = "/home/coder/trong/computervision/checkin_face_anti_spoofing/trained_models/recognition/w600k_r50.onnx"

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Tạo session trước với CUDA provider
session = onnxruntime.InferenceSession(EMB_PATH, providers=providers)
actual_providers = session.get_providers()

# Tạo model với session đã có CUDA
rec_model = ArcFaceONNX(EMB_PATH, session=session)

# Kiểm tra provider thực tế được áp dụng
if 'CUDAExecutionProvider' in actual_providers:
    ctx_id = 0  # GPU
    print(f"✅ Using GPU (CUDA) - providers: {actual_providers}")
else:
    ctx_id = -1  # CPU
    print(f"⚠️ Using CPU - providers: {actual_providers}")

rec_model.prepare(ctx_id=ctx_id, input_size=(112,112))

def embed_112(aligned_112_bgr):
    feat = rec_model.get_feat(aligned_112_bgr)  # (512,) thường CHƯA L2-norm
    feat = feat.astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-9)  # L2-normalize -> dùng cosine
    return feat

def build_templates(aligned_root="/home/coder/trong/computervision/checkin_face_anti_spoofing/employees/data_face"):
    names, embs = [], []
    for person in sorted(os.listdir(aligned_root)):
        print(f"[process] {person}")
        pdir = os.path.join(aligned_root, person)
        if not os.path.isdir(pdir): continue
        vecs = []
        for p in glob.glob(os.path.join(pdir, "*.jpg")):
            img = cv2.imread(p)
            if img is None: continue
            e = embed_112(img)
            vecs.append(e)
        if not vecs: continue
        meanv = np.mean(np.stack(vecs), axis=0).astype(np.float32)
        meanv /= (np.linalg.norm(meanv)+1e-9)
        names.append(person.lower()); embs.append(meanv)

    names = np.array(names) 
    embs = np.stack(embs) if embs else np.zeros((0,512),np.float32)

    np.savez("artifacts/templates.npz",
             names=names, embs=embs.astype(np.float32),
             meta={"model_id":"arcface_w600k_r50","align":"5pt_112"})
    
    print(f"[saved] artifacts/templates.npz  identities={len(names)}")
    return names, embs
# ======= tiện ích I/O templates =======
def load_templates(npz_path="artifacts/templates.npz"):
    d = np.load(npz_path, allow_pickle=True)
    names = list(d["names"])
    embs  = d["embs"].astype(np.float32)
    return names, embs

def save_templates(names, embs, npz_path="artifacts/templates.npz"):
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path,
             names=np.array(names),
             embs=embs.astype(np.float32),
             meta={"model_id":"arcface_w600k_r50","align":"5pt_112"})
    print(f"[saved] {npz_path}  identities={len(names)}")

path_root = Path("/home/coder/trong/computervision/checkin_face_anti_spoofing/employees/data_face")

def iter_employee_images(root: Path, name_emp):
    items = []
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if len(subdirs) > 0:
        # Kiểu A: folder theo tên người
        for d in sorted(subdirs):
            if d.name == name_emp:
                for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
                    for p in d.glob(ext):
                        items.append(p)
    return items

# ======= enroll (thêm nhân viên mới) không cần train lại =======
def enroll_new_person(person_name, aligned_images, templates_npz="/home/coder/trong/computervision/checkin_face_anti_spoofing/trained_models/recognition/artifacts/templates.npz", min_imgs=1):
    """
    person_name: tên/thư mục người mới (sẽ lower())
    aligned_images: list các path ảnh đã align 112x112
    """
    vecs = []
    for p in aligned_images:
        img = cv2.imread(str(p))
        if img is None: 
            print(f"[skip] cannot read {p}"); 
            continue
        vecs.append(embed_112(img))
    if len(vecs) < min_imgs:
        print(f"[enroll] not enough valid images ({len(vecs)}/{min_imgs})")
        return False

    meanv = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
    meanv /= (np.linalg.norm(meanv)+1e-9)

    # append vào templates hiện có (nếu chưa có thì tạo mới)
    if os.path.exists(templates_npz):
        names, embs = load_templates(templates_npz)
        names.append(person_name.lower())
        embs = np.vstack([embs, meanv[None, :]]) if embs.size else meanv[None, :]
    else:
        names = [person_name.lower()]
        embs  = meanv[None, :].astype(np.float32)

    save_templates(names, embs, templates_npz)
    print(f"[enroll] added {person_name}, total identities={len(names)}")
    return True

if __name__ == "__main__":
    import os, inspect
    assert os.path.exists(EMB_PATH), f"ONNX not found: {EMB_PATH}"
    print("EMB_PATH ok:", EMB_PATH)

    print("rec_model type:", type(rec_model))
    print("has .prepare?", hasattr(rec_model, "prepare"))
    print("has .get?", hasattr(rec_model, "get"))
    if hasattr(rec_model, "get"):
        try:
            print("get signature:", inspect.signature(rec_model.get))
        except Exception as e:
            print("cannot read signature:", e)
    os.makedirs("artifacts", exist_ok=True)
    build_templates()
    # Example enroll:
    #print(enroll_new_person("NV03_NguyenThanhTrong", iter_employee_images(path_root, "NV03_thanhtronggg"), min_imgs=30))
