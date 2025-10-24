from flask import Flask, render_template, url_for, jsonify, request
import sys
import os
import time
import numpy as np
from insightface.model_zoo import get_model
from insightface.utils import face_align
import cv2
import faiss
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from threading import Lock
from functools import lru_cache
import hashlib
import warnings
import onnxruntime as ort
# from trained_models.face_anti_spoofing.src.anti_spoof_predict import AntiSpoofPredict

# anti = AntiSpoofPredict(device_id=0)
# model_path = '/home/coder/trong/computer_vision/face_auth_system/version2/trained_models/face_anti_spoofing/weights/2.7_80x80_MiniFASNetV2.pth'

# ================== Anti-spoof (ONNX) ==================
onnx_path = "/home/coder/trong/computer_vision/face_auth_system/version2/trained_models/face_anti_spoofing/weights/antispoof_80x80.onnx"

# ONNX session (CUDA -> CPU fallback đã cấu hình bên dưới)
sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])

LIVE_THRESHOLD = 0.50        # ngưỡng quyết định live/spoof (tune theo data)
MIN_FACE_SIZE  = 120         # mặt nhỏ hơn cạnh ngắn này coi là kém chất lượng
BLUR_VAR_THR   = 20.0        # var Laplacian < 20 coi là mờ (tune theo camera)

# def predict_pth(img):
#     img = cv2.resize(img, (80,80))
#     score = anti.predict(img, model_path)[0]
#     print("Score: ", score)
#     real_prob = float(score[1])
#     printt = float(score[0])
#     replay = float(score[2])
#     return real_prob, printt, replay
def _lap_var(gray):
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def preprocess_bgr_to_nchw01(img_bgr, size=(80,80)):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # repo dùng RGB
    img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_LINEAR)
    x = img_rgb.astype(np.float32)            # [0,1]
    x = np.transpose(x, (2,0,1))                        # CHW
    x = np.ascontiguousarray(x)[None, ...]    # [1,3,H,W] float32
    return x

def predict_anti_spoof_facecrop(face_bgr80):
    """Trả về (real_prob, print_prob, replay_prob) cho 1 crop mặt (BGR)."""
    x = preprocess_bgr_to_nchw01(face_bgr80, (80,80))
    logits = sess.run(["logits"], {"input": x})[0]      # [1,C]
    # softmax an toàn số học
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = (e / e.sum(axis=1, keepdims=True))[0]       # (C,)
    return float(probs[1]), float(probs[0]), float(probs[2])

# =======================================================

# Suppress ONNX Runtime warnings
warnings.filterwarnings('ignore', category=UserWarning, module='onnxruntime')

DRAW_LAND = True
DET_PATH = "/home/coder/trong/computer_vision/face_auth_system/version2/trained_models/detection/det_10g.onnx"
EMB_PATH      = "/home/coder/trong/computer_vision/face_auth_system/version2/trained_models/recognition/w600k_r50.onnx"
TEMPLATES_NPZ = "/home/coder/trong/computer_vision/face_auth_system/version2/trained_models/artifacts/templates.npz"
THRESH        = 0.58

# GPU Configuration
USE_GPU = os.environ.get('USE_GPU', '1') == '1'
if USE_GPU:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ctx_id = 0
    print(f"🚀 GPU MODE: Using {providers[0]}")
else:
    providers = ['CPUExecutionProvider']
    ctx_id = -1
    print(f"⚙️  CPU MODE: Using {providers[0]}")

# Tạo model với providers thích hợp
try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)  # Only show errors

    scrfd = get_model(DET_PATH, providers=providers)
    scrfd.prepare(ctx_id=ctx_id, input_size=(480,480), det_thresh=0.6, nms=0.5)

    rec_model = ArcFaceONNX(EMB_PATH)
    rec_model.prepare(ctx_id=ctx_id, input_size=(112,112))

    print(f"✅ Models loaded successfully on {providers[0]}")
except Exception as e:
    print(f"⚠️  {providers[0]} initialization failed: {e}")
    if USE_GPU:
        print("🔄 Falling back to CPU...")
        providers = ['CPUExecutionProvider']
        ctx_id = -1
        scrfd = get_model(DET_PATH, providers=providers)
        scrfd.prepare(ctx_id=ctx_id, input_size=(480,480), det_thresh=0.65, nms=0.5)
        rec_model = ArcFaceONNX(EMB_PATH)
        rec_model.prepare(ctx_id=ctx_id, input_size=(112,112))
        print("✅ Models loaded on CPU")

# Locks để đảm bảo thread-safety cho ONNX models
detection_lock   = Lock()
recognition_lock = Lock()
faiss_lock       = Lock()
anti_lock        = Lock()   # ✅ anti-spoof lock

def load_templates(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    names = list(d["names"])
    embs  = d["embs"].astype(np.float32)
    return names, embs

def sanitize_embs(embs: np.ndarray) -> np.ndarray:
    embs = np.asarray(embs)
    if embs.ndim == 1:
        embs = embs[None, :]
    elif embs.ndim == 3 and embs.shape[1] == 1:
        embs = embs[:, 0, :]
    elif embs.ndim > 2:
        embs = embs.reshape(embs.shape[0], -1)
    return np.ascontiguousarray(embs.astype(np.float32))

names, embs = load_templates(TEMPLATES_NPZ)
embs = sanitize_embs(embs)
faiss_index = faiss.IndexFlatIP(embs.shape[1])
print(embs.shape)
faiss_index.add(embs)
print(f"[GALLERY] identities={len(names)}  dim={embs.shape[1] if embs.size else 0}")

# ========= UTILS =========
def l2n(v):
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def embed_aligned112(img112_bgr):
    if img112_bgr.ndim == 2:
        img112_bgr = cv2.cvtColor(img112_bgr, cv2.COLOR_GRAY2BGR)
    if img112_bgr.shape[:2] != (112,112):
        img112_bgr = cv2.resize(img112_bgr, (112,112), interpolation=cv2.INTER_AREA)
    feat = rec_model.get_feat(img112_bgr)
    return l2n(feat)

def search_top(q_emb, faiss_index, topk=3):
    q_emb = sanitize_embs(q_emb)
    with faiss_lock:
        D, I = faiss_index.search(q_emb.astype(np.float32), topk)
    sims, idxs = D[0], I[0]
    if len(idxs) == 0 or idxs[0] < 0:
        return "unknown", -1.0, -1, []
    best_sim, best_idx = float(sims[0]), int(idxs[0])
    label = names[best_idx] if best_sim >= THRESH else "unknown"
    top = [(names[int(ix)], float(sims[j]), int(ix)) for j, ix in enumerate(idxs) if int(ix) >= 0]
    return label, best_sim, best_idx, top

def detect_faces(img):
    with detection_lock:
        bboxes, kpss = scrfd.detect(img, max_num=1)
    return bboxes, kpss

def recognize_face(img, kpss):
    if kpss is None or len(kpss) == 0:
        return "unknown", -1.0, -1, []
    kps = kpss[0]
    aligned = face_align.norm_crop(img, landmark=kps, image_size=112)
    with recognition_lock:
        q = embed_aligned112(aligned)
    label, best_sim, best_idx, top = search_top(q, faiss_index, topk=3)
    return label, best_sim, best_idx, top

# ================== Flask ==================
app = Flask(__name__)

from collections import OrderedDict
class SimpleCache:
    def __init__(self, maxsize=100):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.lock = Lock()
    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    def set(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)

result_cache = SimpleCache(maxsize=100)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/checkin')
def checkin():
    return render_template('checkin.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username and password:
        return render_template('checkin.html')
    else:
        return render_template('index.html')

@app.route('/face_detect', methods=['POST'])
def face_detect():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'}), 400

    file = request.files['image']
    image_bytes = file.read()
    img_hash = hashlib.md5(image_bytes).hexdigest()

    cached_result = result_cache.get(img_hash)
    if cached_result is not None:
        return jsonify(cached_result)

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        result = {'success': False, 'message': 'Invalid image'}
        result_cache.set(img_hash, result)
        return jsonify(result), 400

    # Detect 1 face
    bboxes, kpss = detect_faces(img)
    if len(bboxes) == 0:
        result = {'success': False, 'message': 'No faces detected'}
        result_cache.set(img_hash, result)
        return jsonify(result), 200

    bbox = bboxes[0]
    x1, y1, x2, y2, conf_det = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), float(bbox[4])
    if conf_det < 0.7:
        result = {'success': False, 'message': 'Low confidence detection'}
        result_cache.set(img_hash, result)
        return jsonify(result), 200

    # === Crop mặt (pad nhẹ) ===
    pad = 20
    H, W = img.shape[:2]
    x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
    x2p = min(W, x2 + pad); y2p = min(H, y2 + pad)
    face_crop = img[y1p:y2p, x1p:x2p]

    # === Quality gate (kích thước & độ nét) ===
    min_side = min(face_crop.shape[:2]) if face_crop.size else 0
    if min_side < MIN_FACE_SIZE:
        is_real = False
        spoof_confidence = 1.0
    else:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        if _lap_var(gray) < BLUR_VAR_THR:
            is_real = False
            spoof_confidence = 1.0
        else:
            # === Anti-spoof ONNX ===
            with anti_lock:
                real_prob, prob_print, prob_replay = predict_anti_spoof_facecrop(face_crop)
                print("=== Anti-spoofing Results ===")
                print(f"Anti-spoof scores - real: {real_prob:.4f}, print: {prob_print:.4f}, replay: {prob_replay:.4f}")
                print("=============================")
            is_real = bool(real_prob >= LIVE_THRESHOLD)
            # spoof_confidence = max(prob_print, prob_replay)  # nếu muốn hiểu là “xác suất tấn công”
            spoof_confidence = float(1.0 - real_prob)          # tổng các lớp spoof
    # with anti_lock:
    #     real_prob, prob_print, prob_replay = predict_anti_spoof_facecrop(face_crop)
    #     print("=== Anti-spoofing Results ===")
    #     print(f"Anti-spoof scores - real: {real_prob:.4f}, print: {prob_print:.4f}, replay: {prob_replay:.4f}")
    #     print("=============================")
    # is_real = bool(real_prob >= LIVE_THRESHOLD)
    # # spoof_confidence = max(prob_print, prob_replay)  # nếu muốn hiểu là “xác suất tấn công”
    # spoof_confidence = float(1.0 - real_prob)          # tổng các lớp spoof
    # Nếu spoof -> có thể trả thẳng, không nhận diện
    if not is_real:
        result = {
            'success': True,
            'message': 'Face detected (spoofed)',
            'faces_count': 1,
            'bbox': [x1, y1, x2, y2],
            'confidence': conf_det,
            'employee_id': None,
            'employee_name': 'unknown',
            'similarity': 0.0,
            'department': None,
            'is_real': False,
            'spoof_score': spoof_confidence
        }
        result_cache.set(img_hash, result)
        return jsonify(result), 200

    # === Recognition khi đã pass anti-spoof ===
    label, best_sim, best_idx, top = recognize_face(img, kpss)

    result = {
        'success': True,
        'message': 'Face detected successfully',
        'faces_count': 1,
        'bbox': [x1, y1, x2, y2],
        'confidence': conf_det,
        'employee_id': 'NV02',
        'employee_name': label,
        'similarity': best_sim,
        'department': 'Kỹ thuật',
        'is_real': is_real,
        'spoof_score': float(spoof_confidence),
    }

    result_cache.set(img_hash, result)
    return jsonify(result), 200

if __name__ == '__main__':
    print("Đang khởi chạy ứng dụng Flask...")
    print("Truy cập http://localhost:5000/ để xem trang chấm công")
    print("⚡ Multi-threading enabled for better performance")
    app.run(host='localhost', port=5000, debug=False, threaded=True)
