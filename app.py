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
from collections import deque

# ===== Temporal gate state (5-frame window) =====
WINDOW_N = 5
real_buf   = deque(maxlen=WINDOW_N)
motion_buf = deque(maxlen=WINDOW_N)
blur_buf   = deque(maxlen=WINDOW_N)
size_buf   = deque(maxlen=WINDOW_N)
prev_face_gray = None

# Thêm ngưỡng motion (bạn có thể tinh chỉnh)
MOTION_THR = 10.0    # ngưỡng chuyển động (tune theo camera)

# ================== Anti-spoof (ONNX) ==================
onnx_path = "/home/coder/trong/computer_vision/face_auth_system/version2/trained_models/face_anti_spoofing/weights/antispoof_80x80.onnx"

# ONNX session (CUDA -> CPU fallback đã cấu hình bên dưới)
sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])

LIVE_THRESHOLD = 0.50        # ngưỡng quyết định live/spoof (tune theo data)
MIN_FACE_SIZE  = 120         # mặt nhỏ hơn cạnh ngắn này coi là kém chất lượng
BLUR_VAR_THR   = 250.0        # var Laplacian < BLUR_VAR_THR coi là mờ (tune theo camera)
ENABLE_SHARPEN = True

def enhance_face_auto(
    face_bgr,
    denoise_strength=8,         # 5–10: khử noise nhẹ
    clahe_clip=2.0, clahe_grid=8,
    usm_sigma=1.2,              # radius sharpen
    amount_min=0.4, amount_max=1.8,
    low_thr=15.0, high_thr=80.0,
    gamma_corr=True
):
    """
    Smart enhancer:
    1️⃣  Bilateral denoise giữ chi tiết
    2️⃣  Auto exposure (gamma correction)
    3️⃣  CLAHE + Unsharp Mask (adaptive amount)
    """
    # --- Step 1: Denoise nhẹ (Bilateral) ---
    img_dn = cv2.bilateralFilter(face_bgr, d=0,
                                 sigmaColor=denoise_strength,
                                 sigmaSpace=denoise_strength)

    # --- Step 2: Auto exposure (Gamma correction) ---
    if gamma_corr:
        ycc = cv2.cvtColor(img_dn, cv2.COLOR_BGR2YCrCb)
        y = ycc[:, :, 0]
        meanY = np.mean(y)
        gamma = np.interp(meanY, [50, 180], [1.4, 0.7])  # tối -> tăng sáng
        gamma = np.clip(gamma, 0.6, 1.8)
        table = np.array([(i / 255.0) ** (1.0 / gamma) * 255
                          for i in np.arange(256)]).astype("uint8")
        img_gamma = cv2.LUT(img_dn, table)
    else:
        img_gamma = img_dn

    # --- Step 3: CLAHE + Unsharp Mask ---
    ycc = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2YCrCb)
    y = ycc[:, :, 0]
    lapv0 = cv2.Laplacian(y, cv2.CV_64F).var()

    clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                            tileGridSize=(clahe_grid, clahe_grid))
    y_eq = clahe.apply(y)

    amount = np.interp(lapv0, [low_thr, high_thr],
                       [amount_max, amount_min])
    amount = float(np.clip(amount, amount_min, amount_max))

    blur = cv2.GaussianBlur(y_eq, (0, 0), usm_sigma)
    detail = cv2.subtract(y_eq, blur)
    y_sharp = cv2.addWeighted(y_eq, 1.0, detail, amount, 0)

    ycc[:, :, 0] = np.clip(y_sharp, 0, 255).astype(np.uint8)
    sharp_bgr = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    # --- Thống kê debug ---
    y2 = cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    lapv1 = cv2.Laplacian(y2, cv2.CV_64F).var()
    meanY2 = np.mean(y2)
    meta = {
        "lapv_before": float(lapv0),
        "lapv_after": float(lapv1),
        "amount": amount,
        "gamma": gamma,
        "meanY": meanY,
        "meanY_after": meanY2
    }
    return sharp_bgr, meta

def sharpen_face_auto(face_bgr,
                      clahe_clip=2.0, clahe_grid=8,
                      usm_sigma=1.2, amount_min=0.4, amount_max=1.8,
                      low_thr=15.0, high_thr=80.0):
    """
    Sharpen theo pipeline: YCrCb(Y) + CLAHE -> Unsharp Mask (USM),
    với 'amount' tự động dựa trên Laplacian variance ban đầu.
    Trả về: (face_bgr_sharp, meta_dict)
    """
    # --- đo độ nét ban đầu trên Y ---
    ycc = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycc[:, :, 0]
    lapv0 = cv2.Laplacian(y, cv2.CV_64F).var()

    # --- CLAHE để tăng tương phản cục bộ, giúp cạnh rõ hơn ---
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    y_eq = clahe.apply(y)

    # --- Unsharp Mask (USM) ---
    # amount tự động: ảnh càng mờ (lapv thấp) -> amount càng cao
    amount = np.interp(lapv0, [low_thr, high_thr], [amount_max, amount_min])
    amount = float(np.clip(amount, amount_min, amount_max))

    # Làm mờ nhẹ để lấy phần chi tiết
    blur = cv2.GaussianBlur(y_eq, (0, 0), usm_sigma)
    detail = cv2.subtract(y_eq, blur)              # detail = y_eq - blur
    y_sharp = cv2.addWeighted(y_eq, 1.0, detail, amount, 0)  # y_eq + amount*detail

    # Gộp lại BGR
    ycc[:, :, 0] = np.clip(y_sharp, 0, 255).astype(np.uint8)
    sharp_bgr = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    # --- đo lại lapv sau sharpen (để log/so sánh) ---
    y2 = cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    lapv1 = cv2.Laplacian(y2, cv2.CV_64F).var()

    meta = {
        "lapv_before": float(lapv0),
        "lapv_after":  float(lapv1),
        "amount":      amount
    }
    return sharp_bgr, meta

def _lap_var(gray):
    #gray = cv2.GaussianBlur(gray, (3,3), 0)
    return cv2.Laplacian(gray, cv2.CV_64F).var()
def improved_lap_var(face_bgr):
    y = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)[:,:,0]
    y = cv2.GaussianBlur(y, (3,3), 0)
    lapv = cv2.Laplacian(y, cv2.CV_64F).var()
    return lapv

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
anti_lock        = Lock() 

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
print(f"[Start App] Face Recognition ✅")

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
    
    global prev_face_gray

    # Detect 1 face
    bboxes, kpss = detect_faces(img)
    if len(bboxes) == 0:
        result = {'success': False, 'message': 'No faces detected'}
        result_cache.set(img_hash, result)

        # reset khi mất mặt
        real_buf.clear()
        motion_buf.clear()
        blur_buf.clear()
        size_buf.clear()
        prev_face_gray = None
        print("[RESET] No face detected → buffers cleared.")
        return jsonify(result), 200

    bbox = bboxes[0]
    x1, y1, x2, y2, conf_det = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), float(bbox[4])
    if conf_det < 0.7:
        result = {'success': False, 'message': 'Low confidence detection'}
        result_cache.set(img_hash, result)
        return jsonify(result), 200

    # === Crop mặt (pad nhẹ) ===
    pad = 50
    H, W = img.shape[:2]
    x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
    x2p = min(W, x2 + pad); y2p = min(H, y2 + pad)
    face_crop = img[y1p:y2p, x1p:x2p]

    # === Quality gate (kích thước & độ nét) ===
    min_side = min(face_crop.shape[:2]) if face_crop.size else 0
    #gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    #lapv = _lap_var(gray)
    #lapv = improved_lap_var(face_crop)
    # === Smart enhance: denoise + auto exposure + sharpen ===
    # face_crop_enh, enh = enhance_face_auto(face_crop)
    # print(f"[ENH] LapVar {enh['lapv_before']:.1f}→{enh['lapv_after']:.1f} "
    #     f"gamma={enh['gamma']:.2f} amount={enh['amount']:.2f} "
    #     f"meanY {enh['meanY']:.1f}→{enh['meanY_after']:.1f}")

    # # Dùng ảnh đã enhance cho blur/motion/anti-spoof
    # gray = cv2.cvtColor(face_crop_enh, cv2.COLOR_BGR2GRAY)
    # lapv = _lap_var(gray)

    if ENABLE_SHARPEN:
        face_crop_proc, shp = sharpen_face_auto(face_crop)
        print(f"[SHARP] {shp}")
    else:
        face_crop_proc = face_crop

    gray = cv2.cvtColor(face_crop_proc, cv2.COLOR_BGR2GRAY)
    lapv = _lap_var(gray)

    motion_score = 0.0
    if prev_face_gray is not None:
        h = min(prev_face_gray.shape[0], gray.shape[0])
        w = min(prev_face_gray.shape[1], gray.shape[1])
        diff = cv2.absdiff(cv2.resize(prev_face_gray, (w, h)),
                        cv2.resize(gray, (w, h)))
        motion_score = float(np.mean(diff))
    prev_face_gray = gray.copy()

    # Nếu crop invalid
    if min_side <= 0:
        result = {'success': False, 'message': 'Invalid face crop'}
        result_cache.set(img_hash, result)
        return jsonify(result), 200

    # === 2) Anti-spoof (real_prob) cho frame hiện tại ===
    with anti_lock:
        real_prob, prob_print, prob_replay = predict_anti_spoof_facecrop(face_crop)
    print(f"[FRAME {len(real_buf)}] size={min_side}px  blur={lapv:.2f}  motion={motion_score:.2f}  real={real_prob:.3f}")

    # === 3) Đưa vào buffer 5-frame ===
    size_buf.append(min_side)
    blur_buf.append(lapv)
    motion_buf.append(motion_score)
    real_buf.append(real_prob)
    # === 4) Nếu chưa đủ 5 frame → pending ===
    if len(real_buf) < WINDOW_N:
        print(f"[PENDING] collected {len(real_buf)}/{WINDOW_N} frames")
        # Có thể trả về trạng thái pending cho UI
        result = {
            'success': True,
            'message': f'Pending {len(real_buf)}/{WINDOW_N} frames',
            'faces_count': 1,
            'bbox': [x1, y1, x2, y2],
            'confidence': conf_det,
            'employee_id': None,
            'employee_name': 'unknown',
            'similarity': 0.0,
            'department': None,
            'is_real': False,
            'spoof_score': 1.0,              # tạm thời coi là spoof trong lúc chờ
            'pending': True,
            'window': len(real_buf)
        }
        result_cache.set(img_hash, result)
        return jsonify(result), 200
    # === 5) Khi đã đủ 5 frame → tính thống kê cửa sổ ===
    min_face5   = int(np.min(size_buf))
    avg_blur5   = float(np.median(blur_buf))
    avg_motion5 = float(np.mean(motion_buf))
    avg_real5   = float(np.mean(real_buf))
    print(f"[WINDOW-5] min_face={min_face5}px  blur~={avg_blur5:.2f}  motion~={avg_motion5:.2f}  real~={avg_real5:.3f}")

    # === 6) GATE tuần tự: size → motion → blur → real ===
    fail_reason = None
    is_real = False
    spoof_confidence = float(1.0 - avg_real5)

    # GATE 1: kích thước khuôn mặt
    if min_face5 < MIN_FACE_SIZE:
        fail_reason = f"Face too small ({min_face5}px < {MIN_FACE_SIZE})"
        print(f"[GATE FAIL] {fail_reason}")

    # GATE 2: chuyển động khuôn mặt
    elif avg_motion5 < MOTION_THR:
        fail_reason = f"Low motion ({avg_motion5:.2f} < {MOTION_THR})"
        print(f"[GATE FAIL] {fail_reason}")

    # GATE 3: độ nét khuôn mặt
    elif avg_blur5 < BLUR_VAR_THR:
        fail_reason = f"Image too blurry (LapVar {avg_blur5:.2f} < {BLUR_VAR_THR})"
        print(f"[GATE FAIL] {fail_reason}")

    # GATE 4: xác suất model real
    elif avg_real5 < LIVE_THRESHOLD:
        fail_reason = f"Low live probability ({avg_real5:.3f} < {LIVE_THRESHOLD})"
        print(f"[GATE FAIL] {fail_reason}")

    # Nếu qua hết 4 gate
    else:
        is_real = True
        print("[GATE PASS ✅] Face passed all 4 gates (size→motion→blur→real).")

    # === 7) Ra quyết định và reset buffer ===
    if not is_real:
        result = {
            'success': True,
            'message': f'Face spoofed ({fail_reason})',
            'faces_count': 1,
            'bbox': [x1, y1, x2, y2],
            'confidence': conf_det,
            'employee_id': None,
            'employee_name': 'unknown',
            'similarity': 0.0,
            'department': None,
            'is_real': False,
            'spoof_score': 1.0,
            'fail_reason': fail_reason,
            'pending': False,
            'window_stats': {
                'min_face': min_face5,
                'avg_blur': avg_blur5,
                'avg_motion': avg_motion5,
                'avg_real': avg_real5
            }
        }
        result_cache.set(img_hash, result)
        # Reset sau khi ra quyết định
        real_buf.clear()
        motion_buf.clear()
        blur_buf.clear()
        size_buf.clear()
        prev_face_gray = None
        print("[RESET] Cleared 5-frame buffers after FAIL decision.")
        print("[FAILURE] Returning spoofed result.")
        print("=============================================")
        return jsonify(result), 200
    # === Recognition khi đã pass anti-spoof window-5 ===
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
        'pending': False,
        'window_stats': {
            'min_face': min_face5,
            'avg_blur': avg_blur5,
            'avg_motion': avg_motion5,
            'avg_real': avg_real5
        }
    }
    result_cache.set(img_hash, result)
    real_buf.clear()
    motion_buf.clear()
    blur_buf.clear()
    size_buf.clear()
    prev_face_gray = None
    print("[RESET] Cleared 5-frame buffers after decision.")
    return jsonify(result), 200

    # if min_side < MIN_FACE_SIZE:
    #     is_real = False
    #     spoof_confidence = 1.0
    #     print(f"Face too small: {min_side}px")
    # else:
        
    #     if _lap_var(gray) < BLUR_VAR_THR:
    #         is_real = False
    #         spoof_confidence = 1.0
    #         print(f"Face too blurry: varLaplacian={_lap_var(gray):.2f}")
    #     else:
    #         # === Anti-spoof ONNX ===
    #         with anti_lock:
    #             real_prob, prob_print, prob_replay = predict_anti_spoof_facecrop(face_crop)
    #             print("=== Anti-spoofing Results ===")
    #             print(f"Anti-spoof scores - real: {real_prob:.4f}, print: {prob_print:.4f}, replay: {prob_replay:.4f}")
    #             print("=============================")
    #         is_real = bool(real_prob >= LIVE_THRESHOLD)
    #         # spoof_confidence = max(prob_print, prob_replay)  # nếu muốn hiểu là “xác suất tấn công”
    #         spoof_confidence = float(1.0 - real_prob)          # tổng các lớp spoof

    # print(f"Is real: {is_real}, spoof score: {spoof_confidence:.4f}, gray varLaplacian: {_lap_var(gray):.2f}")
    # with anti_lock:
    #     real_prob, prob_print, prob_replay = predict_anti_spoof_facecrop(face_crop)
    #     print("=== Anti-spoofing Results ===")
    #     print(f"Anti-spoof scores - real: {real_prob:.4f}, print: {prob_print:.4f}, replay: {prob_replay:.4f}")
    #     print("=============================")
    # is_real = bool(real_prob >= LIVE_THRESHOLD)
    # # spoof_confidence = max(prob_print, prob_replay)  # nếu muốn hiểu là “xác suất tấn công”
    # spoof_confidence = float(1.0 - real_prob)          # tổng các lớp spoof
    # Nếu spoof -> có thể trả thẳng, không nhận diện
    # if not is_real:
    #     result = {
    #         'success': True,
    #         'message': 'Face detected (spoofed)',
    #         'faces_count': 1,
    #         'bbox': [x1, y1, x2, y2],
    #         'confidence': conf_det,
    #         'employee_id': None,
    #         'employee_name': 'unknown',
    #         'similarity': 0.0,
    #         'department': None,
    #         'is_real': False,
    #         'spoof_score': spoof_confidence
    #     }
    #     result_cache.set(img_hash, result)
    #     return jsonify(result), 200

    # # === Recognition khi đã pass anti-spoof ===
    # label, best_sim, best_idx, top = recognize_face(img, kpss)

    # result = {
    #     'success': True,
    #     'message': 'Face detected successfully',
    #     'faces_count': 1,
    #     'bbox': [x1, y1, x2, y2],
    #     'confidence': conf_det,
    #     'employee_id': 'NV02',
    #     'employee_name': label,
    #     'similarity': best_sim,
    #     'department': 'Kỹ thuật',
    #     'is_real': is_real,
    #     'spoof_score': float(spoof_confidence),
    # }

    # result_cache.set(img_hash, result)
    # return jsonify(result), 200

if __name__ == '__main__':
    print("Đang khởi chạy ứng dụng Flask...")
    print("Truy cập http://localhost:5000/ để xem trang chấm công")
    print("⚡ Multi-threading enabled for better performance")
    app.run(host='localhost', port=5000, debug=False, threaded=True)
