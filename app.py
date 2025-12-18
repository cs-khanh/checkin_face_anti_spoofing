from flask import Flask, render_template, url_for, jsonify, request, flash, redirect, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
import sys
import os
from database.connectDB import get_db_connection
# Set CUDA library path BEFORE importing onnxruntime/insightface
conda_prefix = os.environ.get('CONDA_PREFIX', '')
if not conda_prefix:
    # Hardcode path if CONDA_PREFIX not set
    conda_prefix = "/home/coder/trong/computervision/checkin_face_anti_spoofing/.env_cv"

cuda_libs = [
    f"{conda_prefix}/lib/python3.9/site-packages/nvidia/cublas/lib",
    f"{conda_prefix}/lib/python3.9/site-packages/nvidia/cudnn/lib",
    f"{conda_prefix}/lib/python3.9/site-packages/nvidia/cufft/lib",
    f"{conda_prefix}/lib/python3.9/site-packages/nvidia/cuda_runtime/lib",
]
ld_path = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = ':'.join(cuda_libs) + ':' + ld_path

import time
import numpy as np
from insightface.model_zoo import get_model
from insightface.utils import face_align
import cv2
import faiss
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from threading import Lock
from functools import lru_cache, wraps
import hashlib
import warnings
import onnxruntime as ort
from collections import deque
import time
import random
import base64
import re
from PIL import Image
import io
root_path = '/home/coder/trong/computervision/checkin_face_anti_spoofing/'
last_frame_time = None  
FRAME_TIMEOUT = 2.0  # giây

db_pool = get_db_connection()

# ===== Collect Data Config =====
COLLECT_OUTPUT_DIR = os.path.join(root_path, 'collect_output')
MAX_IMAGES_PER_PERSON = 60

# Tạo thư mục output cho collect_data nếu chưa tồn tại
if not os.path.exists(COLLECT_OUTPUT_DIR):
    os.makedirs(COLLECT_OUTPUT_DIR)

# ===== Check-in Images Config =====
CHECKIN_IMAGES_DIR = os.path.join(root_path, 'checkin_images')
if not os.path.exists(CHECKIN_IMAGES_DIR):
    os.makedirs(CHECKIN_IMAGES_DIR)

# ===== Temporal gate state (5-frame window) =====
WINDOW_N = 5
real_buf   = deque(maxlen=WINDOW_N)
motion_buf = deque(maxlen=WINDOW_N)
blur_buf   = deque(maxlen=WINDOW_N)
size_buf   = deque(maxlen=WINDOW_N)
prev_face_gray = None

# Thêm ngưỡng motion (bạn có thể tinh chỉnh)
MOTION_THR = 2.0    # ngưỡng chuyển động (tune theo camera)

# ================== Anti-spoof (ONNX) ==================
onnx_path = root_path + "trained_models/face_anti_spoofing/weights/antispoof_80x80.onnx"

# ONNX session (CUDA -> CPU fallback đã cấu hình bên dưới)
sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])

LIVE_THRESHOLD = 0.55        # ngưỡng quyết định live/spoof (tune theo data)
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
        gamma = np.clip(gamma, 0.8, 1.4)
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
DET_PATH = root_path + "trained_models/detection/det_10g.onnx"
EMB_PATH      = root_path + "trained_models/recognition/w600k_r50.onnx"
TEMPLATES_NPZ = root_path + "trained_models/recognition/artifacts/templates.npz"
THRESH        = 0.40  # cosine similarity threshold để nhận diện

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

    # Create an ONNX Runtime session with the requested providers when possible
    try:
        available_providers = ort.get_available_providers()
        providers_used = [p for p in providers if p in available_providers]
        if not providers_used:
            providers_used = ['CPUExecutionProvider']
        rec_session = ort.InferenceSession(EMB_PATH, providers=providers_used)
    except Exception:
        # fallback: let ArcFaceONNX create its own session
        rec_session = None

    # Pass the session into ArcFaceONNX so its __init__ can set input/output attrs
    if rec_session is not None:
        rec_model = ArcFaceONNX(EMB_PATH, session=rec_session)
    else:
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
        try:
            available_providers = ort.get_available_providers()
            providers_used = [p for p in providers if p in available_providers]
            if not providers_used:
                providers_used = ['CPUExecutionProvider']
            rec_session = ort.InferenceSession(EMB_PATH, providers=providers_used)
        except Exception:
            rec_session = None

        if rec_session is not None:
            rec_model = ArcFaceONNX(EMB_PATH, session=rec_session)
        else:
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

# ========= DATABASE UTILS =========
def verify_login(username, password):
    """
    Kiểm tra username và password từ database PostgreSQL
    Password trong DB phải được hash bằng werkzeug.security.generate_password_hash
    Returns: True nếu hợp lệ, False nếu không
    """
    if not db_pool:
        print("❌ Database pool not available")
        return False, 'Database unavailable'
    
    conn = None
    try:
        conn = db_pool.getconn()
        cursor = conn.cursor()
        
        # Lấy password hash từ database
        cursor.execute(
            "SELECT username, password, role FROM login WHERE username = %s",
            (username,)
        )
        result = cursor.fetchone()
        
        cursor.close()
        db_pool.putconn(conn)
        
        if result:
            stored_username, stored_password_hash, role = result
            # So sánh password hash
            if check_password_hash(stored_password_hash, password):
                # Only allow users with role 'admin' to login
                try:
                    role_normalized = (role or '').strip().lower()
                except Exception:
                    role_normalized = ''
                if role_normalized != 'admin':
                    msg = f"Tài khoản '{stored_username}' không đủ quyền (admin required)"
                    print(f"❌ Login denied: user {stored_username} has role '{role}' (admin required)")
                    return False, msg
                print(f"✅ Login successful: {stored_username} (role: {role})")
                return True, None
            else:
                print(f"❌ Login failed: Invalid password for {username}")
                return False, 'Tên đăng nhập hoặc mật khẩu không đúng'
        else:
            print(f"❌ Login failed: User {username} not found")
            return False, 'Tên đăng nhập hoặc mật khẩu không đúng'
            
    except Exception as e:
        print(f"❌ Database error during login: {e}")
        if conn:
            db_pool.putconn(conn)
        return False, f'Lỗi database: {e}'

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

def parse_label_from_npz(label):
    """
    Parse label từ npz format: 'nv01_nguyen_van_a' 
    Returns: (emp_code, full_name)
    VD: 'nv01_nguyen_van_a' -> ('NV01', 'nguyen van a')
    """
    if not label or label == 'unknown':
        return None, 'unknown'
    
    parts = label.split('_')
    if len(parts) < 2:
        return None, label
    
    # Phần đầu là emp_code (in hoa)
    emp_code = parts[0].upper()
    # Phần còn lại là tên (nối lại với dấu cách, giữ nguyên case)
    full_name = ' '.join(parts[1:])
    
    return emp_code, full_name


def search_top(q_emb, faiss_index, topk=3):
    q_emb = sanitize_embs(q_emb)
    with faiss_lock:
        D, I = faiss_index.search(q_emb.astype(np.float32), topk)
    sims, idxs = D[0], I[0]
    if len(idxs) == 0 or idxs[0] < 0:
        return "unknown", -1.0, -1, [], None
    best_sim, best_idx = float(sims[0]), int(idxs[0])
    label = names[best_idx] if best_sim >= THRESH else "unknown"
    
    # FIXED: Parse label để lấy emp_code và full_name
    emp_code, full_name = parse_label_from_npz(label)
    
    top = [(names[int(ix)], float(sims[j]), int(ix)) for j, ix in enumerate(idxs) if int(ix) >= 0]
    return label, best_sim, best_idx, top, emp_code

def detect_faces(img):
    with detection_lock:
        bboxes, kpss = scrfd.detect(img, max_num=1)
    return bboxes, kpss

def recognize_face(img, kpss):
    if kpss is None or len(kpss) == 0:
        return "unknown", -1.0, -1, [], None
    kps = kpss[0]
    aligned = face_align.norm_crop(img, landmark=kps, image_size=112)
    with recognition_lock:
        q = embed_aligned112(aligned)
    label, best_sim, best_idx, top, emp_code = search_top(q, faiss_index, topk=3)
    return label, best_sim, best_idx, top, emp_code

# ================== Flask ==================
app = Flask(__name__)
# Secret key needed for flashing messages
app.secret_key = os.environ.get('FLASK_SECRET', 'change-me-in-production')
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

def generate_captcha():
    """Tạo một phép toán ngẫu nhiên và trả về (câu hỏi, đáp án)"""
    num1 = random.randint(1, 100)
    num2 = random.randint(1, 100)
    operators = ['+', '-', '*']
    operator = random.choice(operators)
    
    if operator == '+':
        answer = num1 + num2
        question = f"{num1} + {num2}"
    elif operator == '-':
        answer = num1 - num2
        question = f"{num1} - {num2}"
    else:  # '*'
        answer = num1 * num2
        question = f"{num1} × {num2}"
    
    return question, answer

@app.route('/')
def index():
    # Tạo CAPTCHA mới và lưu vào session
    question, answer = generate_captcha()
    session['captcha_answer'] = answer
    return render_template('index.html', captcha_question=question)

@app.route('/refresh_captcha')
def refresh_captcha():
    """API để làm mới CAPTCHA"""
    question, answer = generate_captcha()
    session['captcha_answer'] = answer
    return jsonify({'question': question})

@app.route('/checkin')
def checkin():
    return render_template('checkin.html')

@app.route('/manage', methods=['GET', 'POST'])
def manage():
    if request.method == 'GET':
        # Kiểm tra xem user đã đăng nhập chưa
        if session.get('logged_in'):
            # Load data từ database
            employees = []
            logs = []
            models = []
            
            # Query employees và logs từ database
            if db_pool:
                try:
                    conn = db_pool.getconn()
                    cursor = conn.cursor()
                    
                    # Lấy danh sách nhân viên (JOIN với login để lấy role)
                    cursor.execute("""
                        SELECT 
                            e.emp_code, 
                            e.full_name, 
                            e.status, 
                            e.face_file_uri, 
                            e.created_at,
                            l.role
                        FROM employees e
                        LEFT JOIN login l ON e.emp_code = l.emp_code
                        ORDER BY e.emp_code ASC
                    """)
                    emp_rows = cursor.fetchall()
                    for row in emp_rows:
                        employees.append({
                            'id': row[0],           # emp_code
                            'name': row[1],         # full_name
                            'status': row[2],       # status (active/inactive/suspended)
                            'face_uri': row[3],     # face_file_uri
                            'created_at': row[4],   # created_at
                            'role': row[5] or 'user'  # role từ login table (admin/root/user)
                        })
                    
                    # Lấy logs gần đây (15 events gần nhất)
                    cursor.execute("""
                        SELECT ce.event_time, e.full_name, ce.emp_code, ce.match_score, ce.device_name
                        FROM checkin_events ce
                        LEFT JOIN employees e ON ce.emp_code = e.emp_code
                        ORDER BY ce.event_time DESC
                        LIMIT 15
                    """)
                    log_rows = cursor.fetchall()
                    for row in log_rows:
                        event_time = row[0].strftime('%Y-%m-%d %H:%M:%S') if row[0] else 'N/A'
                        emp_name = row[1] or 'Unknown'
                        emp_code = row[2] or 'N/A'
                        match_score = f"{row[3]*100:.1f}%" if row[3] is not None else 'N/A'
                        device = row[4] or 'Unknown'
                        logs.append(f"[{event_time}] {emp_name} ({emp_code}) - Score: {match_score} - Device: {device}")
                    
                    cursor.close()
                    db_pool.putconn(conn)
                    
                except Exception as e:
                    print(f"❌ Error loading manage data from DB: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Lấy danh sách models từ thư mục trained_models
            models_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
            if os.path.exists(models_dir):
                for filename in os.listdir(models_dir):
                    filepath = os.path.join(models_dir, filename)
                    if os.path.isfile(filepath):
                        size_bytes = os.path.getsize(filepath)
                        # Format size
                        if size_bytes < 1024:
                            size_str = f"{size_bytes} B"
                        elif size_bytes < 1024*1024:
                            size_str = f"{size_bytes/1024:.1f} KB"
                        else:
                            size_str = f"{size_bytes/(1024*1024):.1f} MB"
                        
                        models.append({
                            'name': filename,
                            'size': size_str
                        })
            
            return render_template('manage.html', employees=employees, logs=logs, models=models)
        else:
            flash('Vui lòng đăng nhập để truy cập trang này.', 'warning')
            return redirect(url_for('index'))
    
    # POST request - xử lý đăng nhập
    username = request.form.get('username')
    password = request.form.get('password')
    captcha_answer = request.form.get('captcha')
    
    # Kiểm tra CAPTCHA trước
    try:
        user_answer = int(captcha_answer) if captcha_answer else None
        correct_answer = session.get('captcha_answer')
        
        if user_answer is None or correct_answer is None or user_answer != correct_answer:
            flash('CAPTCHA không đúng. Vui lòng thử lại.', 'danger')
            return redirect(url_for('index'))
    except ValueError:
        flash('CAPTCHA phải là một số.', 'danger')
        return redirect(url_for('index'))
    
    # Kiểm tra username và password từ database
    success, msg = verify_login(username, password)
    if success:
        # Xóa CAPTCHA và đánh dấu đã đăng nhập
        session.pop('captcha_answer', None)
        session['logged_in'] = True
        session['username'] = username
        # Redirect để tránh form resubmit khi F5
        return redirect(url_for('home'))
    else:
        flash(msg or 'Tên đăng nhập hoặc mật khẩu không đúng', 'danger')
        return redirect(url_for('index'))

@app.route('/home')
def home():
    """Trang home sau khi đăng nhập"""
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập trang này.', 'warning')
        return redirect(url_for('index'))
    return render_template('home.html')

@app.route('/logout')
def logout():
    """Đăng xuất và xóa session"""
    session.clear()
    flash('Đã đăng xuất thành công.', 'success')
    return redirect(url_for('index'))

@app.route('/monitor')
def monitor():
    """Chuyển hướng đến Grafana dashboard"""
    # Kiểm tra đăng nhập
    if not session.get('logged_in'):
        flash('Vui lòng đăng nhập để truy cập trang này.', 'warning')
        return redirect(url_for('index'))
    
    # Redirect trực tiếp đến Grafana
    return redirect("https://view.csenguyenminhphuc.id.vn/")

@app.route('/api/checkin', methods=['POST'])
def api_checkin():
    """API để ghi nhận sự kiện check-in"""
    try:
        data = request.get_json()
        emp_code = data.get('emp_code')
        employee_name = data.get('employee_name', 'Unknown')
        match_score = data.get('match_score', 0.0)
        image_data = data.get('image')  # base64 image data
        
        if not emp_code:
            return jsonify({
                'success': False,
                'message': 'Thiếu mã nhân viên'
            }), 400
        
        if not db_pool:
            return jsonify({
                'success': False,
                'message': 'Database không khả dụng'
            }), 500
        
        # Save image if provided
        image_uri = None
        if image_data:
            try:
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                # Decode base64
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Convert to RGB if needed
                if image.mode in ('RGBA', 'P'):
                    image = image.convert('RGB')
                
                # Generate filename: empcode_timestamp.jpg
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                filename = f"{emp_code}_{timestamp}.jpg"
                filepath = os.path.join(CHECKIN_IMAGES_DIR, filename)
                
                # Save image
                image.save(filepath, 'JPEG', quality=95)
                
                # Store relative path for database
                image_uri = f"checkin_images/{filename}"
                
            except Exception as e:
                print(f"⚠️ Failed to save check-in image: {e}")
                # Continue without image - don't fail the check-in
        
        conn = None
        try:
            conn = db_pool.getconn()
            cursor = conn.cursor()
            
            # FIXED: Kiểm tra xem emp_code có tồn tại không (trước khi insert)
            cursor.execute("SELECT emp_code, full_name FROM employees WHERE emp_code = %s", (emp_code,))
            emp_exists = cursor.fetchone()
            
            if not emp_exists:
                conn.rollback()
                cursor.close()
                db_pool.putconn(conn)
                return jsonify({
                    'success': False,
                    'message': f'Mã nhân viên {emp_code} không tồn tại trong hệ thống.'
                }), 400
            
            # Insert checkin event with image_uri
            cursor.execute(
                """INSERT INTO checkin_events (emp_code, match_score, image_uri, device_name, note)
                   VALUES (%s, %s, %s, %s, %s)
                   RETURNING id, event_time, work_date""",
                (emp_code, match_score, image_uri, 'WEB-KIOSK', f'Check-in via web interface')
            )
            
            result = cursor.fetchone()
            if result is None:
                # Trigger dedup_checkin_event đã chặn (check-in trong vòng 30s)
                conn.rollback()
                cursor.close()
                db_pool.putconn(conn)
                return jsonify({
                    'success': False,
                    'message': f'Bạn đã check-in gần đây (trong vòng 30 giây). Vui lòng đợi thêm.'
                }), 400
                
            event_id, event_time, work_date = result
            
            conn.commit()
            cursor.close()
            db_pool.putconn(conn)
            
            return jsonify({
                'success': True,
                'message': f'Check-in thành công cho {employee_name}',
                'data': {
                    'event_id': event_id,
                    'emp_code': emp_code,
                    'employee_name': employee_name,
                    'event_time': event_time.isoformat(),
                    'work_date': work_date.isoformat(),
                    'match_score': match_score,
                    'image_uri': image_uri
                }
            })
            
        except Exception as e:
            if conn:
                conn.rollback()
                db_pool.putconn(conn)
            print(f"❌ Database error during check-in: {e}")
            return jsonify({
                'success': False,
                'message': f'Lỗi database: {str(e)}'
            }), 500
            
    except Exception as e:
        print(f"❌ Error in check-in API: {e}")
        return jsonify({
            'success': False,
            'message': f'Lỗi server: {str(e)}'
        }), 500

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
    global prev_face_gray
    global last_frame_time
    current_time = time.time()

    if last_frame_time is not None:
        delta_t = current_time - last_frame_time
        if delta_t > FRAME_TIMEOUT:
            # reset các buffer nếu ngắt quãng quá lâu
            real_buf.clear()
            motion_buf.clear()
            blur_buf.clear()
            size_buf.clear()
            prev_face_gray = None
            print(f"[RESET] Frame gap {delta_t:.2f}s > {FRAME_TIMEOUT}s → buffers cleared.")
    last_frame_time = current_time

    if img is None:
        result = {'success': False, 'message': 'Invalid image'}
        result_cache.set(img_hash, result)
        return jsonify(result), 400

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
    face_crop_enh, enh = enhance_face_auto(face_crop)
    print(f"[ENH] LapVar {enh['lapv_before']:.1f}→{enh['lapv_after']:.1f} "
        f"gamma={enh['gamma']:.2f} amount={enh['amount']:.2f} "
        f"meanY {enh['meanY']:.1f}→{enh['meanY_after']:.1f}")

    # Dùng ảnh đã enhance cho blur/motion/anti-spoof
    gray = cv2.cvtColor(face_crop_enh, cv2.COLOR_BGR2GRAY)
    lapv = enh['lapv_after']

    # if ENABLE_SHARPEN:
    #     face_crop_proc, shp = sharpen_face_auto(face_crop)
    #     print(f"[SHARP] {shp}")
    # else:
    #     face_crop_proc = face_crop

    # gray = cv2.cvtColor(face_crop_proc, cv2.COLOR_BGR2GRAY)
    # lapv = _lap_var(gray)

    motion_score = 0.0
    if prev_face_gray is not None:
        h = min(prev_face_gray.shape[0], gray.shape[0])
        w = min(prev_face_gray.shape[1], gray.shape[1])
        diff = cv2.absdiff(cv2.resize(prev_face_gray, (w, h)),
                        cv2.resize(gray, (w, h)))
        motion_score = np.mean(diff) * (gray.shape[0]*gray.shape[1]) / (640*640)

    prev_face_gray = gray.copy()

    # Nếu crop invalid
    if min_side <= 0:
        result = {'success': False, 'message': 'Invalid face crop'}
        result_cache.set(img_hash, result)
        return jsonify(result), 200

    # === 2) Anti-spoof (real_prob) cho frame hiện tại ===
    with anti_lock:
        real_prob, prob_print, prob_replay = predict_anti_spoof_facecrop(face_crop_enh)
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
    # Bypass gate nếu low motion nhưng real prob cao
    if avg_motion5 < MOTION_THR and avg_real5 > 0.9:
        print("[GATE BYPASS] Low motion but high real prob → pass")
        is_real = True
    # GATE 1: kích thước khuôn mặt
    elif min_face5 < MIN_FACE_SIZE:
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
    label, best_sim, best_idx, top, emp_code = recognize_face(img, kpss)

    # FIXED: Parse label để lấy emp_code và full_name
    emp_code_parsed, full_name = parse_label_from_npz(label)
    
    # Sử dụng emp_code từ npz (đã uppercase), không cần query DB nữa
    final_emp_code = emp_code_parsed or emp_code or 'unknown'
    
    result = {
        'success': True,
        'message': 'Face detected successfully',
        'faces_count': 1,
        'bbox': [x1, y1, x2, y2],
        'confidence': conf_det,
        'employee_id': final_emp_code,
        'employee_name': full_name,
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


# ================== Collect Data Functions ==================

def collect_login_required(f):
    """Decorator kiểm tra đăng nhập admin cho collect data"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('collect_admin_logged_in'):
            return redirect(url_for('collect_admin_login'))
        return f(*args, **kwargs)
    return decorated_function


def validate_employee_id(emp_id):
    """Kiểm tra định dạng mã nhân viên (NV + đúng 2 chữ số: NV01, NV02,...)"""
    pattern = r'^NV\d{2}$'
    return bool(re.match(pattern, emp_id, re.IGNORECASE))


def validate_name(name):
    """Kiểm tra họ tên (chỉ chữ cái không dấu và gạch dưới)"""
    pattern = r'^[a-zA-Z_]+$'
    return bool(re.match(pattern, name))


def get_folder_name(emp_id, name):
    """Tạo tên thư mục từ mã NV và họ tên"""
    return f"{emp_id.upper()}_{name}"


def count_images(folder_path):
    """Đếm số ảnh trong thư mục"""
    if not os.path.exists(folder_path):
        return 0
    return len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])


def get_next_image_number(folder_path):
    """Lấy số thứ tự ảnh tiếp theo"""
    if not os.path.exists(folder_path):
        return 1
    
    existing_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not existing_files:
        return 1
    
    numbers = []
    for f in existing_files:
        match = re.search(r'image_(\d+)', f)
        if match:
            numbers.append(int(match.group(1)))
    
    return max(numbers) + 1 if numbers else 1


# ================== Collect Data Routes ==================

@app.route('/collect')
def collect_index():
    """Trang thu thập ảnh chính"""
    return render_template('collect.html')


@app.route('/collect/admin/login', methods=['GET', 'POST'])
def collect_admin_login():
    """Trang đăng nhập admin cho collect data"""
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        success, msg = verify_login(username, password)
        if success:
            session['collect_admin_logged_in'] = True
            session['collect_username'] = username
            return redirect(url_for('collect_admin_gallery'))
        else:
            flash(msg or 'Sai tên đăng nhập hoặc mật khẩu!', 'danger')
            return render_template('collect_login.html', error=msg or 'Sai tên đăng nhập hoặc mật khẩu!')
    
    return render_template('collect_login.html')


@app.route('/collect/admin/logout')
def collect_admin_logout():
    """Đăng xuất admin collect data"""
    session.pop('collect_admin_logged_in', None)
    return redirect(url_for('collect_admin_login'))


@app.route('/collect/admin/gallery')
@collect_login_required
def collect_admin_gallery():
    """Trang thư viện ảnh (chỉ admin)"""
    return render_template('collect_gallery.html')


# ================== Collect Data API Endpoints ==================

@app.route('/collect/api/capture', methods=['POST'])
def collect_api_capture():
    """API lưu ảnh từ base64"""
    try:
        data = request.get_json()
        
        emp_id = data.get('emp_id', '').strip()
        name = data.get('name', '').strip()
        image_data = data.get('image', '')
        
        if not validate_employee_id(emp_id):
            return jsonify({
                'success': False,
                'message': 'Mã nhân viên không hợp lệ! Định dạng: NV + 2 chữ số (VD: NV01, NV02)'
            }), 400
        
        if not validate_name(name):
            return jsonify({
                'success': False,
                'message': 'Họ tên không hợp lệ! Chỉ chấp nhận chữ cái không dấu và gạch dưới'
            }), 400
        
        folder_name = get_folder_name(emp_id, name)
        folder_path = os.path.join(COLLECT_OUTPUT_DIR, folder_name)
        
        current_count = count_images(folder_path)
        if current_count >= MAX_IMAGES_PER_PERSON:
            return jsonify({
                'success': False,
                'message': f'Đã đạt giới hạn {MAX_IMAGES_PER_PERSON} ảnh cho người này!'
            }), 400
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.mode in ('RGBA', 'P'):
                image = image.convert('RGB')
            
            image_number = get_next_image_number(folder_path)
            filename = f"image_{image_number:03d}.jpg"
            filepath = os.path.join(folder_path, filename)
            
            image.save(filepath, 'JPEG', quality=95)
            
            return jsonify({
                'success': True,
                'message': f'Đã lưu ảnh {filename}',
                'filename': filename,
                'count': current_count + 1,
                'max': MAX_IMAGES_PER_PERSON
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Lỗi khi xử lý ảnh: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi server: {str(e)}'
        }), 500


@app.route('/collect/api/list_persons', methods=['GET'])
def collect_api_list_persons():
    """API lấy danh sách người đã thu thập"""
    try:
        persons = []
        
        if os.path.exists(COLLECT_OUTPUT_DIR):
            for folder_name in os.listdir(COLLECT_OUTPUT_DIR):
                folder_path = os.path.join(COLLECT_OUTPUT_DIR, folder_name)
                if os.path.isdir(folder_path):
                    image_count = count_images(folder_path)
                    # Get thumbnail (first image)
                    thumbnail = None
                    for f in os.listdir(folder_path):
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                            thumbnail = f
                            break
                    
                    persons.append({
                        'folder': folder_name,
                        'count': image_count,
                        'thumbnail': thumbnail
                    })
        
        persons.sort(key=lambda x: x['folder'])
        
        return jsonify({
            'success': True,
            'persons': persons
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi: {str(e)}'
        }), 500


@app.route('/collect/api/list_images', methods=['GET'])
def collect_api_list_images():
    """API lấy danh sách ảnh của một người"""
    try:
        folder = request.args.get('folder', '').strip()
        
        if not folder:
            return jsonify({
                'success': False,
                'message': 'Thiếu thông tin folder'
            }), 400
        
        folder_path = os.path.join(COLLECT_OUTPUT_DIR, folder)
        
        if not os.path.exists(folder_path):
            return jsonify({
                'success': False,
                'message': 'Không tìm thấy thư mục'
            }), 404
        
        images = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append({
                    'filename': filename,
                    'url': f'/collect/output/{folder}/{filename}'
                })
        
        images.sort(key=lambda x: x['filename'])
        
        return jsonify({
            'success': True,
            'images': images,
            'count': len(images)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi: {str(e)}'
        }), 500


@app.route('/collect/api/delete_image', methods=['POST'])
@collect_login_required
def collect_api_delete_image():
    """API xóa một ảnh"""
    try:
        data = request.get_json()
        folder = data.get('folder', '')
        filename = data.get('filename', '')
        
        if not folder or not filename:
            return jsonify({
                'success': False,
                'message': 'Thiếu thông tin folder hoặc filename'
            }), 400
        
        filepath = os.path.join(COLLECT_OUTPUT_DIR, folder, filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'message': 'Không tìm thấy file'
            }), 404
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Đã xóa {filename}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi: {str(e)}'
        }), 500


@app.route('/collect/api/delete_person', methods=['POST'])
@collect_login_required
def collect_api_delete_person():
    """API xóa toàn bộ thư mục của một người"""
    try:
        data = request.get_json()
        folder = data.get('folder', '')
        
        if not folder:
            return jsonify({
                'success': False,
                'message': 'Thiếu thông tin folder'
            }), 400
        
        folder_path = os.path.join(COLLECT_OUTPUT_DIR, folder)
        
        if not os.path.exists(folder_path):
            return jsonify({
                'success': False,
                'message': 'Không tìm thấy thư mục'
            }), 404
        
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
        
        os.rmdir(folder_path)
        
        return jsonify({
            'success': True,
            'message': f'Đã xóa thư mục {folder}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi: {str(e)}'
        }), 500


@app.route('/collect/api/get_count', methods=['GET'])
def collect_api_get_count():
    """API lấy số ảnh hiện tại của một người"""
    try:
        emp_id = request.args.get('emp_id', '').strip()
        name = request.args.get('name', '').strip()
        
        if not emp_id or not name:
            return jsonify({
                'success': True,
                'count': 0,
                'max': MAX_IMAGES_PER_PERSON
            })
        
        folder_name = get_folder_name(emp_id, name)
        folder_path = os.path.join(COLLECT_OUTPUT_DIR, folder_name)
        
        count = count_images(folder_path)
        
        return jsonify({
            'success': True,
            'count': count,
            'max': MAX_IMAGES_PER_PERSON
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi: {str(e)}'
        }), 500


@app.route('/collect/output/<path:filepath>')
def collect_serve_output(filepath):
    """Phục vụ file ảnh từ thư mục collect_output"""
    return send_from_directory(COLLECT_OUTPUT_DIR, filepath)


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
