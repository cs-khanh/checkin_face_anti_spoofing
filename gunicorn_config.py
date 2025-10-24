"""
Gunicorn configuration file for GPU mode
IMPORTANT: GPU mode MUST use 1 worker to avoid CUDA context sharing issues
"""
import os

# Server socket
bind = "localhost:5000"
backlog = 2048

# Worker processes - GPU MODE: MUST BE 1 WORKER
workers = 1  # CRITICAL: CUDA contexts cannot be shared across forked processes
worker_class = "gthread"  # Use threads instead of processes for concurrency
threads = 4  # 1 thread per worker for higher concurrent requests (real-time)
worker_connections = 1000
timeout = 60  # Giảm timeout xuống 60s (faster processing)
keepalive = 2

# Environment
raw_env = [
    "USE_GPU=1"  # Enable GPU mode
]

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"

# Process naming
proc_name = "face_auth_system"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Performance tuning
preload_app = False  # MUST BE FALSE for GPU: Load models AFTER fork to avoid CUDA issues
# Each worker will load its own CUDA context
max_requests = 1000  # Restart worker sau 1000 requests để tránh memory leak
max_requests_jitter = 50  # Random jitter để tránh restart đồng thời

