# Quick Start Guide

## 🚀 Chạy ngay (GPU Mode)

```bash
# Cách 1: Auto-detect (Recommended)
./run_production.sh

# Cách 2: Force GPU
./run_production.sh gpu

# Cách 3: Dùng config trực tiếp
gunicorn -c gunicorn_config.py app:app
```

## ⚙️ Chạy CPU Mode

```bash
# Cách 1: Force CPU
./run_production.sh cpu

# Cách 2: Dùng config trực tiếp
gunicorn -c gunicorn_cpu_config.py app:app

# Cách 3: Environment variable
USE_GPU=0 python app.py
```

## 📋 Config Files

| File | Mode | Workers | Threads | Preload |
|------|------|---------|---------|---------|
| `gunicorn_config.py` | GPU | 1 | 4 | No |
| `gunicorn_cpu_config.py` | CPU | 4 | 1 | Yes |

## ✅ Verify GPU Mode

Sau khi start, check logs:

**GPU Mode (Success):**
```
🚀 GPU MODE: Using CUDAExecutionProvider
✅ Models loaded successfully on CUDAExecutionProvider
[INFO] Booting worker with pid: XXXXX
```

**CPU Fallback:**
```
⚠️  CUDAExecutionProvider initialization failed: ...
🔄 Falling back to CPU...
✅ Models loaded on CPU
```

## 🐛 Common Issues

### Issue 1: CUDA Error
```
CUDA failure 3: initialization error
```

**Solution:**
```bash
# Check you're using GPU config (1 worker)
cat gunicorn_config.py | grep "workers ="
# Should show: workers = 1

# If showing workers = 4, you're using wrong config
gunicorn -c gunicorn_config.py app:app  # Correct GPU config
```

### Issue 2: No GPU Detected
```
⚠️  Warning: No GPU detected
```

**Check:**
```bash
nvidia-smi  # Should show GPU info
# If error, install NVIDIA drivers
```

### Issue 3: Slow on GPU
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Should see:
# - GPU util: 30-70%
# - Memory used: 2-4GB

# If GPU util = 0%, app is using CPU
# Fix: export USE_GPU=1
```

## 📊 Expected Performance

| Mode | Latency | Throughput |
|------|---------|------------|
| GPU | 30-50ms | ~20-25 req/s |
| CPU | 100-200ms | ~5-10 req/s |

## 💡 Tips

1. **Production:** Use `./run_production.sh` (auto-detect)
2. **Development:** Use `python app.py` (GPU auto, threaded)
3. **Monitor GPU:** `watch -n 1 nvidia-smi`
4. **Check logs:** Look for 🚀 (GPU) or ⚙️ (CPU)
5. **Multiple users:** GPU mode handles 4 concurrent with threads

## 🔗 More Info

- Full GPU guide: `README_GPU.md`
- Config details: `gunicorn_config.py` (GPU) / `gunicorn_cpu_config.py` (CPU)

