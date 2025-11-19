packages = [
    "flask",
    "gunicorn",
    "torch",
    "torchvision",
    "torchaudio",
    "onnxruntime",
    "insightface",
    "numpy",
    "scipy",
    "cv2",       # opencv-python
    "PIL",       # Pillow
    "faiss"
]

for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, "__version__", "Unknown version")
        print(f"{pkg}: {version}")
    except ImportError:
        print(f"{pkg}: NOT INSTALLED")

# Kiểm tra FAISS GPU số GPU
try:
    import faiss
    print(f"\nFAISS GPU detected: {faiss.get_num_gpus()} GPU(s) available")
except ImportError:
    print("\nFAISS not installed")
