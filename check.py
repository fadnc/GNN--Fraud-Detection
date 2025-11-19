print("========= PYTHON ENVIRONMENT CHECK =========")

# ---------- TORCH ----------
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch ERROR:", e)

# ---------- TORCHVISION / AUDIO ----------
try:
    import torchvision, torchaudio
    print(f"TorchVision: {torchvision.__version__}")
    print(f"Torchaudio:  {torchaudio.__version__}")
except Exception as e:
    print("TorchVision/Torchaudio ERROR:", e)

# ---------- PYTHON BASICS ----------
try:
    import numpy, pandas, scipy, networkx
    print(f"Numpy:  {numpy.__version__}")
    print(f"Pandas: {pandas.__version__}")
    print(f"SciPy:  {scipy.__version__}")
    print(f"NetworkX: {networkx.__version__}")
except Exception as e:
    print("NumPy/Pandas/SciPy ERROR:", e)

# ---------- SKLEARN ----------
try:
    import sklearn
    print(f"Scikit-Learn: {sklearn.__version__}")
except Exception as e:
    print("Sklearn ERROR:", e)

# ---------- PYTORCH GEOMETRIC ----------
try:
    import torch_geometric
    print(f"PyG (torch-geometric): {torch_geometric.__version__}")
except Exception as e:
    print("PyG ERROR:", e)

# ---------- PY G EXTENSIONS ----------
try:
    import torch_scatter, torch_sparse, torch_cluster, torch_spline_conv
    print("PyG extensions: OK")
except Exception as e:
    print("Extensions ERROR:", e)

print("============================================")
print("If no ERRORS printed above → your setup is PERFECT.")
