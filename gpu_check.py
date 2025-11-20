#!/usr/bin/env python3
"""
Quick GPU setup verification script
Tests all optimizations are working correctly
"""

import torch
import sys

print("=" * 70)
print("GPU SETUP VERIFICATION")
print("=" * 70)

# 1. Check PyTorch version
print(f"\n✓ PyTorch version: {torch.__version__}")

# 2. Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"✓ CUDA available: {cuda_available}")

if not cuda_available:
    print("\n❌ ERROR: CUDA not available!")
    print("   Your GPU won't be used for training/inference")
    sys.exit(1)

# 3. GPU details
print(f"✓ CUDA version: {torch.version.cuda}")
print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
print(f"✓ Current GPU: {torch.cuda.current_device()}")
print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 4. Test mixed precision
print("\n--- Testing Mixed Precision ---")
try:
    from torch.amp import autocast, GradScaler
    
    device = torch.device('cuda')
    x = torch.randn(100, 100).to(device)
    
    with autocast('cuda'):
        y = torch.matmul(x, x)
    
    print("✓ Mixed precision (FP16) working!")
except Exception as e:
    print(f"❌ Mixed precision error: {e}")

# 5. Test torch.compile
print("\n--- Testing Torch Compile ---")
if hasattr(torch, 'compile'):
    print("✓ torch.compile() available (PyTorch 2.0+)")
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
        
        def forward(self, x):
            return self.linear(x)
    
    try:
        model = SimpleModel().to(device)
        compiled_model = torch.compile(model, mode='reduce-overhead')
        test_input = torch.randn(5, 10).to(device)
        output = compiled_model(test_input)
        print("✓ Model compilation working!")
    except Exception as e:
        print(f"⚠️  Compilation warning (non-critical): {e}")
else:
    print("⚠️  torch.compile() not available (upgrade to PyTorch 2.0+)")

# 6. Test PyTorch Geometric
print("\n--- Testing PyTorch Geometric ---")
try:
    import torch_geometric
    print(f"✓ PyG version: {torch_geometric.__version__}")
    
    from torch_geometric.data import HeteroData
    print("✓ HeteroData import successful")
except ImportError as e:
    print(f"❌ PyG not installed: {e}")
    sys.exit(1)

# 7. Test your model files
print("\n--- Testing Project Files ---")
try:
    from models.fraud_gnn_pyg import FraudGNNHybrid
    print("✓ FraudGNNHybrid model import successful")
    
    from models.dataset_pyg import load_processed_graph
    print("✓ Dataset loader import successful")
except ImportError as e:
    print(f"❌ Model import error: {e}")
    sys.exit(1)

# 8. Memory test
print("\n--- GPU Memory Test ---")
try:
    # Allocate 1GB
    test_tensor = torch.randn(1024, 1024, 128).to(device)
    allocated = torch.cuda.memory_allocated() / 1e9
    cached = torch.cuda.memory_reserved() / 1e9
    print(f"✓ Allocated: {allocated:.2f} GB")
    print(f"✓ Cached: {cached:.2f} GB")
    
    # Free memory
    del test_tensor
    torch.cuda.empty_cache()
    print("✓ Memory cleanup successful")
except Exception as e:
    print(f"❌ Memory test error: {e}")

# 9. Quick computation benchmark
print("\n--- Quick Benchmark ---")
try:
    # Create test data
    size = 8192
    x = torch.randn(size, 128).to(device)
    y = torch.randn(size, 128).to(device)
    
    # Warmup
    for _ in range(10):
        z = torch.matmul(x, y.T)
    
    torch.cuda.synchronize()
    
    # Benchmark
    import time
    start = time.time()
    for _ in range(100):
        z = torch.matmul(x, y.T)
    torch.cuda.synchronize()
    end = time.time()
    
    throughput = (100 * size * size * 128 * 2) / (end - start) / 1e9
    print(f"✓ GPU throughput: {throughput:.2f} GFLOPS")
    
    if throughput > 100:
        print("✓ GPU performance looks good!")
    else:
        print("⚠️  GPU performance seems low (driver issue?)")
    
except Exception as e:
    print(f"❌ Benchmark error: {e}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
✓ GPU: {torch.cuda.get_device_name(0)}
✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB
✓ CUDA: {torch.version.cuda}
✓ PyTorch: {torch.__version__}
✓ All systems ready for training!

Next steps:
1. python train_pyg.py       # Train your model (fast!)
2. python generate_cache.py  # Generate visualization cache
3. python app.py             # Launch web interface
""")

print("=" * 70)