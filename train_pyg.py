"""
GPU-Optimized Training for RTX 3050 (4GB VRAM)
Key optimizations:
1. Mixed precision training (AMP)
2. Gradient accumulation for larger effective batch size
3. Persistent data on GPU
4. Optimized batch sizes for 4GB VRAM
5. Gradient checkpointing for memory efficiency
6. Torch compile for faster execution
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import os
from models.fraud_gnn_pyg import FraudGNNHybrid
from models.dataset_pyg import load_processed_graph, make_edge_index_splits
from torch.amp import autocast, GradScaler

# GPU configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_MIXED_PRECISION = True  # FP16 training for faster speed
GRADIENT_ACCUMULATION_STEPS = 2  # Simulate larger batch size
USE_COMPILE = True  # PyTorch 2.0+ compilation

print("=" * 60)
print("GPU-Optimized GNN Training")
print("=" * 60)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Mixed Precision: {USE_MIXED_PRECISION}")
    print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}x")
print("=" * 60)

def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    metrics['precision'], metrics['recall'], metrics['f1'] = p, r, f
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = 0.0
    return metrics

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    probs = F.softmax(logits, dim=1)
    pt = probs[range(len(targets)), targets]
    logp = torch.log(pt + 1e-12)
    loss = -alpha * ((1 - pt) ** gamma) * logp
    return loss.mean()

def train_epoch(model, data, optimizer, train_idx, scaler, batch_size=8192,
                use_focal=True, gradient_accumulation_steps=2):
    """Optimized training epoch with mixed precision and gradient accumulation"""
    
    model.train()
    total_loss = 0.0
    n_batches = 0

    edge_index = data['user','transacts','merchant'].edge_index
    edge_attr = data['user','transacts','merchant'].edge_attr
    labels = data['user','transacts','merchant'].y

    # Move node features to GPU once (persistent)
    data['user'].x = data['user'].x.to(DEVICE)
    data['merchant'].x = data['merchant'].x.to(DEVICE)

    # Shuffle training indices
    perm = torch.randperm(train_idx.size(0), device=train_idx.device)
    train_idx = train_idx[perm]
    
    # Zero gradients at start
    optimizer.zero_grad()

    for batch_num, i in enumerate(range(0, train_idx.size(0), batch_size)):
        batch_edges = train_idx[i:i+batch_size]

        # FIXED: Recompute node embeddings for EACH batch to avoid graph reuse
        from torch import amp
        with amp.autocast("cuda", enabled=USE_MIXED_PRECISION):
            node_embs = model.forward_nodes(data)

        src = edge_index[0, batch_edges].to(DEVICE)
        dst = edge_index[1, batch_edges].to(DEVICE)
        e_attr = edge_attr[batch_edges].to(DEVICE)
        y = labels[batch_edges].to(DEVICE)

        # Mixed precision forward pass
        with autocast(enabled=USE_MIXED_PRECISION):
            logits = model.forward_edges(node_embs, src, dst, e_attr)
            
            if use_focal:
                loss = focal_loss(logits, y, alpha=0.25, gamma=2.0)
            else:
                loss = F.cross_entropy(logits, y)
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass with mixed precision
        if USE_MIXED_PRECISION:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights after accumulation steps
        if (batch_num + 1) % gradient_accumulation_steps == 0:
            if USE_MIXED_PRECISION:
                # Unscale gradients and clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        n_batches += 1

    return total_loss / max(n_batches, 1)

@torch.no_grad()
def evaluate(model, data, split_idx, batch_size=16384):
    """Optimized evaluation with mixed precision"""
    model.eval()

    data['user'].x = data['user'].x.to(DEVICE)
    data['merchant'].x = data['merchant'].x.to(DEVICE)

    # Compute node embeddings once
    with autocast(enabled=USE_MIXED_PRECISION):
        node_embs = model.forward_nodes(data)

    edge_index = data['user','transacts','merchant'].edge_index
    edge_attr = data['user','transacts','merchant'].edge_attr
    labels = data['user','transacts','merchant'].y

    preds, probs, y_true = [], [], []

    for i in range(0, split_idx.size(0), batch_size):
        batch_edges = split_idx[i:i+batch_size]

        src = edge_index[0, batch_edges].to(DEVICE)
        dst = edge_index[1, batch_edges].to(DEVICE)
        e_attr = edge_attr[batch_edges].to(DEVICE)
        y = labels[batch_edges].to(DEVICE)

        with autocast(enabled=USE_MIXED_PRECISION):
            logits = model.forward_edges(node_embs, src, dst, e_attr)
            p = F.softmax(logits, dim=1)[:,1].cpu().numpy()
            pr = logits.argmax(dim=1).cpu().numpy()

        preds.append(pr)
        probs.append(p)
        y_true.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    probs = np.concatenate(probs)
    y_true = np.concatenate(y_true)

    return compute_metrics(y_true, preds, probs)

def main():
    print("\n📊 Loading graph data...")
    data, u_map, m_map = load_processed_graph("data/processed/fraud_graph_pyg.pt", device='cpu')
    train_idx, val_idx, test_idx = make_edge_index_splits(data)
    
    print(f"✓ Graph loaded")
    print(f"  Train: {train_idx.size(0):,} edges")
    print(f"  Val: {val_idx.size(0):,} edges")
    print(f"  Test: {test_idx.size(0):,} edges")

    # Model dimensions
    user_in = data['user'].x.size(1)
    merch_in = data['merchant'].x.size(1)
    edge_attr_dim = data['user','transacts','merchant'].edge_attr.size(1)
    hidden = 128

    print(f"\n🧠 Initializing model...")
    print(f"  User features: {user_in}")
    print(f"  Merchant features: {merch_in}")
    print(f"  Edge features: {edge_attr_dim}")
    print(f"  Hidden dim: {hidden}")

    model = FraudGNNHybrid(user_in, merch_in, edge_attr_dim, hidden_dim=hidden).to(DEVICE)
    
    # Compile model for faster execution (PyTorch 2.0+)
    if USE_COMPILE and hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')
        print("  ✓ Model compiled")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=USE_MIXED_PRECISION)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    print(f"\n🚀 Starting training...")
    print(f"  Epochs: 100")
    print(f"  Batch size: 8192")
    print(f"  Effective batch: {8192 * GRADIENT_ACCUMULATION_STEPS} (with accumulation)")
    print("=" * 60)

    best_val_f1 = 0.0
    patience_counter = 0
    early_stop_patience = 15

    for epoch in range(1, 101):
        # Training
        loss = train_epoch(
            model, data, optimizer, train_idx, scaler,
            batch_size=8192,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
        )
        
        # Validation
        val_metrics = evaluate(model, data, val_idx)
        
        # Update learning rate
        scheduler.step(val_metrics['f1'])
        
        # Test evaluation (periodic)
        if epoch % 10 == 0:
            test_metrics = evaluate(model, data, test_idx)
            test_str = f" | Test F1 {test_metrics['f1']:.4f} AUC {test_metrics.get('auc',0):.4f}"
        else:
            test_str = ""
        
        print(f"Epoch {epoch:3d} | Loss {loss:.4f} | Val F1 {val_metrics['f1']:.4f} "
              f"AUC {val_metrics.get('auc',0):.4f}{test_str}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), "best_fraudgnn.pth")
            print(f"  ✓ Saved best model (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        # GPU memory stats (periodic)
        if epoch % 10 == 0 and torch.cuda.is_available():
            print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB / "
                  f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak")

    print("\n" + "=" * 60)
    print(f"✓ Training finished!")
    print(f"  Best validation F1: {best_val_f1:.4f}")
    print("=" * 60)

    # Final test evaluation
    print("\n📊 Final Test Evaluation...")
    model.load_state_dict(torch.load("best_fraudgnn.pth"))
    test_metrics = evaluate(model, data, test_idx)
    
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC:       {test_metrics.get('auc', 0):.4f}")

if __name__ == "__main__":
    main()