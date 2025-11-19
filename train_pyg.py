# train_pyg.py
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import os
from models.fraud_gnn_pyg import FraudGNNHybrid
from models.dataset_pyg import load_processed_graph, make_edge_index_splits

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def train_epoch(model, data, optimizer, train_idx, batch_size=16384):
    model.train()
    total_loss = 0.0
    n_batches = 0

    edge_index = data['user','transacts','merchant'].edge_index
    edge_attr = data['user','transacts','merchant'].edge_attr
    labels = data['user','transacts','merchant'].y

    # Shuffle train indices
    perm = torch.randperm(train_idx.size(0))
    train_idx = train_idx[perm]

    for i in range(0, train_idx.size(0), batch_size):
        batch_edges = train_idx[i:i+batch_size]
        src = edge_index[0,batch_edges].to(DEVICE)
        dst = edge_index[1,batch_edges].to(DEVICE)
        e_attr = edge_attr[batch_edges].to(DEVICE)
        y = labels[batch_edges].to(DEVICE)

        # Prepare subgraph node feature tensors (select all nodes for simplicity)
        # If memory becomes an issue, implement neighbor sampling; here we use full node features
        data_to_device = {}
        # We'll move node features to device
        data['user'].x = data['user'].x.to(DEVICE)
        data['merchant'].x = data['merchant'].x.to(DEVICE)

        logits = model(data.to(DEVICE))
        # logits shape = (E_all, 2) — but to avoid computing all edges every batch, we instead compute only for batch edges:
        # For simplicity implement model.forward_edge which computes only for batch edges (but our model currently returns for all edges).
        # To keep simple: compute logits_all then select
        logits_all = logits  # (E_all,2)
        logits_batch = logits_all[batch_edges].to(DEVICE)

        loss = F.cross_entropy(logits_batch, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # move node features back to cpu to free memory if needed
        # (optional)
    return total_loss / max(1, n_batches)

@torch.no_grad()
def evaluate(model, data, split_idx):
    model.eval()
    edge_index = data['user','transacts','merchant'].edge_index
    edge_attr = data['user','transacts','merchant'].edge_attr
    labels = data['user','transacts','merchant'].y

    # move to device
    data['user'].x = data['user'].x.to(DEVICE)
    data['merchant'].x = data['merchant'].x.to(DEVICE)

    logits_all = model(data.to(DEVICE))  # (E_all,2)
    probs = F.softmax(logits_all, dim=1)[:,1].cpu().numpy()
    preds = logits_all.argmax(dim=1).cpu().numpy()
    y_true = labels.cpu().numpy()

    metrics = compute_metrics(y_true[split_idx.cpu().numpy()], preds[split_idx.cpu().numpy()], probs[split_idx.cpu().numpy()])
    return metrics

def main():
    # Load graph
    data, u_map, m_map = load_processed_graph("data/processed/fraud_graph_pyg.pt", device='cpu')
    train_idx, val_idx, test_idx = make_edge_index_splits(data)
    print("Train/Val/Test sizes:", train_idx.size(0), val_idx.size(0), test_idx.size(0))

    # dims
    user_in = data['user'].x.size(1)
    merch_in = data['merchant'].x.size(1)
    edge_attr_dim = data['user','transacts','merchant'].edge_attr.size(1)
    hidden = 128

    model = FraudGNNHybrid(user_in, merch_in, edge_attr_dim, hidden_dim=hidden).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_f1 = 0.0
    for epoch in range(1, 101):
        loss = train_epoch(model, data, optimizer, train_idx, batch_size=16384)
        val_metrics = evaluate(model, data, val_idx)
        test_metrics = evaluate(model, data, test_idx) if epoch % 10 == 0 else None
        print(f"Epoch {epoch} | Train loss {loss:.4f} | Val F1 {val_metrics['f1']:.4f} | Val AUC {val_metrics.get('auc',0):.4f}")
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), "best_fraudgnn.pth")
            print("Saved best model.")

    print("Training finished. Best val F1:", best_val_f1)

if __name__ == "__main__":
    main()
