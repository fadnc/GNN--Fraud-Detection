import json
import os
import torch
from torch_geometric.data import HeteroData
from models.fraud_gnn_pyg import FraudGNNHybrid
from tqdm import tqdm

GRAPH_PATH = "data/processed/fraud_graph_pyg.pt"
MODEL_PATH = "best_fraudgnn.pth"
OUT_DIR = "cache"

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)

# -------------------------------------------------------------------
# LOAD GRAPH
# -------------------------------------------------------------------
print("\nLoading graph...")
graph_ck = torch.load(GRAPH_PATH, map_location="cpu")
hetero = graph_ck.get("data", graph_ck)

user_x = hetero['user'].x
merch_x = hetero['merchant'].x
edge_index = hetero['user','transacts','merchant'].edge_index
edge_attr = hetero['user','transacts','merchant'].edge_attr  # shape: (500000, 7)

num_edges = edge_index.size(1)
num_users = user_x.size(0)
num_merchants = merch_x.size(0)

print(f"Users: {num_users}, Merchants: {num_merchants}, Edges: {num_edges}")

# -------------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------------
print("\nLoading model...")
model = FraudGNNHybrid(
    user_in_dim=user_x.size(1),
    merchant_in_dim=merch_x.size(1),
    edge_attr_dim=edge_attr.size(1),  # 7
    hidden_dim=128,
    relation_names=[('user','transacts','merchant'), ('merchant','receives','user')]
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------------------------------------------------
# COMPUTE NODE EMBEDDINGS
# -------------------------------------------------------------------
print("\nComputing node embeddings...")
with torch.no_grad():
    hetero['user'].x = user_x.to(DEVICE)
    hetero['merchant'].x = merch_x.to(DEVICE)
    node_embs = model.forward_nodes(hetero)

# Store node risk initially 0
nodes = []

for i in range(num_users):
    nodes.append({
        "id": f"user_{i}",
        "type": "user",
        "risk_score": 0.0,
        "is_suspicious": False
    })

for j in range(num_merchants):
    nodes.append({
        "id": f"merch_{j}",
        "type": "merchant",
        "risk_score": 0.0,
        "is_suspicious": False
    })

# -------------------------------------------------------------------
# COMPUTE EDGE PREDICTIONS (in batches)
# -------------------------------------------------------------------
edges = []
batch_size = 4096

print("\nComputing fraud predictions for edges...")
for start in tqdm(range(0, num_edges, batch_size)):
    end = min(start + batch_size, num_edges)

    src = edge_index[0, start:end]
    dst = edge_index[1, start:end]
    attr = edge_attr[start:end]

    src_h = node_embs["user"][src.to(DEVICE)]
    dst_h = node_embs["merchant"][dst.to(DEVICE)]
    attr = attr.to(DEVICE)

    with torch.no_grad():
        logits = model.edge_classifier(src_h, dst_h, attr)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

    # store edges
    for i, p in zip(range(start, end), probs):
        u = int(edge_index[0, i])
        m = int(edge_index[1, i])
        edge = {
            "edge_id": i,
            "source": f"user_{u}",
            "target": f"merch_{m}",
            "edge_attr": edge_attr[i].tolist(),
            "amount": float(edge_attr[i][0].item()),  # assuming amount=attr[0]
            "pred_prob": float(p),
            "is_suspicious": p > 0.5
        }
        edges.append(edge)

        # update node risk
        risk_pct = p * 100
        if risk_pct > nodes[u]["risk_score"]:
            nodes[u]["risk_score"] = risk_pct
            nodes[u]["is_suspicious"] = p > 0.5
        merch_idx = num_users + m
        if risk_pct > nodes[merch_idx]["risk_score"]:
            nodes[merch_idx]["risk_score"] = risk_pct
            nodes[merch_idx]["is_suspicious"] = p > 0.5

# -------------------------------------------------------------------
# SAVE JSON
# -------------------------------------------------------------------
print("\nSaving JSON cache...")

with open(os.path.join(OUT_DIR, "nodes.json"), "w") as f:
    json.dump(nodes, f)

with open(os.path.join(OUT_DIR, "edges.json"), "w") as f:
    json.dump(edges, f)

print("\n✔ DONE — Cache ready.")
print("files: cache/nodes.json + cache/edges.json")
