import json
import random
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import pandas as pd
from torch_geometric.data import HeteroData

# ----------------------------
# Import your PyG Model
# ----------------------------
from models.fraud_gnn_pyg import FraudGNNHybrid

APP_ROOT = os.path.dirname(__file__)
GRAPH_PATH = os.path.join(APP_ROOT, "data", "processed", "fraud_graph_pyg.pt")
MODEL_PATH = os.path.join(APP_ROOT, "best_fraudgnn.pth")
CSV_PATH = os.path.join(APP_ROOT, "data", "raw", "transactions_large_clean.csv")

app = Flask(__name__, static_folder="static", template_folder="templates")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ============================================================
# 1) LOAD GRAPH (PyG HeteroData)
# ============================================================
print("Loading graph...")
graph_ck = torch.load(GRAPH_PATH, map_location="cpu")
data = graph_ck.get("data", graph_ck)  # supports both {data:...} or direct
hetero = data  # expecting HeteroData

# ============================================================
# 2) LOAD MODEL + PRECOMPUTE NODE EMBEDDINGS
# ============================================================
print("Loading GNN model...")
model = FraudGNNHybrid(
    user_in_dim=hetero['user'].x.size(1),
    merchant_in_dim=hetero['merchant'].x.size(1),
    edge_attr_dim=hetero['user','transacts','merchant'].edge_attr.size(1),
    hidden_dim=128,
    relation_names=[('user','transacts','merchant'), ('merchant','receives','user')]
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Computing cached node embeddings...")
with torch.no_grad():
    hetero['user'].x = hetero['user'].x.to(DEVICE)
    hetero['merchant'].x = hetero['merchant'].x.to(DEVICE)
    node_embs = model.forward_nodes(hetero)  # dict with 'user' and 'merchant'

# ============================================================
# 3) PREPARE EDGES LIST (FOR SAMPLING + LOOKUP)
# ============================================================
print("Preparing edge list...")
edge_index = hetero['user','transacts','merchant'].edge_index.cpu()
edge_attr = hetero['user','transacts','merchant'].edge_attr.cpu()

edges_list = []
for i in range(edge_index.size(1)):
    src = int(edge_index[0, i].item())
    dst = int(edge_index[1, i].item())
    eattr = edge_attr[i].tolist()
    # attempt to set amount at position 0 or  - fallback
    amount = float(eattr[0]) if len(eattr) > 0 else 0.0
    edges_list.append({
        "edge_id": i,
        "src": src,
        "dst": dst,
        "edge_attr": eattr,
        "amount": amount
    })

num_users = hetero['user'].x.size(0)
num_merchants = hetero['merchant'].x.size(0)

# ============================================================
# 4) CSV (node details / transactions)
# ============================================================
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    df = pd.DataFrame(columns=["src", "dst", "amount", "ts", "label"])

# ============================================================
# UTILS
# ============================================================
def batch_predict_edge_probs(edge_entries):
    """
    edge_entries: list of dicts {src: int, dst: int, edge_attr: list}
    Returns list of float probs (fraud probability)
    """
    if len(edge_entries) == 0:
        return []
    src_idx = torch.tensor([e['src'] for e in edge_entries], dtype=torch.long, device=DEVICE)
    dst_idx = torch.tensor([e['dst'] for e in edge_entries], dtype=torch.long, device=DEVICE)
    eattr = torch.tensor([e.get('edge_attr', [0.0]) for e in edge_entries], dtype=torch.float32, device=DEVICE)

    src_h = node_embs['user'][src_idx]
    dst_h = node_embs['merchant'][dst_idx]

    with torch.no_grad():
        logits = model.edge_classifier(src_h, dst_h, eattr)  # (B,2)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
    return probs

def predict_single_edge_by_id(eid):
    if eid < 0 or eid >= len(edges_list):
        raise ValueError("edge_id out of range")
    e = edges_list[eid]
    preds = batch_predict_edge_probs([{"src": e['src'], "dst": e['dst'], "edge_attr": e['edge_attr']}])
    return preds[0] if preds else 0.0

# ============================================================
# 5) SUBGRAPH SAMPLING FOR D3
#    - sample users, collect edges incident to them
#    - run one batch predict for selected edges and attach per-node risk (max of incident edges)
# ============================================================
def sample_subgraph(num_nodes_target=200, num_edges_target=1500, seed=42):
    random.seed(seed)
    sampled_users = random.sample(range(num_users), min(num_users, num_nodes_target))
    sampled_edges = [e for e in edges_list if e["src"] in sampled_users]

    if len(sampled_edges) > num_edges_target:
        sampled_edges = random.sample(sampled_edges, num_edges_target)

    sampled_merchants = sorted(set(e["dst"] for e in sampled_edges))

    # get predictions for sampled edges in a single batch
    edge_entries = [{"src": e['src'], "dst": e['dst'], "edge_attr": e['edge_attr']} for e in sampled_edges]
    probs = batch_predict_edge_probs(edge_entries)

    # attach probability to sampled_edges
    for e, p in zip(sampled_edges, probs):
        e['pred_prob'] = float(p)
        e['is_suspicious'] = p > 0.5

    # compute per-node risk (max of incident edge probs)
    user_risk = {u: 0.0 for u in sampled_users}
    merch_risk = {m: 0.0 for m in sampled_merchants}
    for e in sampled_edges:
        src = e['src']; dst = e['dst']; p = e.get('pred_prob', 0.0)
        if src in user_risk:
            user_risk[src] = max(user_risk[src], p)
        if dst in merch_risk:
            merch_risk[dst] = max(merch_risk[dst], p)

    # build nodes
    nodes = []
    for u in sampled_users:
        nodes.append({
            "id": f"user_{u}",
            "orig_id": u,
            "type": "user",
            "name": f"User {u}",
            "risk_score": int(user_risk[u]*100),
            "is_suspicious": user_risk[u] > 0.5
        })
    for m in sampled_merchants:
        nodes.append({
            "id": f"merch_{m}",
            "orig_id": m,
            "type": "merchant",
            "name": f"Merchant {m}",
            "risk_score": int(merch_risk[m]*100),
            "is_suspicious": merch_risk[m] > 0.5
        })

    links = []
    for e in sampled_edges:
        links.append({
            "edge_id": e['edge_id'],
            "source": f"user_{e['src']}",
            "target": f"merch_{e['dst']}",
            "amount": e['amount'],
            "pred_prob": float(e.get('pred_prob', 0.0)),
            "is_suspicious": bool(e.get('is_suspicious', False))
        })

    return {"nodes": nodes, "links": links}

# ============================================================
# 6) ENDPOINTS
# ============================================================
@app.route("/")
def index_route():
    return render_template("index.html")

def convert_links_to_graph(links):
    nodes = {}
    edges = []

    for e in links:
        src = e["source"]
        dst = e["target"]

        # create nodes if not present
        if src not in nodes:
            nodes[src] = {"id": src, "type": "user" if src.startswith("user_") else "merchant"}
        if dst not in nodes:
            nodes[dst] = {"id": dst, "type": "user" if dst.startswith("user_") else "merchant"}

        edges.append({
            "edge_id": e["edge_id"],
            "source": src,
            "target": dst,
            "amount": e["amount"],
            "pred_prob": e["pred_prob"],
            "is_suspicious": e["is_suspicious"]
        })

    return {
        "nodes": list(nodes.values()),
        "edges": edges
    }

@app.route("/graph")
def graph_api():
    n_nodes = int(request.args.get("nodes", 200))
    n_edges = int(request.args.get("edges", 1500))
    seed = int(request.args.get("seed", 42))
    sub = sample_subgraph(n_nodes, n_edges, seed)
    return jsonify({
        "nodes": sub["nodes"],
        "edges": sub["edges"]})
        
@app.route("/predict", methods=["POST"])
def predict_single_api():
    """
    Accepts JSON:
      { "edge_id": 123 }  OR
      { "src": 12, "dst": 345, "edge_attr": [...] }
    Returns:
      { "edge_id": ..., "fraud_prob": 0.23 }
    """
    payload = request.get_json()
    try:
        if "edge_id" in payload:
            eid = int(payload["edge_id"])
            prob = predict_single_edge_by_id(eid)
            return jsonify({"edge_id": eid, "fraud_prob": float(prob)})
        else:
            src = int(payload["src"]); dst = int(payload["dst"])
            entry = {"src": src, "dst": dst, "edge_attr": payload.get("edge_attr", [0.0])}
            prob = batch_predict_edge_probs([entry])[0]
            return jsonify({"edge_id": None, "fraud_prob": float(prob)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

# snippet for predict_edges in app.py — replace existing endpoint with this block
@app.route("/predict_edges", methods=["POST"])
def predict_edges():
    """
    Expect JSON { "edges": [ { source, target, edge_attr } ] }
    where source,target are "user_123" / "merch_456" OR integers (orig ids).
    """
    payload = request.get_json(force=True)
    edges = payload.get("edges", [])
    if not isinstance(edges, list):
        return jsonify({"error": "edges must be a list"}), 400

    src_list = []
    dst_list = []
    attr_list = []
    for e in edges:
        src_raw = e.get("source")
        dst_raw = e.get("target")
        # accept both "user_123" style or integer index
        if isinstance(src_raw, str) and src_raw.startswith("user_"):
            src = int(src_raw.split("_", 1)[1])
        else:
            src = int(src_raw)
        if isinstance(dst_raw, str) and dst_raw.startswith("merch_"):
            dst = int(dst_raw.split("_", 1)[1])
        else:
            dst = int(dst_raw)
        src_list.append(src)
        dst_list.append(dst)
        # edge_attr may be [] or list numeric
        attr = e.get("edge_attr", [])
        attr_list.append(attr)

    # convert to tensors (on DEVICE) and run model edge classifier
    src_t = torch.tensor(src_list, dtype=torch.long, device=DEVICE)
    dst_t = torch.tensor(dst_list, dtype=torch.long, device=DEVICE)
    # ensure attr_list is float tensor shape (N, F)
    if len(attr_list) == 0:
        attr_t = torch.zeros((len(src_list), 1), dtype=torch.float32, device=DEVICE)
    else:
        attr_t = torch.tensor(attr_list, dtype=torch.float32, device=DEVICE)

    # lookup cached embeddings (node_embs computed when app started)
    src_h = node_embs["user"][src_t]
    dst_h = node_embs["merchant"][dst_t]

    with torch.no_grad():
        logits = model.edge_classifier(src_h, dst_h, attr_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

    return jsonify({"fraud_probs": probs})


@app.route("/node_details")
def node_details():
    node = request.args.get("id")  # e.g., user_123
    if not node:
        return jsonify({"error":"missing id"}), 400
    if node.startswith("user_"):
        nid = int(node.replace("user_", ""))
        subset = df[df.src == nid]
    else:
        nid = int(node.replace("merch_", ""))
        subset = df[df.dst == nid]

    tx_count = len(subset)
    avg_amount = float(subset["amount"].mean()) if tx_count > 0 else 0.0
    risk = float(subset["label"].mean()) if ("label" in subset.columns and tx_count>0) else 0.0
    counterparties = subset["dst" if node.startswith("user_") else "src"].value_counts().to_dict()

    return jsonify({
        "id": node,
        "degree": tx_count,
        "risk": risk,
        "summary": {"tx_count": tx_count, "avg_amount": avg_amount},
        "top_counterparties": counterparties
    })

@app.route("/node_transactions")
def node_transactions():
    node = request.args.get("id")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 10))

    if node.startswith("user_"):
        nid = int(node.replace("user_", ""))
        subset = df[df.src == nid]
    else:
        nid = int(node.replace("merch_", ""))
        subset = df[df.dst == nid]

    total = len(subset)
    subset = subset.sort_values("ts", ascending=False)
    page_data = subset.iloc[(page - 1) * per_page : page * per_page]
    return jsonify({"transactions": page_data.to_dict(orient="records"), "total": total, "page": page})

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print("App running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
