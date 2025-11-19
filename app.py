import json
import random
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd

# ================================
# CONFIG
# ================================
APP_ROOT = os.path.dirname(__file__)
NODES_JSON = os.path.join(APP_ROOT, "data", "processed", "nodes.json")
EDGES_JSON = os.path.join(APP_ROOT, "data", "processed", "edges.json")
CSV_PATH   = os.path.join(APP_ROOT, "data", "raw", "transactions_large_clean.csv")

app = Flask(__name__, static_folder="static", template_folder="templates")

# ================================
# LOAD CACHED NODES + EDGES
# ================================
print("Loading cached nodes + edges JSON...")

with open(NODES_JSON, "r") as f:
    nodes_list = json.load(f)

with open(EDGES_JSON, "r") as f:
    edges_list = json.load(f)

# Build quick lookup dictionaries
nodes_by_id = {n["id"]: n for n in nodes_list}
edges_by_id = {e["edge_id"]: e for e in edges_list}

num_users     = len([n for n in nodes_list if n["type"] == "user"])
num_merchants = len([n for n in nodes_list if n["type"] == "merchant"])

print(f"Loaded {len(nodes_list)} nodes, {len(edges_list)} edges.")


# ================================
# LOAD CSV (node details + transactions)
# ================================
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    df = pd.DataFrame(columns=["src", "dst", "amount", "ts", "label"])
# Normalize CSV columns
expected_cols = ["src", "dst", "amount", "ts", "label"]

col_map = {
    "user_id": "src",
    "userid": "src",
    "user": "src",
    "source": "src",

    "merchant_id": "dst",
    "merchantid": "dst",
    "merchant": "dst",
    "target": "dst",

    "timestamp": "ts",
    "time": "ts",
}

df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

# ================================
# UTILS
# ================================
def sample_subgraph(
    num_nodes_target=200,
    num_edges_target=1500,
    seed=42
):
    """
    Create a subgraph:
    - randomly sample users
    - include edges where src ∈ sampled_users
    - include merchants touched by these edges
    """

    random.seed(seed)

    # ---- 1) sample N users ----
    all_users = [n for n in nodes_list if n["type"] == "user"]
    sampled_users = random.sample(
        all_users,
        min(num_nodes_target, len(all_users))
    )

    # extract numeric user IDs (safe)
    sampled_user_ids = {
        n.get("orig_id", int(n["id"].replace("user_", "")))
        for n in sampled_users
    }

    # ---- 2) get edges belonging to sampled users ----
    relevant_edges = []
    for e in edges_list:
        src_id = int(e["source"].replace("user_", ""))
        if src_id in sampled_user_ids:
            relevant_edges.append(e)

    # limit edges
    if len(relevant_edges) > num_edges_target:
        relevant_edges = random.sample(relevant_edges, num_edges_target)

    # ---- 3) collect merchants ----
    merchant_ids = {
        int(e["target"].replace("merch_", ""))
        for e in relevant_edges
    }

    # FIX: merchant orig_id always missing → compute from id string
    sampled_merchants = [
        n for n in nodes_list
        if n["type"] == "merchant"
        and int(n["id"].replace("merch_", "")) in merchant_ids
    ]

    # ---- 4) Build response ----
    nodes = sampled_users + sampled_merchants
    links = relevant_edges

    return {
        "nodes": nodes,
        "edges": links
    }



# ================================
# ROUTES
# ================================

@app.route("/")
def index_route():
    return render_template("index.html")


@app.route("/graph")
def graph_api():
    n_nodes = int(request.args.get("nodes", 20))
    n_edges = int(request.args.get("edges", 150))
    seed    = int(request.args.get("seed", 42))

    sub = sample_subgraph(n_nodes, n_edges, seed)

    return jsonify({
        "nodes": sub["nodes"],
        "edges": sub["edges"]
    })


@app.route("/predict", methods=["POST"])
def predict_single_api():
    """
    Now simply returns the stored pred_prob from cached edges.json
    """
    payload = request.get_json()

    try:
        if "edge_id" in payload:
            eid = int(payload["edge_id"])
            if eid in edges_by_id:
                return jsonify({
                    "edge_id": eid,
                    "fraud_prob": edges_by_id[eid]["pred_prob"]
                })
            else:
                return jsonify({"error": "invalid edge id"}), 400

        else:
            return jsonify({"error": "edge_id required"}), 400

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/predict_edges", methods=["POST"])
def predict_edges():
    """
    Now simply returns cached pred_prob for each requested edge
    """
    payload = request.get_json()
    edges = payload.get("edges", [])

    preds = []
    for e in edges:
        eid = e.get("edge_id")
        if eid is not None and eid in edges_by_id:
            preds.append(edges_by_id[eid]["pred_prob"])
        else:
            preds.append(0.0)

    return jsonify({"fraud_probs": preds})


@app.route("/node_details")
def node_details():
    node = request.args.get("id")
    if not node:
        return jsonify({"error": "missing id"}), 400

    # extract node type + index
    if node.startswith("user_"):
        nid = int(node.replace("user_", ""))
        subset = df[df.src == nid]
    else:
        nid = int(node.replace("merch_", ""))
        subset = df[df.dst == nid]

    tx_count = len(subset)
    avg_amount = float(subset["amount"].mean()) if tx_count > 0 else 0.0
    risk = float(subset["label"].mean()) if ("label" in subset.columns and tx_count > 0) else 0.0
    counterparties = (
        subset["dst" if node.startswith("user_") else "src"]
        .value_counts()
        .to_dict()
    )

    return jsonify({
        "id": node,
        "degree": tx_count,
        "risk": risk,
        "summary": {
            "tx_count": tx_count,
            "avg_amount": avg_amount
        },
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
    return jsonify({
        "transactions": page_data.to_dict(orient="records"),
        "total": total,
        "page": page
    })


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ================================
# RUN SERVER
# ================================
if __name__ == "__main__":
    print("App running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)