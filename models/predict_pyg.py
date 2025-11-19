# models/predict_pyg.py
import torch
from models.fraud_gnn_pyg import FraudGNNHybrid

def load_model_and_graph(graph_path, model_path, device="cpu"):
    ck = torch.load(graph_path, map_location="cpu")
    data = ck.get("data", ck)
    device = torch.device(device)
    model = FraudGNNHybrid(
        user_in_dim=data['user'].x.size(1),
        merchant_in_dim=data['merchant'].x.size(1),
        edge_attr_dim=data['user','transacts','merchant'].edge_attr.size(1)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model, data
