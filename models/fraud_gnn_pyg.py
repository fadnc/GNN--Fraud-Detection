# models/fraud_gnn_pyg.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, global_mean_pool
from torch_geometric.nn import to_hetero
from torch_geometric.utils import softmax

class NodeEncoder(nn.Module):
    """Simple MLP node encoder to project node features to hidden_dim."""
    def __init__(self, in_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, x):
        return self.proj(x)

class EdgeTemporalEncoder(nn.Module):
    """Optional small MLP for edge attributes (already scaled)"""
    def __init__(self, in_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, e):
        return self.net(e)

class NeighborhoodNoisePurifier(nn.Module):
    """
    Scores edges and performs gated aggregation. Implemented as attention-like gate.
    For PyG we implement a learnable edge scoring MLP and multiply messages.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def score_edges(self, src_h, dst_h):
        # src_h, dst_h: (E, hidden)
        combined = torch.cat([src_h, dst_h], dim=-1)
        return self.edge_scorer(combined).squeeze(-1)  # (E,)

class CoreNodeIntensifier(nn.Module):
    """
    Small intensifier: learns per-node importance and rescales core nodes.
    """
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        self.intensify = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, node_idx=None):
        # h: (N, hidden)
        imp = self.importance(h)  # (N,1)
        out = h + self.intensify(h) * imp
        return out

class RelationshipSummarizer(nn.Module):
    """
    Hetero aggregator: one SAGEConv per relation and combine by concat + projection.
    """
    def __init__(self, hidden_dim, relation_names):
        super().__init__()
        self.relation_names = relation_names
        self.convs = nn.ModuleDict()
        for rel in relation_names:
            # SAGEConv acts as (in, out) for homogeneous subgraph
            self.convs[rel] = SAGEConv(hidden_dim, hidden_dim)
        # after concat
        self.project = nn.Sequential(
            nn.Linear(hidden_dim * len(relation_names), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x_dict, edge_index_dict):
        # x_dict: {'user': h_u, 'merchant': h_m}
        rel_embs = []
        # For relation 'user_transacts_merchant' etc, we expect keys like ('user','transacts','merchant')
        for rel in self.relation_names:
            key = rel
            if key in edge_index_dict:
                src_type, _, dst_type = key
                edge_index = edge_index_dict[key]
                src_h = x_dict[src_type]
                # SAGEConv expects homogeneous edge_index and x for src/dst combined.
                # We'll call conv with (src_h, dst_h) pattern supported in SAGEConv forward
                conv = self.convs[key]
                h_out = conv((src_h, x_dict[dst_type]), edge_index)  # returns dst node features
                # For summarization we'll pick dst nodes for each relation's output shape aligning to dst_type
                # To combine across relations, ensure same node ordering by taking dst_type embedding
                rel_embs.append(h_out)
            else:
                # zero pad
                rel_embs.append(torch.zeros_like(next(iter(x_dict.values()))))
        combined = torch.cat(rel_embs, dim=-1)
        return self.project(combined)

class FraudEdgeClassifier(nn.Module):
    """
    Edge-level classifier: given source/dest node embeddings and edge attributes predict fraud.
    """
    def __init__(self, hidden_dim, edge_attr_dim, dropout=0.3):
        super().__init__()
        self.edge_encoder = EdgeTemporalEncoder(edge_attr_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2)  # binary logits
        )

    def forward(self, src_h, dst_h, edge_attr):
        e = self.edge_encoder(edge_attr)
        combined = torch.cat([src_h, dst_h, e], dim=-1)
        return self.mlp(combined)

class FraudGNNHybrid(nn.Module):
    """
    Hybrid model:
      - Node encoders for user & merchant
      - Hetero relation summarizer
      - Purifier + Intensifier
      - Edge classifier for transaction edges
    """
    def __init__(self, user_in_dim, merchant_in_dim, edge_attr_dim, hidden_dim=128, relation_names=None, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.user_encoder = NodeEncoder(user_in_dim, hidden_dim, dropout)
        self.merchant_encoder = NodeEncoder(merchant_in_dim, hidden_dim, dropout)

        if relation_names is None:
            relation_names = [('user','transacts','merchant'), ('merchant','receives','user')]
        self.relation_names = relation_names

        self.relation_summarizer = RelationshipSummarizer(hidden_dim, relation_names)
        self.purifier = NeighborhoodNoisePurifier(hidden_dim)
        self.intensifier = CoreNodeIntensifier(hidden_dim, dropout)
        self.edge_classifier = FraudEdgeClassifier(hidden_dim, edge_attr_dim, dropout)

        # small final projection for node embeddings
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        # data: HeteroData mini-batch containing nodes and edge_index, edge_attr for ('user','transacts','merchant')
        x_user = data['user'].x
        x_merchant = data['merchant'].x

        h_user = self.user_encoder(x_user)  # (U, hidden)
        h_merchant = self.merchant_encoder(x_merchant)  # (M, hidden)

        x_dict = {'user': h_user, 'merchant': h_merchant}
        edge_index_dict = {}
        for k in data.edge_index_dict:
            edge_index_dict[k] = data.edge_index_dict[k]

        # Relationship summarizer (returns projected combined of relations for dst nodes;
        # we will keep it simple and apply on user side by passing the dict)
        # NOTE: RelationshipSummarizer returns a tensor aligned to destination nodes - for simplicity call it once for user dest
        # For this implementation we'll just run per relation conv and aggregate manually:
        rel_emb = self.relation_summarizer({'user': h_user, 'merchant': h_merchant}, edge_index_dict)

        # Intensify node features
        h_user = self.intensifier(h_user)
        h_merchant = self.intensifier(h_merchant)

        # Project
        h_user = self.node_proj(h_user)
        h_merchant = self.node_proj(h_merchant)

        # Edge-level logits for transactions
        edge_index = data['user','transacts','merchant'].edge_index  # (2, E)
        edge_attr = data['user','transacts','merchant'].edge_attr    # (E, F)
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        src_h = h_user[src_idx]
        dst_h = h_merchant[dst_idx]

        logits = self.edge_classifier(src_h, dst_h, edge_attr)
        return logits
