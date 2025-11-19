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
    Per-relation SAGEConv modules stored using string-safe keys.
    """
    def __init__(self, hidden_dim, relation_names):
        super().__init__()
        self.hidden_dim = hidden_dim

        # map tuple → safe string name
        self.rel_keys = {rel: f"{rel[0]}__{rel[1]}__{rel[2]}" for rel in relation_names}
        self.relation_names = relation_names

        self.convs = nn.ModuleDict()
        for rel in relation_names:
            key = self.rel_keys[rel]
            self.convs[key] = SAGEConv(hidden_dim, hidden_dim)

        self.project = nn.Sequential(
            nn.Linear(hidden_dim * len(relation_names), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x_dict, edge_index_dict, dst_type):
        rel_embs = []
        for rel in self.relation_names:
            key = self.rel_keys[rel]
            src_type, _, dst_t = rel

            # If relation does not target this node type, append zeros
            if dst_t != dst_type:
                rel_embs.append(torch.zeros_like(x_dict[dst_type]))
                continue

            # If edge is missing → zeros
            if rel not in edge_index_dict:
                rel_embs.append(torch.zeros_like(x_dict[dst_type]))
                continue

            edge_index = edge_index_dict[rel].to(x_dict[src_type].device)

            # Run relation-specific conv
            out = self.convs[key]((x_dict[src_type], x_dict[dst_type]), edge_index)
            rel_embs.append(out)

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
    Hybrid GNN:
      - Node encoders (user & merchant)
      - Relation summarizer (per relation)
      - Core intensifier
      - Edge-level classifier
      - forward_nodes() used once per epoch
      - forward_edges() used for batch edge scoring
    """
    def __init__(self, user_in_dim, merchant_in_dim, edge_attr_dim,
                 hidden_dim=128, relation_names=None, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        if relation_names is None:
            relation_names = [
                ('user','transacts','merchant'),
                ('merchant','receives','user')
            ]
        self.relation_names = relation_names

        # Node encoders
        self.user_encoder = NodeEncoder(user_in_dim, hidden_dim, dropout)
        self.merchant_encoder = NodeEncoder(merchant_in_dim, hidden_dim, dropout)

        # Two separate summarizers (for user & merchant destination nodes)
        self.rel_summary_user = RelationshipSummarizer(hidden_dim, relation_names)
        self.rel_summary_merch = RelationshipSummarizer(hidden_dim, relation_names)

        # Intensifier (post-aggregation feature booster)
        self.intensifier = CoreNodeIntensifier(hidden_dim, dropout)

        # Final node projection
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge classifier
        self.edge_classifier = FraudEdgeClassifier(hidden_dim, edge_attr_dim, dropout)

    # ---------------------------------------------------------
    # 1. Compute all node embeddings once
    # ---------------------------------------------------------
    def forward_nodes(self, data):
        device = data['user'].x.device

        # Encode raw node features
        h_user = self.user_encoder(data['user'].x.to(device))
        h_merchant = self.merchant_encoder(data['merchant'].x.to(device))

        # Build dict for summarizer
        x_dict = {'user': h_user, 'merchant': h_merchant}
        edge_index_dict = data.edge_index_dict

        # Relation summarization for destination = user
        h_u_rel = self.rel_summary_user(x_dict, edge_index_dict, dst_type='user')

        # Relation summarization for destination = merchant
        h_m_rel = self.rel_summary_merch(x_dict, edge_index_dict, dst_type='merchant')

        # Add relation signals
        h_user = h_user + h_u_rel
        h_merchant = h_merchant + h_m_rel

        # Intensify
        h_user = self.intensifier(h_user)
        h_merchant = self.intensifier(h_merchant)

        # Projection
        h_user = self.node_proj(h_user)
        h_merchant = self.node_proj(h_merchant)

        return {
            'user': h_user,
            'merchant': h_merchant
        }

    # ---------------------------------------------------------
    # 2. Compute edge logits for a batch
    # ---------------------------------------------------------
    def forward_edges(self, node_embs, src_idx, dst_idx, edge_attr):
        src_h = node_embs['user'][src_idx]
        dst_h = node_embs['merchant'][dst_idx]
        return self.edge_classifier(src_h, dst_h, edge_attr)
