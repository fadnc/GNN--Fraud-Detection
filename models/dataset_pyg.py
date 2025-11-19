# data/dataset_pyg.py
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
import os

def load_processed_graph(path="data/processed/fraud_graph_pyg.pt", device='cuda'):
    checkpoint = torch.load(path, map_location='cpu')
    data = checkpoint['data']
    user_map = checkpoint['user_map']
    merchant_map = checkpoint['merchant_map']
    return data, user_map, merchant_map

def make_edge_neighbor_loader(data: HeteroData, input_nodes=None,
                              num_neighbors={('user','transacts','merchant'): [10,10], ('merchant','receives','user'): [10,10]},
                              batch_size=1024, num_workers=4, shuffle=True):
    """
    Create a NeighborLoader that returns mini-batches for edge-level prediction.
    For simplicity we use the top-level loader sampling neighborhoods for nodes involved in edges.
    """
    # Build edge-level index for target relation
    # We use hetero NeighborLoader (PyG 2.0+ supports HeteroData)
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=('user','x'),  # dummy; we'll pass edge_index_mask via edge_label_index below when training
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return loader

def make_edge_index_splits(data):
    # return indices for train/val/test edges for ('user','transacts','merchant')
    mask = data['user','transacts','merchant'].train_mask
    train_idx = mask.nonzero(as_tuple=False).view(-1)
    mask = data['user','transacts','merchant'].val_mask
    val_idx = mask.nonzero(as_tuple=False).view(-1)
    mask = data['user','transacts','merchant'].test_mask
    test_idx = mask.nonzero(as_tuple=False).view(-1)
    return train_idx, val_idx, test_idx
