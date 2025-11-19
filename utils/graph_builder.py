import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler
import os


class FraudGraphBuilderPyGClean:
    def __init__(self):
        self.user_scaler = StandardScaler()
        self.merchant_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()

    def load_csv(self, path):
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"[LOAD] {len(df):,} rows | fraud ratio: {df['label'].mean():.4f}")
        return df

    def extract_node_features(self, df):
        # USER AGGREGATED FEATURES
        user_agg = df.groupby("user_id").agg({
            "amount": ["mean", "std", "min", "max", "count"],
            "home_lat": "mean",
            "home_lon": "mean",
            "user_region_id": "first",
            "user_country_id": "first",
            "label": "mean"
        }).fillna(0)
        user_agg.columns = ["_".join(map(str, c)) for c in user_agg.columns]

        # MERCHANT AGGREGATED FEATURES
        merch_agg = df.groupby("merchant_id").agg({
            "amount": ["mean", "std", "min", "max", "count"],
            "merchant_lat": "mean",
            "merchant_lon": "mean",
            "merchant_region_id": "first",
            "merchant_country_id": "first",
            "merchant_category": "first",
            "label": "mean"
        }).fillna(0)
        merch_agg.columns = ["_".join(map(str, c)) for c in merch_agg.columns]

        return user_agg, merch_agg

    def build_graph(self, df, user_features, merchant_features):
        # ID mapping
        users = np.sort(df["user_id"].unique())
        merchants = np.sort(df["merchant_id"].unique())
        u2i = {u: i for i, u in enumerate(users)}
        m2i = {m: i for i, m in enumerate(merchants)}

        # Edge index
        src = df["user_id"].map(u2i).to_numpy(dtype=np.int64)
        dst = df["merchant_id"].map(m2i).to_numpy(dtype=np.int64)
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        data = HeteroData()

        # USER NODE FEATURES
        user_mat = user_features.reindex(users, fill_value=0).select_dtypes(include=[np.number]).values
        user_mat = self.user_scaler.fit_transform(user_mat)
        data["user"].x = torch.tensor(user_mat, dtype=torch.float32)

        # MERCHANT NODE FEATURES
        merch_mat = merchant_features.reindex(merchants, fill_value=0).select_dtypes(include=[np.number]).values
        merch_mat = self.merchant_scaler.fit_transform(merch_mat)
        data["merchant"].x = torch.tensor(merch_mat, dtype=torch.float32)

        # ---- EDGE FEATURES ----
        # Continuous signals
        amount_norm = (df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-9)
        hour = df["timestamp"].dt.hour / 23.0
        dow = df["timestamp"].dt.dayofweek / 6.0
        time_diff = df.groupby("user_id")["timestamp"].diff().dt.total_seconds().fillna(0) / 3600.0
        distance = df["distance_km"].astype(float)

        vel_count = df["velocity_count_1h"].astype(float)
        vel_amount = df["velocity_amount_1h"].astype(float)

        # Combine into matrix
        edge_feats = np.column_stack([
            amount_norm.to_numpy(),
            hour.to_numpy(),
            dow.to_numpy(),
            time_diff.to_numpy(),
            distance.to_numpy(),
            vel_count.to_numpy(),
            vel_amount.to_numpy(),
        ]).astype(np.float32)

        # Normalize edge features
        edge_feats = self.edge_scaler.fit_transform(edge_feats)

        data["user", "transacts", "merchant"].edge_index = edge_index
        data["user", "transacts", "merchant"].edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        data["user", "transacts", "merchant"].y = torch.tensor(df["label"].values, dtype=torch.long)

        # Reverse relationship
        rev_edge_index = torch.flip(edge_index, dims=[0])
        data["merchant", "receives", "user"].edge_index = rev_edge_index
        data["merchant", "receives", "user"].edge_attr = torch.tensor(edge_feats, dtype=torch.float32)

        print(f"[GRAPH] users={len(users)} | merchants={len(merchants)} | edges={edge_index.shape[1]}")
        print(f"[GRAPH] user feature dim = {data['user'].x.size(1)}")
        print(f"[GRAPH] merchant feature dim = {data['merchant'].x.size(1)}")
        print(f"[GRAPH] edge feature dim = {data['user','transacts','merchant'].edge_attr.size(1)}")

        return data, u2i, m2i

    def temporal_split(self, data):
        E = data["user", "transacts", "merchant"].edge_index.shape[1]
        train_end = int(0.6 * E)
        val_end = train_end + int(0.2 * E)

        train_mask = torch.zeros(E, dtype=torch.bool)
        val_mask = torch.zeros(E, dtype=torch.bool)
        test_mask = torch.zeros(E, dtype=torch.bool)

        train_mask[:train_end] = True
        val_mask[train_end:val_end] = True
        test_mask[val_end:] = True

        data["user", "transacts", "merchant"].train_mask = train_mask
        data["user", "transacts", "merchant"].val_mask = val_mask
        data["user", "transacts", "merchant"].test_mask = test_mask

        print(f"[SPLIT] train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")

    def save(self, data, u2i, m2i, out_path="data/processed/fraud_graph_pyg.pt"):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save({"data": data, "user_map": u2i, "merchant_map": m2i}, out_path)
        print(f"[SAVE] Saved graph → {out_path}")


if __name__ == "__main__":
    path = "data/raw/transactions_large_clean.csv"
    builder = FraudGraphBuilderPyGClean()

    df = builder.load_csv(path)
    ufeat, mfeat = builder.extract_node_features(df)
    data, u2i, m2i = builder.build_graph(df, ufeat, mfeat)
    builder.temporal_split(data)
    builder.save(data, u2i, m2i)
