import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from tqdm import tqdm
import os

def generate_large_fraud_csv(
    output_path="transactions_large.csv",
    n_transactions=500_000,
    n_users=5000,
    n_merchants=1200,
    fraud_ratio=0.03   # 3% fraud
):
    print(f"Generating {n_transactions:,} transactions...")
    
    # ID ranges
    user_ids = np.arange(1, n_users + 1)
    merchant_ids = np.arange(1, n_merchants + 1)

    # Time range (last 90 days)
    start_time = datetime.now() - timedelta(days=90)

    # Pre-allocate arrays
    users = np.random.choice(user_ids, n_transactions)
    merchants = np.random.choice(merchant_ids, n_transactions)

    # Amount distribution (skewed: many small, few huge)
    amounts = np.exp(np.random.normal(3.0, 1.0, n_transactions)) * 10

    # Timestamps randomly increasing
    timestamps = [start_time + timedelta(seconds=i*random.randint(1, 25)) 
                  for i in range(n_transactions)]

    df = pd.DataFrame({
        "transaction_id": np.arange(1, n_transactions + 1),
        "user_id": users,
        "merchant_id": merchants,
        "amount": amounts,
        "timestamp": timestamps,
    })

    # Inject fraud patterns
    fraud = np.zeros(n_transactions, dtype=int)

    # Pattern 1: High amounts → higher fraud
    idx_high = df["amount"] > df["amount"].quantile(0.95)
    fraud[idx_high] = (np.random.rand(idx_high.sum()) < 0.15).astype(int)

    # Pattern 2: suspicious merchants
    suspicious_merchants = np.random.choice(merchant_ids, size=25, replace=False)
    idx_susp_merch = df["merchant_id"].isin(suspicious_merchants)
    fraud[idx_susp_merch] = (np.random.rand(idx_susp_merch.sum()) < 0.20).astype(int)

    # Pattern 3: Rapid-fire transactions by same user
    df_sorted = df.sort_values("timestamp")
    user_shift = df_sorted["user_id"].shift(1)
    time_diff = (df_sorted["timestamp"] - df_sorted["timestamp"].shift(1)).dt.total_seconds()
    idx_rapid = (df_sorted["user_id"] == user_shift) & (time_diff < 30)
    idx_rapid = idx_rapid.fillna(False).values
    fraud[df_sorted.index[idx_rapid]] = (np.random.rand(idx_rapid.sum()) < 0.25).astype(int)

    df["label"] = fraud

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved CSV → {output_path}")
    print(f"Fraud ratio: {df['label'].mean():.4f}")

if __name__ == "__main__":
    generate_large_fraud_csv(
        output_path="data/raw/transactions_large.csv",
        n_transactions=500_000,   # Change to 1_000_000 for 1M
        n_users=8000,
        n_merchants=2000,
        fraud_ratio=0.03
    )
