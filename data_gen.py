import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from geopy.distance import geodesic
import os

def random_location():
    """Generate a random (lat, lon) within India."""
    lat = random.uniform(8.0, 33.0)
    lon = random.uniform(68.0, 97.0)
    return lat, lon

def generate_clean_fraud_csv(
    output_path="data/raw/transactions_large_clean.csv",
    n_transactions=500_000,
    n_users=8000,
    n_merchants=2000
):
    print(f"[GEN] Generating {n_transactions:,} transactions...")

    # --- USER PROFILES ---
    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "user_region_id": np.random.randint(1, 30, n_users),
        "user_country_id": 1  # all India for simplicity
    })

    user_locs = np.array([random_location() for _ in range(n_users)])
    users["home_lat"] = user_locs[:, 0]
    users["home_lon"] = user_locs[:, 1]

    # --- MERCHANT PROFILES ---
    merchants = pd.DataFrame({
        "merchant_id": np.arange(1, n_merchants + 1),
        "merchant_region_id": np.random.randint(1, 30, n_merchants),
        "merchant_country_id": 1,  # all India for simplicity
        "merchant_category": np.random.randint(1000, 2000, n_merchants)
    })

    merch_locs = np.array([random_location() for _ in range(n_merchants)])
    merchants["merchant_lat"] = merch_locs[:, 0]
    merchants["merchant_lon"] = merch_locs[:, 1]

    # Random draw for each transaction
    user_ids = np.random.choice(users["user_id"], n_transactions)
    merchant_ids = np.random.choice(merchants["merchant_id"], n_transactions)

    # Timestamps
    start_time = datetime.now() - timedelta(days=60)
    timestamps = [start_time + timedelta(seconds=i * random.randint(1, 40))
                  for i in range(n_transactions)]

    # Amount distribution
    amounts = np.exp(np.random.normal(3.8, 1.2, n_transactions)) * 15

    df = pd.DataFrame({
        "transaction_id": np.arange(1, n_transactions + 1),
        "user_id": user_ids,
        "merchant_id": merchant_ids,
        "amount": amounts,
        "timestamp": timestamps
    })

    # Merge profiles
    user_map = users.set_index("user_id")
    merch_map = merchants.set_index("merchant_id")
    df = df.join(user_map, on="user_id")
    df = df.join(merch_map, on="merchant_id", rsuffix="_m")

    # Distance
    distances = []
    for i in range(n_transactions):
        u = (df.iloc[i].home_lat, df.iloc[i].home_lon)
        m = (df.iloc[i].merchant_lat, df.iloc[i].merchant_lon)
        distances.append(geodesic(u, m).km)

    df["distance_km"] = np.array(distances)

    # Basic time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_night"] = (df["hour"].between(0, 6)).astype(int)

    # Velocity features
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["velocity_count_1h"] = 0
    df["velocity_amount_1h"] = 0.0

    user_last_tx = {}

    for i in range(n_transactions):
        uid = df.at[i, "user_id"]
        t = df.at[i, "timestamp"]
        amt = df.at[i, "amount"]

        if uid not in user_last_tx:
            user_last_tx[uid] = []

        # keep only last 1h
        user_last_tx[uid] = [(ts, a) for ts, a in user_last_tx[uid] if (t - ts).total_seconds() < 3600]

        df.at[i, "velocity_count_1h"] = len(user_last_tx[uid])
        df.at[i, "velocity_amount_1h"] = sum(a for ts, a in user_last_tx[uid])

        user_last_tx[uid].append((t, amt))

    # --- FRAUD RULES (REALISTIC) ---
    fraud = np.zeros(n_transactions, dtype=int)

    # Rule 1: Very high amount → ~1–2% flagged
    high_amt = df["amount"] > df["amount"].quantile(0.99)
    fraud[high_amt] = (np.random.rand(high_amt.sum()) < 0.20).astype(int)

    # Rule 2: long distance → rare (~1%)
    long_dist = df["distance_km"] > df["distance_km"].quantile(0.98)
    fraud[long_dist] = (np.random.rand(long_dist.sum()) < 0.15).astype(int)

    # Rule 3: velocity spike → strong signal
    high_vel = df["velocity_count_1h"] > 3
    fraud[high_vel] = (np.random.rand(high_vel.sum()) < 0.30).astype(int)

    df["label"] = fraud

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[DONE] Saved cleaned CSV → {output_path}")
    print(f"[INFO] Fraud ratio: {df['label'].mean():.4f}")

if __name__ == "__main__":
    generate_clean_fraud_csv()
