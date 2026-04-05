"""
Synthetic Credit Card Fraud Dataset Generator
Generates realistic transaction data with graph structure for GNN training.
"""

import numpy as np
import pandas as pd
import random
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# ─────────────────────────────────────────
# PLACEHOLDER: Real dataset loading
# ─────────────────────────────────────────
# To use a real dataset (e.g., Kaggle Credit Card Fraud Detection):
#
#   import kagglehub
#   path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
#   df = pd.read_csv(f"{path}/creditcard.csv")
#
# Or IEEE-CIS Fraud Detection:
#   df_train = pd.read_csv("path/to/train_transaction.csv")
#   df_identity = pd.read_csv("path/to/train_identity.csv")
#   df = df_train.merge(df_identity, on="TransactionID", how="left")
#
# Replace `generate_synthetic_dataset()` return value with your real df.
# ─────────────────────────────────────────

random.seed(42)
np.random.seed(42)

MERCHANT_CATEGORIES = [
    "grocery", "gas_station", "restaurant", "electronics",
    "travel", "entertainment", "online_retail", "pharmacy",
    "clothing", "utilities"
]

DEVICE_TYPES = ["mobile", "desktop", "tablet", "pos_terminal"]
CHANNELS = ["online", "in_store", "atm", "contactless"]


def generate_customers(n=5000):
    customers = []
    for i in range(n):
        customers.append({
            "customer_id": f"CUST_{i:04d}",
            "credit_limit": np.random.choice([1000, 2000, 5000, 10000, 20000]),
            "avg_monthly_spend": np.random.uniform(200, 5000),
            "fraud_history": int(np.random.random() < 0.05),
            "age": np.random.randint(18, 80),
            "risk_score": np.random.uniform(0, 1),
        })
    return pd.DataFrame(customers)


def generate_cards(customers_df):
    cards = []
    for _, cust in customers_df.iterrows():
        n_cards = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        for j in range(n_cards):
            cards.append({
                "card_id": f"CARD_{cust['customer_id']}_{j}",
                "customer_id": cust["customer_id"],
                "credit_limit": cust["credit_limit"],
                "avg_spend": cust["avg_monthly_spend"] / n_cards,
                "fraud_history": cust["fraud_history"],
            })
    return pd.DataFrame(cards)


def generate_merchants(n=2000):
    merchants = []
    for i in range(n):
        cat = random.choice(MERCHANT_CATEGORIES)
        risk = 0.8 if cat in ["electronics", "online_retail", "travel"] else np.random.uniform(0, 0.3)
        merchants.append({
            "merchant_id": f"MERCH_{i:04d}",
            "category": cat,
            "risk_score": risk,
            "avg_transaction_amount": np.random.uniform(10, 500),
            "country": np.random.choice(["US", "UK", "FR", "DE", "CN", "JP"], p=[0.6, 0.1, 0.07, 0.07, 0.1, 0.06]),
        })
    return pd.DataFrame(merchants)


def generate_devices(n=8000):
    devices = []
    for i in range(n):
        devices.append({
            "device_id": f"DEV_{i:04d}",
            "device_type": random.choice(DEVICE_TYPES),
            "reuse_count": np.random.randint(1, 20),
            "known_fraudulent": int(np.random.random() < 0.03),
        })
    return pd.DataFrame(devices)


def generate_transactions(cards_df, merchants_df, devices_df, n=200000):
    transactions = []
    start_date = datetime(2023, 1, 1)
    card_ids = cards_df["card_id"].tolist()
    merchant_ids = merchants_df["merchant_id"].tolist()
    device_ids = devices_df["device_id"].tolist()

    # Track per-card recent activity for burst / escalation patterns
    card_recent_ts = {cid: [] for cid in card_ids}       # list[datetime]
    card_recent_amt = {cid: [] for cid in card_ids}      # list[float]

    # Simulate a realistic per-card timeline (monotonic) so "burst" is meaningful.
    card_current_time = {
        cid: start_date + timedelta(days=random.randint(0, 30), seconds=random.randint(0, 24 * 3600 - 1))
        for cid in card_ids
    }

    # Build fraud ring: small cluster of colluding merchants
    fraud_ring_merchants = random.sample(merchant_ids, 5)
    fraud_ring_cards = random.sample(card_ids, 15)

    for i in range(n):
        card_id = random.choice(card_ids)
        card_row = cards_df[cards_df["card_id"] == card_id].iloc[0]
        merchant_id = random.choice(merchant_ids)
        merchant_row = merchants_df[merchants_df["merchant_id"] == merchant_id].iloc[0]
        device_id = random.choice(device_ids)
        device_row = devices_df[devices_df["device_id"] == device_id].iloc[0]

        # Advance this card's clock.
        # Mostly minutes-hours between purchases, sometimes seconds to allow bursts.
        if np.random.random() < 0.03:
            dt_seconds = random.randint(5, 120)          # bursty
        else:
            dt_seconds = random.randint(60, 6 * 3600)    # normal
        timestamp = card_current_time[card_id] + timedelta(seconds=dt_seconds)
        card_current_time[card_id] = timestamp

        # Base amount
        amount = np.random.lognormal(mean=4.0, sigma=1.2)
        amount = float(min(max(amount, 1.0), card_row["credit_limit"]))

        channel = random.choice(CHANNELS)
        is_international = int(merchant_row["country"] != "US")

        # ── Fraud labeling logic ──
        is_fraud = 0
        fraud_reason = []

        # Pattern 1: Burst transactions (5+ in 30 min)
        recent_ts = [t for t in card_recent_ts[card_id] if (timestamp - t).total_seconds() < 1800]
        if len(recent_ts) >= 4 and np.random.random() < 0.65:
            is_fraud = 1
            fraud_reason.append("burst")

        # Pattern 2: High-amount + international + high-risk merchant
        if amount > 800 and is_international and float(merchant_row["risk_score"]) > 0.6 and np.random.random() < 0.8:
            is_fraud = 1
            fraud_reason.append("amount_intl_risk")

        # Pattern 3: Known fraudulent device (probabilistic, not always fraud)
        if int(device_row["known_fraudulent"]) and np.random.random() < 0.25:
            is_fraud = 1
            fraud_reason.append("bad_device")

        # Pattern 4: Fraud ring (card + merchant both in ring)
        if card_id in fraud_ring_cards and merchant_id in fraud_ring_merchants and np.random.random() < 0.85:
            is_fraud = 1
            fraud_reason.append("fraud_ring")

        # Pattern 5: Amount escalation (sudden 10x typical for this card)
        if card_recent_amt[card_id]:
            baseline = float(np.mean(card_recent_amt[card_id][-5:]))
            if baseline > 0 and amount > 10.0 * baseline and amount > 200 and np.random.random() < 0.6:
                is_fraud = 1
                fraud_reason.append("amount_escalation")

        # Pattern 6: Random baseline fraud (~0.8%)
        if np.random.random() < 0.008:
            is_fraud = 1
            fraud_reason.append("random")

        card_recent_ts[card_id].append(timestamp)
        card_recent_amt[card_id].append(amount)
        if len(card_recent_ts[card_id]) > 30:
            card_recent_ts[card_id] = card_recent_ts[card_id][-30:]
        if len(card_recent_amt[card_id]) > 30:
            card_recent_amt[card_id] = card_recent_amt[card_id][-30:]

        transactions.append({
            "transaction_id": f"TXN_{i:06d}",
            "card_id": card_id,
            "customer_id": card_row["customer_id"],
            "merchant_id": merchant_id,
            "device_id": device_id,
            "amount": round(amount, 2),
            "timestamp": timestamp.isoformat(),
            "merchant_category": merchant_row["category"],
            "channel": channel,
            "is_international": is_international,
            "transaction_status": "approved",
            "is_fraud": is_fraud,
            "fraud_reasons": "|".join(fraud_reason) if fraud_reason else "none",
        })

    df = pd.DataFrame(transactions)
    fraud_rate = df["is_fraud"].mean()
    print(f"[DataGen] Generated {len(df)} transactions | Fraud rate: {fraud_rate:.2%}")
    return df


def generate_synthetic_dataset(output_dir="data/raw"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("[DataGen] Generating customers...")
    customers = generate_customers(5000)

    print("[DataGen] Generating cards...")
    cards = generate_cards(customers)

    print("[DataGen] Generating merchants...")
    merchants = generate_merchants(2000)

    print("[DataGen] Generating devices...")
    devices = generate_devices(8000)

    print("[DataGen] Generating transactions...")
    transactions = generate_transactions(cards, merchants, devices, n=200000)

    customers.to_csv(f"{output_dir}/customers.csv", index=False)
    cards.to_csv(f"{output_dir}/cards.csv", index=False)
    merchants.to_csv(f"{output_dir}/merchants.csv", index=False)
    devices.to_csv(f"{output_dir}/devices.csv", index=False)
    transactions.to_csv(f"{output_dir}/transactions.csv", index=False)

    stats = {
        "n_customers": len(customers),
        "n_cards": len(cards),
        "n_merchants": len(merchants),
        "n_devices": len(devices),
        "n_transactions": len(transactions),
        "fraud_rate": float(transactions["is_fraud"].mean()),
        "date_range": [
            transactions["timestamp"].min(),
            transactions["timestamp"].max()
        ]
    }
    with open(f"{output_dir}/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[DataGen] ✓ All files saved to {output_dir}/")
    print(f"[DataGen] Stats: {json.dumps(stats, indent=2)}")
    return transactions, cards, merchants, devices, customers


if __name__ == "__main__":
    generate_synthetic_dataset()
