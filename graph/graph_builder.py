"""
Graph Construction Module — crash-safe version.
Uses a lightweight SimpleGraph wrapper instead of PyG HeteroData
to avoid any C++ extension calls during graph construction.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import pickle


# ─────────────────────────────────────────
# PLACEHOLDER: Custom graph loaders
# ─────────────────────────────────────────
# If you have a pre-built graph (e.g., from Neo4j or DGL), load it here
# and wrap it into SimpleGraph (see class below).
# ─────────────────────────────────────────


class SimpleGraph:
    """
    Lightweight graph container replacing PyG HeteroData.
    Avoids HeteroData's C++ codepath that crashes on macOS + Py3.13.
    Stores node feature tensors and edge_index tensors as plain dicts.
    """
    def __init__(self):
        self.x_dict          = {}   # node_type -> (N, feat_dim) tensor
        self.edge_index_dict = {}   # (src, rel, dst) -> (2, E) tensor

    def __repr__(self):
        nodes = {k: v.shape for k, v in self.x_dict.items()}
        edges = {str(k): v.shape for k, v in self.edge_index_dict.items()}
        return f"SimpleGraph(nodes={nodes}, edges={edges})"


class FraudGraphBuilder:
    def __init__(self):
        self.card_enc     = LabelEncoder()
        self.merchant_enc = LabelEncoder()
        self.device_enc   = LabelEncoder()
        self.category_enc = LabelEncoder()
        self.channel_enc  = LabelEncoder()
        self.scaler       = StandardScaler()
        self._fitted      = False

    def fit(self, transactions_df, cards_df, merchants_df, devices_df):
        self.card_enc.fit(cards_df["card_id"])
        self.merchant_enc.fit(merchants_df["merchant_id"])
        self.device_enc.fit(devices_df["device_id"])
        self.category_enc.fit(merchants_df["category"])
        self.channel_enc.fit(transactions_df["channel"].dropna())
        txn_feats = self._extract_txn_features(transactions_df)
        self.scaler.fit(txn_feats)
        self._fitted = True
        print(f"[GraphBuilder] Fitted on {len(transactions_df)} transactions")

    def _extract_txn_features(self, df):
        ts   = pd.to_datetime(df["timestamp"])
        feat = pd.DataFrame({
            "amount_log":       np.log1p(df["amount"]),
            "is_international": df["is_international"].astype(float),
            "hour_sin":         np.sin(2 * np.pi * ts.dt.hour / 24),
            "hour_cos":         np.cos(2 * np.pi * ts.dt.hour / 24),
            "dow_sin":          np.sin(2 * np.pi * ts.dt.dayofweek / 7),
            "dow_cos":          np.cos(2 * np.pi * ts.dt.dayofweek / 7),
        })
        return feat.fillna(0)

    def build_graph(self, transactions_df, cards_df, merchants_df, devices_df):
        assert self._fitted, "Call fit() first"

        graph = SimpleGraph()

        # ── Card nodes ──
        card_ids    = self.card_enc.classes_
        card_map    = {c: i for i, c in enumerate(card_ids)}
        cards_idx   = cards_df.set_index("card_id").reindex(card_ids).fillna(0)
        card_feats  = np.stack([
            cards_idx["credit_limit"].values / 20000.0,
            cards_idx["avg_spend"].values / 5000.0,
            cards_idx["fraud_history"].values.astype(float),
        ], axis=1)
        graph.x_dict["card"] = torch.tensor(card_feats, dtype=torch.float32)

        # ── Merchant nodes ──
        merch_ids   = self.merchant_enc.classes_
        merch_map   = {m: i for i, m in enumerate(merch_ids)}
        merch_idx   = merchants_df.set_index("merchant_id").reindex(merch_ids).fillna(0)
        cat_enc     = self.category_enc.transform(
                          merch_idx["category"].fillna("grocery")) / max(len(self.category_enc.classes_)-1, 1)
        merch_feats = np.stack([
            merch_idx["risk_score"].values.astype(float),
            cat_enc,
            merch_idx["avg_transaction_amount"].values / 500.0,
        ], axis=1)
        graph.x_dict["merchant"] = torch.tensor(merch_feats, dtype=torch.float32)

        # ── Device nodes ──
        dev_ids    = self.device_enc.classes_
        dev_map    = {d: i for i, d in enumerate(dev_ids)}
        devs_idx   = devices_df.set_index("device_id").reindex(dev_ids).fillna(0)
        dev_feats  = np.stack([
            devs_idx["reuse_count"].values / 20.0,
            devs_idx["known_fraudulent"].values.astype(float),
        ], axis=1)
        graph.x_dict["device"] = torch.tensor(dev_feats, dtype=torch.float32)

        # ── Edges ──
        txn   = transactions_df.copy()
        valid = (txn["card_id"].isin(card_map) &
                 txn["merchant_id"].isin(merch_map) &
                 txn["device_id"].isin(dev_map))
        txn   = txn[valid].reset_index(drop=True)

        c_idx = torch.tensor([card_map[c]    for c in txn["card_id"]],     dtype=torch.long)
        m_idx = torch.tensor([merch_map[m]   for m in txn["merchant_id"]], dtype=torch.long)
        d_idx = torch.tensor([dev_map[d]     for d in txn["device_id"]],   dtype=torch.long)

        graph.edge_index_dict[("card", "pays",     "merchant")] = torch.stack([c_idx, m_idx])
        graph.edge_index_dict[("card", "uses",     "device")]   = torch.stack([c_idx, d_idx])
        graph.edge_index_dict[("device","seen_at", "merchant")] = torch.stack([d_idx, m_idx])

        # Store labels on the transaction df for training
        txn_feats_scaled = self.scaler.transform(self._extract_txn_features(txn))
        txn["_edge_feat_0"] = txn_feats_scaled[:, 0]  # amount_log (scaled)

        print(f"[GraphBuilder] {graph}")
        return graph, txn

    def save(self, path="models/graph_builder.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path="models/graph_builder.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


def build_temporal_sequences(transactions_df, seq_len=10):
    txn = transactions_df.copy()
    txn["timestamp"] = pd.to_datetime(txn["timestamp"])
    txn = txn.sort_values(["card_id", "timestamp"])

    chan_enc = LabelEncoder().fit(txn["channel"])
    cat_enc  = LabelEncoder().fit(txn["merchant_category"])

    sequences = {}
    per_txn_sequences = {}
    for card_id, group in txn.groupby("card_id"):
        feats = []
        for _, row in group.iterrows():
            ts = row["timestamp"]
            current_feat = [
                np.log1p(row["amount"]),
                float(row["is_international"]),
                np.sin(2 * np.pi * ts.hour / 24),
                np.cos(2 * np.pi * ts.hour / 24),
                float(chan_enc.transform([row["channel"]])[0]) / max(len(chan_enc.classes_)-1,1),
                float(cat_enc.transform([row["merchant_category"]])[0]) / max(len(cat_enc.classes_)-1,1),
            ]
            feats.append(current_feat)
            txn_seq = feats[-seq_len:]
            if len(txn_seq) < seq_len:
                txn_seq = [[0.0] * 6] * (seq_len - len(txn_seq)) + txn_seq
            per_txn_sequences[row["transaction_id"]] = txn_seq
        if len(feats) < seq_len:
            feats = [[0.0] * 6] * (seq_len - len(feats)) + feats
        sequences[card_id] = feats[-seq_len:]

    print(f"[GraphBuilder] Sequences built for {len(sequences)} cards")
    return sequences, chan_enc, cat_enc, per_txn_sequences


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from data.generate_data import generate_synthetic_dataset
    txn_df, cards_df, merchants_df, devices_df, _ = generate_synthetic_dataset()
    builder = FraudGraphBuilder()
    builder.fit(txn_df, cards_df, merchants_df, devices_df)
    graph, txn_clean = builder.build_graph(txn_df, cards_df, merchants_df, devices_df)
    print("Graph:", graph)
