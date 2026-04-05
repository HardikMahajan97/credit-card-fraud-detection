"""
Real-Time Streaming Inference Pipeline — macOS / Python 3.13 safe.
"""

import torch, numpy as np, pandas as pd
import json, time, logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("streaming")


def _seq_mask_from_seq(seq: torch.Tensor) -> torch.Tensor:
    # seq: (1, T, F)
    return (seq.abs().sum(dim=-1) == 0)


class RollingFeatureStore:
    def __init__(self, seq_len=10):
        self.seq_len   = seq_len
        self.sequences = {}
        self.timestamps = {}

    def update(self, card_id, features, timestamp):
        ts = pd.Timestamp(timestamp)
        if card_id not in self.sequences:
            self.sequences[card_id]  = deque(maxlen=self.seq_len)
            self.timestamps[card_id] = deque(maxlen=self.seq_len)
        self.sequences[card_id].append(features)
        self.timestamps[card_id].append(ts)

    def get_sequence(self, card_id):
        feat_dim = 10
        if card_id not in self.sequences:
            return np.zeros((self.seq_len, feat_dim), dtype=np.float32)
        seq = list(self.sequences[card_id])
        while len(seq) < self.seq_len:
            seq = [[0.0]*feat_dim] + seq
        return np.array(seq[-self.seq_len:], dtype=np.float32)

    def get_burst_count(self, card_id, window_seconds=1800):
        if card_id not in self.timestamps:
            return 0
        ts_list = list(self.timestamps[card_id])
        if not ts_list:
            return 0
        now = max(ts_list)
        return sum(1 for t in ts_list if (now - t).total_seconds() <= window_seconds)


class StreamingFraudPipeline:
    def __init__(self, model, graph, explainer, card_encoder,
                 seq_len=10, threshold=0.5, device=None, checkpoint_dir: Optional[str] = None,
                 merchant_map: Optional[dict] = None, device_map: Optional[dict] = None):
        self.model        = model.eval()
        self.graph        = graph
        self.explainer    = explainer
        self.card_encoder = card_encoder
        self.seq_len      = seq_len
        self.device       = device or torch.device("cpu")
        self.feat_store   = RollingFeatureStore(seq_len)
        self._x_dict      = {k: v.to(self.device) for k, v in graph.x_dict.items()}
        self._ei_dict     = {k: v.to(self.device) for k, v in graph.edge_index_dict.items()}
        self.total_scored = 0
        self.total_flagged= 0
        self.latencies    = deque(maxlen=1000)
        self.merchant_map = merchant_map or {}
        self.device_map = device_map or {}

        # Load tuned threshold + calibration if present
        self.threshold = float(threshold)
        self.calibration = None
        if checkpoint_dir:
            p = Path(checkpoint_dir) / "best_threshold.json"
            if p.exists():
                j = json.loads(p.read_text())
                self.threshold = float(j.get("threshold", self.threshold))
                self.calibration = j.get("calibration")
        logger.info(f"StreamingFraudPipeline ready (threshold={self.threshold:.3f}, calibrated={self.calibration is not None})")

    def update_graph(self, txn):
        """Append edges for a new transaction to the live graph.

        Out-of-vocabulary (OOV) handling:
          - card_id not seen at training time  → falls back to index 0 (first trained card).
          - merchant_id not in merchant_map    → falls back to index 0.
          - device_id   not in device_map      → falls back to index 0.
        This is safe for edge bookkeeping but the model will use the node-0
        embedding as a proxy for the unknown entity.  For production use,
        reserve a dedicated "unknown" node (index 0) during training so the
        model learns a meaningful fallback representation.
        """
        if not hasattr(self, "_ei_dict"):
            self._ei_dict = {}
        card_id = txn.get("card_id")
        merchant_id = txn.get("merchant_id")
        device_id = txn.get("device_id")
        if card_id is None or merchant_id is None or device_id is None:
            return

        # OOV: unknown card silently maps to index 0 (see docstring above).
        try:
            c_idx = int(self.card_encoder.transform([card_id])[0])
        except Exception:
            c_idx = 0

        # OOV: unknown merchant/device silently maps to index 0.
        m_idx = int(self.merchant_map.get(merchant_id, 0))
        d_idx = int(self.device_map.get(device_id, 0))

        def _append_edge(key, src, dst):
            ei = self.graph.edge_index_dict.get(key)
            new_col = torch.tensor([[src], [dst]], dtype=torch.long)
            if ei is None or ei.numel() == 0:
                self.graph.edge_index_dict[key] = new_col
            else:
                self.graph.edge_index_dict[key] = torch.cat([ei, new_col], dim=1)
            self._ei_dict[key] = self.graph.edge_index_dict[key].to(self.device)

        _append_edge(("card", "pays", "merchant"), c_idx, m_idx)
        _append_edge(("card", "uses", "device"), c_idx, d_idx)
        _append_edge(("device", "seen_at", "merchant"), d_idx, m_idx)

    def _features(self, txn):
        ts = pd.Timestamp(txn.get("timestamp", datetime.now().isoformat()))
        card_id = txn.get("card_id", "")
        amount = float(txn.get("amount", 0))

        recent_amts = [s[0] for s in list(self.feat_store.sequences.get(card_id, []))]
        rolling_mean = float(np.mean([np.expm1(a) for a in recent_amts[-5:]])) if recent_amts else amount
        amount_delta = np.log1p(abs(amount - rolling_mean))

        prev_ts_list = list(self.feat_store.timestamps.get(card_id, []))
        if prev_ts_list:
            secs = min((ts - prev_ts_list[-1]).total_seconds(), 604800)
        else:
            secs = 604800
        secs_feat = np.log1p(secs)

        burst = min(self.feat_store.get_burst_count(card_id) / 10.0, 1.0)
        return np.array([
            np.log1p(amount),
            float(txn.get("is_international", 0)),
            np.sin(2*np.pi*ts.hour/24),
            np.cos(2*np.pi*ts.hour/24),
            np.sin(2*np.pi*ts.dayofweek/7),
            np.cos(2*np.pi*ts.dayofweek/7),
            amount_delta,
            secs_feat,
            0.0,  # is_new_merchant placeholder: not trackable without persistent merchant-history state
            burst,
        ], dtype=np.float32)

    def _apply_calibration(self, prob_val: float) -> float:
        if not self.calibration:
            return prob_val
        a = float(self.calibration.get("a", 1.0))
        b = float(self.calibration.get("b", 0.0))
        p = float(np.clip(prob_val, 1e-6, 1 - 1e-6))
        logit = np.log(p / (1 - p))
        return float(1.0 / (1.0 + np.exp(-(a * logit + b))))

    def _card_idx(self, card_id):
        # OOV: card_id not seen at training time falls back to index 0.
        # See update_graph() docstring for the long-term mitigation strategy.
        try:
            return int(self.card_encoder.transform([card_id])[0])
        except Exception:
            return 0

    @torch.no_grad()
    def score_transaction(self, txn, update_graph: bool = False):
        t0      = time.perf_counter()
        card_id = txn.get("card_id", "")
        if update_graph:
            self.update_graph(txn)
        feats   = self._features(txn)
        self.feat_store.update(card_id, feats.tolist(), txn.get("timestamp",""))
        burst   = self.feat_store.get_burst_count(card_id)
        cidx    = torch.tensor([self._card_idx(card_id)], dtype=torch.long, device=self.device)
        seq     = torch.tensor(self.feat_store.get_sequence(card_id),
                               dtype=torch.float32).unsqueeze(0).to(self.device)
        raw     = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device)
        seq_mask = _seq_mask_from_seq(seq).to(self.device)

        prob, embs, scores = self.model(self._x_dict, self._ei_dict, cidx, seq, raw, seq_mask=seq_mask)
        prob_val = float(prob[0].cpu())
        prob_val = self._apply_calibration(prob_val)
        emb_np   = embs["fused_emb"][0].cpu().numpy()
        sc_val   = {k: float(v[0].cpu()) for k, v in scores.items()}

        if burst >= 5:
            prob_val = min(prob_val + 0.3, 1.0)
            sc_val["burst_score"] = max(sc_val.get("burst_score", 0), 0.85)

        result = self.explainer.explain(
            transaction=txn, fraud_prob=prob_val,
            fused_embedding=emb_np, anomaly_scores=sc_val,
            graph_neighbors={"burst_count": burst}
        )
        latency = (time.perf_counter() - t0) * 1000
        self.latencies.append(latency)
        self.total_scored += 1
        if prob_val >= self.threshold:
            self.total_flagged += 1
        result["latency_ms"] = round(latency, 2)
        return result

    def get_stats(self):
        lat = list(self.latencies)
        return {
            "total_scored":  self.total_scored,
            "total_flagged": self.total_flagged,
            "flag_rate":     self.total_flagged / max(self.total_scored, 1),
            "avg_latency_ms": float(np.mean(lat)) if lat else 0,
            "p95_latency_ms": float(np.percentile(lat, 95)) if lat else 0,
            "p99_latency_ms": float(np.percentile(lat, 99)) if lat else 0,
        }


def simulate_streaming(pipeline, transactions_df, n=100, delay_ms=10):
    logger.info(f"[Stream] Starting: {n} transactions")
    sample  = transactions_df.sample(min(n, len(transactions_df))).to_dict("records")
    results = []

    for i, txn in enumerate(sample):
        r = pipeline.score_transaction(txn)
        results.append(r)
        if r["fraud_prediction"]:
            logger.warning(f"[{i+1}/{n}] 🚨 FRAUD | prob={r['fraud_probability']:.3f} | "
                           f"risk={r['risk_level']} | {r['latency_ms']:.1f}ms")
        elif (i+1) % 25 == 0:
            logger.info(f"[{i+1}/{n}] ✅ OK | prob={r['fraud_probability']:.3f} | {r['latency_ms']:.1f}ms")
        time.sleep(delay_ms / 1000.0)

    stats = pipeline.get_stats()
    logger.info(f"[Stream] Done. Stats: {json.dumps(stats, indent=2)}")
    return results, stats
