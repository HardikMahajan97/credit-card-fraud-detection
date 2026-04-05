"""
Lightweight smoke tests for the fraud detection pipeline.

Run with:  python -m pytest tests/ -v
       or: python tests/test_smoke.py
"""

import sys
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _small_dfs(n_txn: int = 200):
    """Return tiny (transactions, cards, merchants, devices, customers) DataFrames."""
    import pandas as pd
    from datetime import datetime, timedelta

    np.random.seed(0)
    n_cards = 20
    n_merch = 10
    n_dev = 30

    cards = pd.DataFrame({
        "card_id":       [f"CARD_{i:03d}" for i in range(n_cards)],
        "credit_limit":  np.random.choice([1000, 5000, 10000], n_cards),
        "avg_spend":     np.random.uniform(100, 2000, n_cards),
        "fraud_history": np.random.randint(0, 2, n_cards),
    })
    merchants = pd.DataFrame({
        "merchant_id":              [f"M_{i:03d}" for i in range(n_merch)],
        "category":                 np.random.choice(["grocery", "online_retail", "travel"], n_merch),
        "risk_score":               np.random.uniform(0, 1, n_merch),
        "avg_transaction_amount":   np.random.uniform(20, 300, n_merch),
    })
    devices = pd.DataFrame({
        "device_id":        [f"DEV_{i:03d}" for i in range(n_dev)],
        "reuse_count":      np.random.randint(0, 15, n_dev),
        "known_fraudulent": np.random.randint(0, 2, n_dev),
    })
    base_ts = datetime(2024, 1, 1)
    txn = pd.DataFrame({
        "transaction_id":    [f"TXN_{i:05d}" for i in range(n_txn)],
        "card_id":           np.random.choice(cards["card_id"], n_txn),
        "merchant_id":       np.random.choice(merchants["merchant_id"], n_txn),
        "device_id":         np.random.choice(devices["device_id"], n_txn),
        "amount":            np.random.uniform(5, 500, n_txn),
        "timestamp":         [(base_ts + timedelta(minutes=i * 5)).isoformat() for i in range(n_txn)],
        "merchant_category": np.random.choice(["grocery", "online_retail", "travel"], n_txn),
        "channel":           np.random.choice(["online", "in_store", "atm"], n_txn),
        "is_international":  np.random.randint(0, 2, n_txn),
        "is_fraud":          np.random.choice([0, 1], n_txn, p=[0.95, 0.05]),
        "customer_id":       "CUST_000",
        "transaction_status": "approved",
        "fraud_reasons":     "none",
    })
    customers = pd.DataFrame({"customer_id": ["CUST_000"], "age": [35]})
    return txn, cards, merchants, devices, customers


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphBuilder:
    def test_fit_build(self):
        from graph.graph_builder import FraudGraphBuilder, SimpleGraph

        txn, cards, merchants, devices, _ = _small_dfs()
        builder = FraudGraphBuilder()
        builder.fit(txn, cards, merchants, devices)
        graph, txn_clean = builder.build_graph(txn, cards, merchants, devices)
        assert isinstance(graph, SimpleGraph)
        assert "card" in graph.x_dict
        assert "merchant" in graph.x_dict
        assert "device" in graph.x_dict
        assert ("card", "pays", "merchant") in graph.edge_index_dict
        assert len(txn_clean) <= len(txn)

    def test_save_load(self, tmp_path):
        from graph.graph_builder import FraudGraphBuilder

        txn, cards, merchants, devices, _ = _small_dfs()
        builder = FraudGraphBuilder()
        builder.fit(txn, cards, merchants, devices)
        path = str(tmp_path / "builder.pkl")
        builder.save(path)
        loaded = FraudGraphBuilder.load(path)
        assert loaded._fitted
        assert list(loaded.card_enc.classes_) == list(builder.card_enc.classes_)

    def test_temporal_sequences(self):
        from graph.graph_builder import build_temporal_sequences

        txn, *_ = _small_dfs(100)
        seqs, chan_enc, cat_enc = build_temporal_sequences(txn, seq_len=5)
        assert len(seqs) > 0
        for card_id, seq in seqs.items():
            assert len(seq) == 5
            assert len(seq[0]) == 6


class TestModel:
    def test_forward(self):
        import torch
        from graph.graph_builder import FraudGraphBuilder, build_temporal_sequences
        from models.fusion_model import build_model

        txn, cards, merchants, devices, _ = _small_dfs(100)
        builder = FraudGraphBuilder()
        builder.fit(txn, cards, merchants, devices)
        graph, txn_clean = builder.build_graph(txn, cards, merchants, devices)
        seqs, _, _ = build_temporal_sequences(txn, seq_len=5)

        config = {
            "gnn_variant": "graphsage", "gnn_out_dim": 16,
            "transformer_out_dim": 16, "hidden_dim": 32,
            "dropout": 0.0, "seq_len": 5, "use_anomaly_heads": True,
        }
        model = build_model(config)
        model.eval()

        B = 4
        card_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        seq = torch.zeros(B, 5, 6)
        raw = torch.zeros(B, 6)
        x_dict = graph.x_dict
        ei_dict = graph.edge_index_dict

        with torch.no_grad():
            probs, embs, scores = model(x_dict, ei_dict, card_idx, seq, raw)
        assert probs.shape == (B,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_build_model_config(self):
        from models.fusion_model import build_model
        config = {"gnn_variant": "graphsage", "gnn_out_dim": 32,
                  "transformer_out_dim": 32, "hidden_dim": 64,
                  "dropout": 0.3, "seq_len": 10, "use_anomaly_heads": True}
        model = build_model(config)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params > 0


class TestMetrics:
    def test_compute_metrics_balanced(self):
        from training.trainer import compute_metrics

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_metrics(y_true, y_prob, threshold=0.5)

        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0
        assert m["roc_auc"] == 1.0
        assert m["tp"] == 2
        assert m["tn"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 0

    def test_compute_metrics_keys(self):
        from training.trainer import compute_metrics

        y_true = np.array([0, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.3])
        m = compute_metrics(y_true, y_prob)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                    "fpr", "tp", "fp", "tn", "fn"):
            assert key in m, f"Missing metric key: {key}"

    def test_compute_metrics_all_negative(self):
        """Edge case: no positive labels — roc_auc / pr_auc should be 0, not error."""
        from training.trainer import compute_metrics

        y_true = np.zeros(10, dtype=int)
        y_prob = np.random.rand(10)
        m = compute_metrics(y_true, y_prob)
        assert m["roc_auc"] == 0.0
        assert m["pr_auc"] == 0.0

    def test_find_best_threshold(self):
        from training.trainer import find_best_threshold

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        thr, f1 = find_best_threshold(y_true, y_prob, objective="f1")
        assert 0.0 < thr < 1.0
        assert f1 > 0.0

    def test_platt_calibrator(self):
        from training.trainer import PlattCalibrator

        y_true = np.array([0, 0, 1, 1, 0, 1], dtype=float)
        y_prob = np.array([0.1, 0.2, 0.7, 0.9, 0.3, 0.8])
        cal = PlattCalibrator()
        cal.fit(y_prob, y_true)
        out = cal.transform(y_prob)
        assert out.shape == y_prob.shape
        assert (out >= 0).all() and (out <= 1).all()
        sd = cal.state_dict()
        assert "a" in sd and "b" in sd


class TestFraudExplainer:
    def test_add_explain(self):
        import torch
        from explainability.graph_rag import FraudExplainer

        explainer = FraudExplainer(embedding_dim=8, max_memory=100)
        embs = torch.randn(10, 8)
        labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=torch.float)
        meta = [{"transaction_id": f"T{i}", "amount": float(i * 10)} for i in range(10)]
        explainer.add_to_memory(embs, meta, labels)

        result = explainer.explain(
            transaction={"transaction_id": "TTEST", "amount": 50.0, "card_id": "C1",
                         "merchant_id": "M1", "channel": "online", "is_international": 0},
            fraud_prob=0.9,
            fused_embedding=embs[0].numpy(),
            anomaly_scores={"burst_score": 0.8},
        )
        assert "fraud_probability" in result
        assert "explanation" in result
        assert "risk_level" in result
        assert result["risk_level"] in ("HIGH", "MEDIUM", "LOW")

    def test_save_load(self, tmp_path):
        import torch
        from explainability.graph_rag import FraudExplainer

        explainer = FraudExplainer(embedding_dim=8)
        embs = torch.randn(5, 8)
        labels = torch.ones(5)
        meta = [{"transaction_id": f"T{i}"} for i in range(5)]
        explainer.add_to_memory(embs, meta, labels)

        path = str(tmp_path / "explainer.pkl")
        explainer.save(path)
        loaded = FraudExplainer.load(path)
        assert len(loaded.fraud_embeddings) == 5
        assert loaded.embedding_dim == 8


class TestStreamingPipeline:
    def _make_pipeline(self):
        import torch
        from graph.graph_builder import FraudGraphBuilder, build_temporal_sequences
        from models.fusion_model import build_model
        from explainability.graph_rag import FraudExplainer
        from streaming.pipeline import StreamingFraudPipeline

        txn, cards, merchants, devices, _ = _small_dfs(100)
        builder = FraudGraphBuilder()
        builder.fit(txn, cards, merchants, devices)
        graph, _ = builder.build_graph(txn, cards, merchants, devices)
        seqs, _, _ = build_temporal_sequences(txn, seq_len=5)

        config = {"gnn_variant": "graphsage", "gnn_out_dim": 16,
                  "transformer_out_dim": 16, "hidden_dim": 32,
                  "dropout": 0.0, "seq_len": 5, "use_anomaly_heads": True}
        model = build_model(config)
        explainer = FraudExplainer(embedding_dim=16)
        merchant_map = {m: i for i, m in enumerate(builder.merchant_enc.classes_)}
        device_map = {d: i for i, d in enumerate(builder.device_enc.classes_)}
        pipeline = StreamingFraudPipeline(
            model=model, graph=graph, explainer=explainer,
            card_encoder=builder.card_enc, seq_len=5, threshold=0.5,
            merchant_map=merchant_map, device_map=device_map,
        )
        return pipeline, builder

    def test_score_transaction(self):
        pipeline, builder = self._make_pipeline()
        txn = {
            "transaction_id": "T001",
            "card_id": builder.card_enc.classes_[0],
            "merchant_id": "M_000",
            "device_id": "DEV_000",
            "amount": 50.0,
            "timestamp": "2024-01-01T12:00:00",
            "merchant_category": "grocery",
            "channel": "online",
            "is_international": 0,
        }
        result = pipeline.score_transaction(txn)
        assert "fraud_probability" in result
        assert 0.0 <= result["fraud_probability"] <= 1.0
        assert "risk_level" in result

    def test_oov_card_fallback(self):
        """Unknown card_id must not raise — falls back to index 0."""
        pipeline, _ = self._make_pipeline()
        txn = {
            "transaction_id": "T_OOV",
            "card_id": "CARD_UNKNOWN_XYZ",
            "merchant_id": "M_000",
            "device_id": "DEV_000",
            "amount": 100.0,
            "timestamp": "2024-01-01T10:00:00",
            "is_international": 0,
        }
        result = pipeline.score_transaction(txn)
        assert "fraud_probability" in result

    def test_update_graph(self):
        pipeline, builder = self._make_pipeline()
        before = pipeline.graph.edge_index_dict[("card", "pays", "merchant")].shape[1]
        txn = {
            "card_id": builder.card_enc.classes_[0],
            "merchant_id": "M_000",
            "device_id": "DEV_000",
        }
        pipeline.update_graph(txn)
        after = pipeline.graph.edge_index_dict[("card", "pays", "merchant")].shape[1]
        assert after == before + 1

    def test_update_graph_oov(self):
        """OOV update_graph must not raise."""
        pipeline, _ = self._make_pipeline()
        txn = {"card_id": "UNKNOWN", "merchant_id": "UNKNOWN_MERCH", "device_id": "UNKNOWN_DEV"}
        pipeline.update_graph(txn)  # should not raise

    def test_score_with_update_graph(self):
        pipeline, builder = self._make_pipeline()
        txn = {
            "transaction_id": "T_UPD",
            "card_id": builder.card_enc.classes_[1],
            "merchant_id": "M_000",
            "device_id": "DEV_000",
            "amount": 200.0,
            "timestamp": "2024-01-01T14:00:00",
            "is_international": 1,
        }
        result = pipeline.score_transaction(txn, update_graph=True)
        assert "fraud_probability" in result


class TestMergeDatasets:
    def test_merge(self, tmp_path):
        import pandas as pd
        from data.raw.merge_datasets import merge_datasets

        # Create small mock CSVs
        gen = pd.DataFrame({
            "transaction_id": ["T1", "T2"],
            "card_id": ["C1", "C2"],
            "amount": [10.0, 20.0],
            "is_fraud": [0, 1],
            "merchant_id": ["M1", "M2"],
            "timestamp": ["2024-01-01", "2024-01-02"],
            "merchant_category": ["grocery", "online_retail"],
            "channel": ["online", "in_store"],
            "is_international": [0, 1],
            "device_id": ["D1", "D2"],
            "customer_id": ["CUST1", "CUST2"],
            "transaction_status": ["approved", "approved"],
            "fraud_reasons": ["none", "test"],
        })
        mock = pd.DataFrame({
            "txn_id": ["M1"],
            "card_id": ["C3"],
            "amount": [30.0],
            "is_fraud": [0],
            "merchant_id": ["M3"],
            "timestamp": ["2024-01-03"],
            "merchant_cat": ["travel"],
            "channel": ["atm"],
            "device_id": ["D3"],
        })
        gen_path = str(tmp_path / "generated.csv")
        mock_path = str(tmp_path / "mock.csv")
        out_path = str(tmp_path / "merged.csv")
        gen.to_csv(gen_path, index=False)
        mock.to_csv(mock_path, index=False)

        merged = merge_datasets([mock_path], gen_path, out_path)
        assert len(merged) == 3
        assert Path(out_path).exists()


class TestCLIParsing:
    def test_no_subcommand_defaults_to_train(self):
        """python main.py (no subcommand) must not raise AttributeError."""
        import argparse
        import sys as _sys
        old_argv = _sys.argv
        try:
            _sys.argv = ["main.py"]
            from main import parse_args
            args = parse_args()
            cmd = args.command or "train"
            # getattr must not raise
            no_stream = getattr(args, "no_stream", False)
            assert cmd == "train"
            assert no_stream is False
        finally:
            _sys.argv = old_argv

    def test_train_subcommand(self):
        import sys as _sys
        old_argv = _sys.argv
        try:
            _sys.argv = ["main.py", "train", "--no-stream"]
            from main import parse_args
            args = parse_args()
            assert args.command == "train"
            assert getattr(args, "no_stream", False) is True
        finally:
            _sys.argv = old_argv

    def test_predict_subcommand_fields(self):
        import sys as _sys
        old_argv = _sys.argv
        try:
            _sys.argv = [
                "main.py", "predict",
                "--card-id", "CARD_001",
                "--merchant-id", "M_001",
                "--device-id", "DEV_001",
                "--amount", "99.5",
                "--update-graph",
            ]
            from main import parse_args, _parse_transaction_args
            args = parse_args()
            assert args.command == "predict"
            assert getattr(args, "update_graph", False) is True
            txn = _parse_transaction_args(args)
            assert txn["card_id"] == "CARD_001"
            assert txn["amount"] == 99.5
        finally:
            _sys.argv = old_argv

    def test_merge_subcommand(self):
        import sys as _sys
        old_argv = _sys.argv
        try:
            _sys.argv = ["main.py", "merge", "--output", "data/raw/final_merged_dataset.csv"]
            from main import parse_args
            args = parse_args()
            assert args.command == "merge"
            assert args.output == "data/raw/final_merged_dataset.csv"
        finally:
            _sys.argv = old_argv


class TestArtifactPersistence:
    def test_persist_and_load(self, tmp_path):
        """Full round-trip: persist artifacts then reload for inference."""
        import torch
        from graph.graph_builder import FraudGraphBuilder, build_temporal_sequences
        from models.fusion_model import build_model
        from explainability.graph_rag import FraudExplainer
        import main as _main

        txn, cards, merchants, devices, _ = _small_dfs(100)
        builder = FraudGraphBuilder()
        builder.fit(txn, cards, merchants, devices)
        graph, _ = builder.build_graph(txn, cards, merchants, devices)
        seqs, _, _ = build_temporal_sequences(txn, seq_len=5)

        config = {"gnn_variant": "graphsage", "gnn_out_dim": 16,
                  "transformer_out_dim": 16, "hidden_dim": 32,
                  "dropout": 0.0, "seq_len": 5, "use_anomaly_heads": True,
                  "threshold": 0.5, "checkpoint_dir": str(tmp_path),
                  "output_dir": str(tmp_path)}
        model = build_model(config)
        torch.save(model.state_dict(), str(tmp_path / "best_model.pt"))

        explainer = FraudExplainer(embedding_dim=16)
        _main._persist_artifacts(config, builder, graph, seqs, explainer)

        paths = _main._artifact_paths(config)
        assert paths["builder"].exists()
        assert paths["graph"].exists()
        assert paths["sequences"].exists()
        assert paths["explainer"].exists()
        assert paths["train_config"].exists()

        # Reload
        loaded_builder = FraudGraphBuilder.load(str(paths["builder"]))
        with open(paths["graph"], "rb") as f:
            g_payload = pickle.load(f)
        with open(paths["sequences"], "rb") as f:
            loaded_seqs = pickle.load(f)
        loaded_explainer = FraudExplainer.load(str(paths["explainer"]))

        assert loaded_builder._fitted
        assert "graph" in g_payload
        assert isinstance(loaded_seqs, dict)
        assert loaded_explainer.embedding_dim == 16


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (run without pytest)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    suites = [
        TestGraphBuilder,
        TestModel,
        TestMetrics,
        TestFraudExplainer,
        TestStreamingPipeline,
        TestMergeDatasets,
        TestCLIParsing,
        TestArtifactPersistence,
    ]
    passed = 0
    failed = 0
    for suite_cls in suites:
        suite = suite_cls()
        methods = [m for m in dir(suite_cls) if m.startswith("test_")]
        for method_name in methods:
            label = f"{suite_cls.__name__}.{method_name}"
            try:
                # inject a real tmp_path for tests that need it
                import tempfile, types
                method = getattr(suite, method_name)
                import inspect
                sig = inspect.signature(method)
                if "tmp_path" in sig.parameters:
                    with tempfile.TemporaryDirectory() as d:
                        method(Path(d))
                else:
                    method()
                print(f"  ✅  {label}")
                passed += 1
            except Exception as exc:
                print(f"  ❌  {label}: {exc}")
                traceback.print_exc()
                failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
