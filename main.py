"""
Main Pipeline Orchestrator — Python 3.13 / macOS safe.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.generate_data import generate_synthetic_dataset
from data.raw.merge_datasets import merge_datasets
from explainability.graph_rag import FraudExplainer
from graph.graph_builder import FraudGraphBuilder, build_temporal_sequences
from models.fusion_model import FocalLoss, build_model
from streaming.pipeline import StreamingFraudPipeline, simulate_streaming
from training.trainer import PlattCalibrator, TransactionDataset, evaluate, train, fraud_kpis, fraud_composite_score

CONFIG = {
    "n_transactions": 50000,
    "seq_len": 10,
    "test_size": 0.15,
    "val_size": 0.15,
    "gnn_variant": "graphsage",
    "gnn_out_dim": 64,
    "transformer_out_dim": 64,
    "hidden_dim": 256,
    "dropout": 0.3,
    "use_anomaly_heads": True,
    "epochs": 60,  # complex multi-modal model needs more iterations on imbalanced data
    "batch_size": 256,
    "lr": 5e-4,  # slightly lower LR; 1e-3 was causing unstable gradients at epoch 1
    "weight_decay": 5e-4,
    "patience": 15,  # allow model to plateau briefly without stopping
    "lr_patience": 5,  # give scheduler more room before halving LR
    "scheduler": "reduce_on_plateau",
    "imbalance_mode": "weighted_sampler",
    "pos_weight": 1.0,  # alpha now carries the imbalance penalty; pos_weight redundant
    "focal_alpha": 0.75,  # 0.75 up-weights fraud (minority) gradient; 0.25 was suppressing it
    "focal_gamma": 2.0,  # standard gamma; reduce from 2.5 to avoid over-focusing on hard examples
    "gnn_max_neighbors": 30,  # cap neighbors per node in GNN aggregation; prevents over-smoothing on dense graphs
    "use_calibration": True,
    "threshold_objective": "f1",
    "threshold": 0.5,
    "stream_n_transactions": 200,
    "stream_delay_ms": 5,
    "rag_sample_size": 3000,
    "data_dir": "data/raw",
    "checkpoint_dir": "models/checkpoints",
    "output_dir": "outputs",
}


def _artifact_paths(config):
    ckpt = Path(config["checkpoint_dir"])
    return {
        "builder": ckpt / "graph_builder.pkl",
        "graph": ckpt / "graph.pkl",
        "sequences": ckpt / "sequences.pkl",
        "explainer": ckpt / "explainer.pkl",
        "model": ckpt / "best_model.pt",
        "threshold": ckpt / "best_threshold.json",
        "train_config": ckpt / "training_config.json",
    }


def _explainer_embedding_dim(config):
    return int(config["hidden_dim"]) // 2


def _persist_artifacts(config, builder, graph, sequences, explainer):
    paths = _artifact_paths(config)
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    builder.save(str(paths["builder"]))
    with open(paths["graph"], "wb") as f:
        payload = {
            "graph": graph,
            "merchant_map": {m: i for i, m in enumerate(builder.merchant_enc.classes_)},
            "device_map": {d: i for i, d in enumerate(builder.device_enc.classes_)},
        }
        pickle.dump(payload, f)
    with open(paths["sequences"], "wb") as f:
        pickle.dump(sequences, f)
    explainer.save(str(paths["explainer"]))
    with open(paths["train_config"], "w") as f:
        json.dump(config, f, indent=2)


def _load_runtime_for_inference(config):
    paths = _artifact_paths(config)
    if not paths["train_config"].exists():
        raise FileNotFoundError(f"Missing artifact: {paths['train_config']}")
    saved_config = json.loads(paths["train_config"].read_text())
    merged_config = {**CONFIG, **saved_config, **config}
    paths = _artifact_paths(merged_config)

    builder = FraudGraphBuilder.load(str(paths["builder"]))
    with open(paths["graph"], "rb") as f:
        graph_payload = pickle.load(f)
    with open(paths["sequences"], "rb") as f:
        sequences = pickle.load(f)

    graph = graph_payload["graph"]
    merchant_map = graph_payload.get("merchant_map", {})
    device_map = graph_payload.get("device_map", {})

    model = build_model(merged_config)
    state = torch.load(paths["model"], map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state)
    model.eval()

    explainer = (
        FraudExplainer.load(str(paths["explainer"]))
        if paths["explainer"].exists()
        else FraudExplainer(embedding_dim=_explainer_embedding_dim(merged_config))
    )
    return merged_config, model, builder, graph, sequences, explainer, merchant_map, device_map


def load_or_generate_data(config):
    txn_path = Path(config["data_dir"]) / "transactions.csv"
    if txn_path.exists():
        print("[Main] Loading existing data...")
        txn_df = pd.read_csv(f"{config['data_dir']}/transactions.csv")
        cards_df = pd.read_csv(f"{config['data_dir']}/cards.csv")
        merchants_df = pd.read_csv(f"{config['data_dir']}/merchants.csv")
        devices_df = pd.read_csv(f"{config['data_dir']}/devices.csv")
        customers_df = pd.read_csv(f"{config['data_dir']}/customers.csv")
    else:
        txn_df, cards_df, merchants_df, devices_df, customers_df = generate_synthetic_dataset(
            output_dir=config["data_dir"]
        )
    print(f"[Main] {len(txn_df)} transactions | fraud={txn_df['is_fraud'].mean():.2%}")
    return txn_df, cards_df, merchants_df, devices_df, customers_df


def time_split(txn_clean, config):
    txn = txn_clean.sort_values("timestamp").reset_index(drop=True)
    n = len(txn)
    t = int(n * config["test_size"])
    v = int(n * config["val_size"])
    train_df = txn.iloc[: n - v - t].reset_index(drop=True)
    val_df = txn.iloc[n - v - t : n - t].reset_index(drop=True)
    test_df = txn.iloc[n - t :].reset_index(drop=True)
    print(f"[Main] Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    return train_df, val_df, test_df


def populate_rag(model, graph, sequences, txn_df, card_enc, explainer, config):
    print("\n[Main] Populating Graph RAG memory...")
    device = next(model.parameters()).device
    x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}

    txn_tmp = txn_df.copy()
    txn_tmp["timestamp"] = pd.to_datetime(txn_tmp["timestamp"])
    txn_tmp = txn_tmp.sort_values(["card_id", "timestamp"]).reset_index(drop=True)
    velocity_map = {}
    for card_id, group in txn_tmp.groupby("card_id"):
        seen_merchants = set()
        hist_times = []
        hist_amounts = []
        for _, row in group.iterrows():
            txn_id = row.get("transaction_id")
            ts = row["timestamp"]
            amt = float(row["amount"])
            if hist_amounts:
                rolling_mean = float(np.mean(hist_amounts[-5:]))
            else:
                rolling_mean = amt
            amount_delta = np.log1p(abs(amt - rolling_mean))
            if hist_times:
                secs = min((ts - hist_times[-1]).total_seconds(), 604800)
            else:
                secs = 604800
            secs_feat = np.log1p(secs)
            merchant_id = row.get("merchant_id")
            is_new_merchant = 1.0 if merchant_id not in seen_merchants else 0.0
            burst_count = sum((ts - t).total_seconds() <= 1800 for t in hist_times)
            burst_count_norm = min(burst_count / 10.0, 1.0)
            velocity_map[txn_id] = (amount_delta, secs_feat, is_new_merchant, burst_count_norm)
            hist_times.append(ts)
            hist_amounts.append(amt)
            seen_merchants.add(merchant_id)

    known = set(card_enc.classes_)
    sample = txn_df[txn_df["card_id"].isin(known)].sample(
        min(config.get("rag_sample_size", 3000), len(txn_df)),
        random_state=42,
    ).reset_index(drop=True)
    bs = 256
    all_embs, all_labels, all_meta = [], [], []
    _warned_emb_health = False  # warn at most once across batches

    model.eval()
    with torch.no_grad():
        for i in range(0, len(sample), bs):
            batch = sample.iloc[i : i + bs]
            cidx = torch.tensor(card_enc.transform(batch["card_id"]), dtype=torch.long, device=device)
            ts = pd.to_datetime(batch["timestamp"])
            amount_vals = batch["amount"].values.astype(float)
            txn_ids = batch["transaction_id"].tolist()
            amount_delta = np.array([velocity_map.get(txn_id, (0.0, np.log1p(604800.0), 0.0, 0.0))[0] for txn_id in txn_ids], dtype=float)
            secs_feat = np.array([velocity_map.get(txn_id, (0.0, np.log1p(604800.0), 0.0, 0.0))[1] for txn_id in txn_ids], dtype=float)
            is_new_merchant = np.array([velocity_map.get(txn_id, (0.0, np.log1p(604800.0), 0.0, 0.0))[2] for txn_id in txn_ids], dtype=float)
            burst_count_norm = np.array([velocity_map.get(txn_id, (0.0, np.log1p(604800.0), 0.0, 0.0))[3] for txn_id in txn_ids], dtype=float)
            raw = torch.tensor(
                np.stack(
                    [
                        np.log1p(amount_vals),
                        batch["is_international"].values.astype(float),
                        np.sin(2 * np.pi * ts.dt.hour / 24),
                        np.cos(2 * np.pi * ts.dt.hour / 24),
                        np.sin(2 * np.pi * ts.dt.dayofweek / 7),
                        np.cos(2 * np.pi * ts.dt.dayofweek / 7),
                        amount_delta,
                        secs_feat,
                        is_new_merchant,
                        burst_count_norm,
                    ],
                    axis=1,
                ).astype(np.float32),
                device=device,
            )
            default_seq = np.zeros((config["seq_len"], 10), dtype=np.float32)
            batch_seqs = [sequences.get(c, default_seq) for c in batch["card_id"]]
            seqs = torch.tensor(np.stack(batch_seqs, axis=0).astype(np.float32), device=device)
            _, embs, _ = model(x_dict, ei_dict, cidx, seqs, raw)
            fused = embs["fused_emb"]

            # Embedding health check — warn once if NaN/inf/near-zero detected
            if not _warned_emb_health:
                emb_np = fused.cpu().numpy()
                n_nan  = int(np.isnan(emb_np).any(axis=1).sum())
                n_inf  = int(np.isinf(emb_np).any(axis=1).sum())
                norms  = np.linalg.norm(emb_np, axis=1)
                n_zero = int((norms < 1e-6).sum())
                if n_nan or n_inf or n_zero:
                    logging.warning(
                        "[populate_rag] Fused embedding health: %d NaN, %d inf, "
                        "%d near-zero out of %d — potential training instability. "
                        "Embeddings will be sanitized by FraudExplainer.",
                        n_nan, n_inf, n_zero, len(emb_np),
                    )
                    _warned_emb_health = True

            all_embs.append(fused)
            all_labels.append(torch.tensor(batch["is_fraud"].values, dtype=torch.float))
            all_meta.extend(batch.to_dict("records"))

    explainer.add_to_memory(torch.cat(all_embs), all_meta, torch.cat(all_labels))
    print(f"[Main] RAG memory: {len(all_meta)} examples")


def run_test_eval(model, graph, sequences, test_df, card_enc, config):
    from torch.utils.data import DataLoader

    print("\n[Main] Test evaluation...")
    device = next(model.parameters()).device
    test_ds = TransactionDataset(test_df, sequences, card_enc, config["seq_len"], config.get("per_txn_sequences"))
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    threshold = float(config.get("threshold", 0.5))
    calibrator = None
    thr_path = Path(config["checkpoint_dir"]) / "best_threshold.json"
    if thr_path.exists():
        j = json.loads(thr_path.read_text())
        threshold = float(j.get("threshold", threshold))
        cal = j.get("calibration")
        if cal:
            calibrator = PlattCalibrator(a=cal.get("a", 1.0), b=cal.get("b", 0.0))

    metrics, y_true, y_prob = evaluate(model, test_loader, graph, FocalLoss(), device, threshold=threshold, calibrator=calibrator)
    f1 = metrics.get("f1", 0.0)
    roc_auc = metrics.get("roc_auc", 0.0)
    pr_auc = metrics.get("pr_auc", 0.0)
    composite = fraud_composite_score(pr_auc, roc_auc, f1)
    metrics["fraud_composite_score"] = composite

    # Fraud-specific KPIs (defaults match fraud_kpis() signature)
    _top_k_frac = 0.01
    _fpr_target = 0.05
    _prec_target = 0.50
    kpis = fraud_kpis(y_true, y_prob,
                      top_k_frac=_top_k_frac,
                      fpr_target=_fpr_target,
                      precision_target=_prec_target)
    metrics.update(kpis)

    print("\n" + "=" * 62)
    print(" FINAL TEST METRICS REPORT")
    print("=" * 62)
    print(f" accuracy     : {metrics.get('accuracy', 0.0):.4f}")
    print(f" precision    : {metrics.get('precision', 0.0):.4f}")
    print(f" recall       : {metrics.get('recall', 0.0):.4f}")
    print(f" f1           : {f1:.4f}")
    print(f" roc_auc      : {roc_auc:.4f}")
    print(f" pr_auc       : {pr_auc:.4f}")
    print(f" fpr          : {metrics.get('fpr', 0.0):.4f}")
    print(f" threshold    : {metrics.get('threshold', threshold):.4f}")
    print(" confusion_matrix:")
    print("              Pred 0    Pred 1")
    print(f"   Actual 0    {metrics.get('tn', 0):7d}   {metrics.get('fp', 0):7d}")
    print(f"   Actual 1    {metrics.get('fn', 0):7d}   {metrics.get('tp', 0):7d}")
    print("-" * 62)
    print(" Fraud-Focused KPIs:")
    print(f"   precision@{int(_top_k_frac*100)}%   : {kpis.get('precision_at_k', 0.0):.4f}")
    print(f"   recall@FPR{int(_fpr_target*100)}%  : {kpis.get(f'recall_at_fpr{int(_fpr_target*100)}pct', 0.0):.4f}")
    print(f"   recall@P{int(_prec_target*100)}%   : {kpis.get(f'recall_at_precision{int(_prec_target*100)}pct', 0.0):.4f}")
    print("-" * 62)
    print(f" fraud_composite: {composite:.4f}  (0.5\u00d7PR-AUC + 0.3\u00d7ROC-AUC + 0.2\u00d7F1)")
    print("=" * 62)
    return metrics


def _seed_pipeline_from_sequences(pipeline, sequences, card_id, timestamp):
    if card_id not in sequences:
        return
    base_ts = pd.Timestamp(timestamp)
    seq = sequences[card_id]
    for i, feat in enumerate(seq):
        ts = base_ts - pd.Timedelta(minutes=(len(seq) - i))
        pipeline.feat_store.update(card_id, feat, ts)


def run_train(config, run_stream=True):
    for d in [config["output_dir"], config["checkpoint_dir"], config["data_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    txn_df, cards_df, merchants_df, devices_df, _ = load_or_generate_data(config)
    print("\n[Main] Building graph...")
    builder = FraudGraphBuilder()
    builder.fit(txn_df, cards_df, merchants_df, devices_df)
    graph, txn_clean = builder.build_graph(txn_df, cards_df, merchants_df, devices_df)
    print("[Main] Building sequences...")
    sequences, _, _, per_txn_sequences = build_temporal_sequences(txn_df, seq_len=config["seq_len"])

    train_df, val_df, test_df = time_split(txn_clean, config)
    print("\n[Main] Building model...")
    model = build_model(config)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Main] Parameters: {params:,}")

    dataset_config = {**config, "per_txn_sequences": per_txn_sequences}
    train_ds = TransactionDataset(train_df, sequences, builder.card_enc, config["seq_len"], per_txn_sequences)
    val_ds = TransactionDataset(val_df, sequences, builder.card_enc, config["seq_len"], per_txn_sequences)
    print(f"[Main] Train={len(train_ds)} Val={len(val_ds)}")
    trained_model, _ = train(model, train_ds, val_ds, graph, config, config["checkpoint_dir"])

    test_metrics = run_test_eval(trained_model, graph, sequences, test_df, builder.card_enc, dataset_config)
    explainer = FraudExplainer(embedding_dim=_explainer_embedding_dim(config))
    populate_rag(trained_model, graph, sequences, train_df, builder.card_enc, explainer, config)
    _persist_artifacts(config, builder, graph, sequences, explainer)

    out = config["output_dir"]
    stream_results, stream_stats = [], {}
    if run_stream:
        print("\n[Main] Streaming simulation...")
        pipeline = StreamingFraudPipeline(
            model=trained_model,
            graph=graph,
            explainer=explainer,
            card_encoder=builder.card_enc,
            seq_len=config["seq_len"],
            threshold=config["threshold"],
            checkpoint_dir=config["checkpoint_dir"],
            merchant_map={m: i for i, m in enumerate(builder.merchant_enc.classes_)},
            device_map={d: i for i, d in enumerate(builder.device_enc.classes_)},
        )
        stream_results, stream_stats = simulate_streaming(
            pipeline,
            txn_clean,
            n=config["stream_n_transactions"],
            delay_ms=config["stream_delay_ms"],
        )
        pd.DataFrame(stream_results).to_csv(f"{out}/stream_results.csv", index=False)
        with open(f"{out}/stream_stats.json", "w") as f:
            json.dump(stream_stats, f, indent=2)

    with open(f"{out}/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(f"{out}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n[Main] ✓ Outputs saved to {out}/")
    return test_metrics, stream_stats, stream_results


def run_predict(config, transaction, update_graph=False):
    cfg, model, builder, graph, sequences, explainer, merchant_map, device_map = _load_runtime_for_inference(config)
    pipeline = StreamingFraudPipeline(
        model=model,
        graph=graph,
        explainer=explainer,
        card_encoder=builder.card_enc,
        seq_len=cfg["seq_len"],
        threshold=cfg.get("threshold", 0.5),
        checkpoint_dir=cfg["checkpoint_dir"],
        merchant_map=merchant_map,
        device_map=device_map,
    )
    _seed_pipeline_from_sequences(
        pipeline,
        sequences,
        transaction.get("card_id", ""),
        transaction.get("timestamp", pd.Timestamp.now("UTC").isoformat()),
    )
    result = pipeline.score_transaction(transaction, update_graph=update_graph)

    if update_graph:
        paths = _artifact_paths(cfg)
        with open(paths["graph"], "wb") as f:
            pickle.dump({"graph": pipeline.graph, "merchant_map": merchant_map, "device_map": device_map}, f)
    print(json.dumps(result, indent=2, default=str))
    return result


def run_update_graph(config, transaction):
    cfg, model, builder, graph, _, explainer, merchant_map, device_map = _load_runtime_for_inference(config)
    pipeline = StreamingFraudPipeline(
        model=model,
        graph=graph,
        explainer=explainer,
        card_encoder=builder.card_enc,
        seq_len=cfg["seq_len"],
        threshold=cfg.get("threshold", 0.5),
        checkpoint_dir=cfg["checkpoint_dir"],
        merchant_map=merchant_map,
        device_map=device_map,
    )
    pipeline.update_graph(transaction)
    paths = _artifact_paths(cfg)
    with open(paths["graph"], "wb") as f:
        pickle.dump({"graph": pipeline.graph, "merchant_map": merchant_map, "device_map": device_map}, f)
    print("[Main] Graph updated and persisted.")


def run_merge(args):
    merged = merge_datasets(args.mockaroo, args.generated, args.output)
    print(f"[Main] Merged dataset rows={len(merged)} cols={len(merged.columns)}")
    print(f"[Main] Saved to {args.output}")


def _parse_transaction_args(args):
    if args.transaction_json:
        try:
            return json.loads(args.transaction_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid --transaction-json payload: {e}") from e
    if args.transaction_file:
        try:
            return json.loads(Path(args.transaction_file).read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --transaction-file '{args.transaction_file}': {e}") from e
    fields_with_invalid_values = [name for name, v in {
        "card-id": args.card_id,
        "merchant-id": args.merchant_id,
        "device-id": args.device_id,
    }.items() if v is None or not isinstance(v, str) or not v.strip()]
    if not fields_with_invalid_values:
        return {
            "transaction_id": args.transaction_id or "manual_txn",
            "card_id": args.card_id.strip(),
            "merchant_id": args.merchant_id.strip(),
            "device_id": args.device_id.strip(),
            "amount": float(args.amount or 0),
            "timestamp": args.timestamp or pd.Timestamp.now("UTC").isoformat(),
            "merchant_category": args.merchant_category or "unknown",
            "channel": args.channel or "pos",
            "is_international": int(args.is_international or 0),
        }
    raise ValueError(
        "Provide --transaction-json, --transaction-file, or non-empty "
        f"--card-id/--merchant-id/--device-id fields. Invalid: {fields_with_invalid_values}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Credit card fraud detection pipeline CLI.")
    parser.add_argument("--data-dir", default=CONFIG["data_dir"])
    parser.add_argument("--checkpoint-dir", default=CONFIG["checkpoint_dir"])
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])

    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Train/evaluate and persist all artifacts.")
    train_p.add_argument("--no-stream", action="store_true", help="Skip streaming simulation.")

    pred_p = sub.add_parser("predict", help="Predict one transaction without retraining.")
    pred_p.add_argument("--transaction-json")
    pred_p.add_argument("--transaction-file")
    pred_p.add_argument("--transaction-id")
    pred_p.add_argument("--card-id")
    pred_p.add_argument("--merchant-id")
    pred_p.add_argument("--device-id")
    pred_p.add_argument("--amount", type=float)
    pred_p.add_argument("--timestamp")
    pred_p.add_argument("--merchant-category")
    pred_p.add_argument("--channel")
    pred_p.add_argument("--is-international", type=int)
    pred_p.add_argument("--update-graph", action="store_true", help="Append transaction links to graph and persist.")

    upd_p = sub.add_parser("update-graph", help="Update graph only from one transaction and persist.")
    upd_p.add_argument("--transaction-json")
    upd_p.add_argument("--transaction-file")
    upd_p.add_argument("--transaction-id")
    upd_p.add_argument("--card-id")
    upd_p.add_argument("--merchant-id")
    upd_p.add_argument("--device-id")
    upd_p.add_argument("--amount", type=float)
    upd_p.add_argument("--timestamp")
    upd_p.add_argument("--merchant-category")
    upd_p.add_argument("--channel")
    upd_p.add_argument("--is-international", type=int)

    merge_p = sub.add_parser("merge", help="Merge generated and Mockaroo CSVs.")
    merge_p.add_argument("--generated", default="data/raw/transactions.csv")
    merge_p.add_argument(
        "--mockaroo",
        nargs="+",
        default=[
            "data/raw/MOCK_DATA (1).csv",
            "data/raw/MOCK_DATA (2).csv",
            "data/raw/MOCK_DATA (3).csv",
            "data/raw/MOCK_DATA (4).csv",
            "data/raw/MOCK_DATA (5).csv",
        ],
    )
    merge_p.add_argument("--output", default="data/raw/final_merged_dataset.csv")
    return parser.parse_args()


def main():
    print("=" * 60)
    print("  Credit Card Fraud Detection — GNN + Transformer + RAG")
    print("=" * 60)
    args = parse_args()
    config = {**CONFIG, "data_dir": args.data_dir, "checkpoint_dir": args.checkpoint_dir, "output_dir": args.output_dir}
    os.makedirs(config["data_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)

    cmd = args.command or "train"
    if cmd == "train":
        run_train(config, run_stream=not getattr(args, "no_stream", False))
    elif cmd == "predict":
        txn = _parse_transaction_args(args)
        run_predict(config, txn, update_graph=getattr(args, "update_graph", False))
    elif cmd == "update-graph":
        txn = _parse_transaction_args(args)
        run_update_graph(config, txn)
    elif cmd == "merge":
        run_merge(args)
    else:
        raise ValueError(f"Unknown command: {cmd}")
    print("[Main] Done.")


if __name__ == "__main__":
    main()
    txn_sorted = txn_df.sort_values(["card_id", "timestamp"]).copy()
    txn_sorted["timestamp"] = pd.to_datetime(txn_sorted["timestamp"])
    txn_sorted["avg_amount_5"] = (
        txn_sorted.groupby("card_id")["amount"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    txn_sorted["prev_timestamp"] = txn_sorted.groupby("card_id")["timestamp"].shift(1)
    txn_sorted["secs_since_last"] = (
        (txn_sorted["timestamp"] - txn_sorted["prev_timestamp"]).dt.total_seconds().clip(upper=604800)
    )
    txn_sorted["secs_since_last"] = txn_sorted["secs_since_last"].fillna(604800.0)
    txn_sorted["merchant_seen_before"] = txn_sorted.groupby("card_id")["merchant_id"].cumcount() > 0
    txn_sorted["is_new_merchant"] = (~txn_sorted["merchant_seen_before"]).astype(float)
    txn_sorted["burst_count_norm"] = txn_sorted.groupby("card_id")["timestamp"].transform(
        lambda s: [min(sum((t - s).dt.total_seconds().abs() <= 1800) / 10.0, 1.0) for t in s]
    )
    txn_features = txn_sorted.set_index("transaction_id")
