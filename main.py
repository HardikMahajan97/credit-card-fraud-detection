"""
Main Pipeline Orchestrator — Python 3.13 / macOS safe.
"""

import sys, os, json, torch, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.generate_data       import generate_synthetic_dataset
from graph.graph_builder      import FraudGraphBuilder, build_temporal_sequences
from models.fusion_model      import FusionFraudDetector, build_model
from training.trainer         import TransactionDataset, train
from explainability.graph_rag import FraudExplainer
from streaming.pipeline       import StreamingFraudPipeline, simulate_streaming

CONFIG = {
    "n_transactions": 50000,
    "seq_len": 10,
    "test_size": 0.15,
    "val_size":  0.15,
    "gnn_variant": "graphsage",
    "gnn_out_dim": 64,
    "transformer_out_dim": 64,
    "hidden_dim": 256,
    "dropout": 0.25,
    "use_anomaly_heads": True,
    "epochs": 15,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 5,

    # Imbalance & decisioning
    "imbalance_mode": "weighted_sampler",
    "pos_weight": 6.0,
    "focal_alpha": 0.5,
    "focal_gamma": 2.0,
    "use_calibration": True,
    "threshold_objective": "f1",
    "threshold": 0.5,

    "stream_n_transactions": 200,
    "stream_delay_ms": 5,
    "data_dir": "data/raw",
    "checkpoint_dir": "models/checkpoints",
    "output_dir": "outputs",
}


def load_or_generate_data(config):
    txn_path = Path(config["data_dir"]) / "transactions.csv"
    if txn_path.exists():
        print("[Main] Loading existing data...")
        txn_df       = pd.read_csv(f"{config['data_dir']}/transactions.csv")
        cards_df     = pd.read_csv(f"{config['data_dir']}/cards.csv")
        merchants_df = pd.read_csv(f"{config['data_dir']}/merchants.csv")
        devices_df   = pd.read_csv(f"{config['data_dir']}/devices.csv")
        customers_df = pd.read_csv(f"{config['data_dir']}/customers.csv")
    else:
        txn_df, cards_df, merchants_df, devices_df, customers_df = \
            generate_synthetic_dataset(output_dir=config["data_dir"])
    print(f"[Main] {len(txn_df)} transactions | fraud={txn_df['is_fraud'].mean():.2%}")
    return txn_df, cards_df, merchants_df, devices_df, customers_df


def time_split(txn_clean, config):
    txn = txn_clean.sort_values("timestamp").reset_index(drop=True)
    n   = len(txn)
    t   = int(n * config["test_size"])
    v   = int(n * config["val_size"])
    train_df = txn.iloc[:n-v-t].reset_index(drop=True)
    val_df   = txn.iloc[n-v-t:n-t].reset_index(drop=True)
    test_df  = txn.iloc[n-t:].reset_index(drop=True)
    print(f"[Main] Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    return train_df, val_df, test_df


def populate_rag(model, graph, sequences, txn_df, card_enc, explainer, config):
    print("\n[Main] Populating Graph RAG memory...")
    device   = next(model.parameters()).device
    x_dict   = {k: v.to(device) for k, v in graph.x_dict.items()}
    ei_dict  = {k: v.to(device) for k, v in graph.edge_index_dict.items()}
    known    = set(card_enc.classes_)
    sample   = txn_df[txn_df["card_id"].isin(known)].sample(
                   min(3000, len(txn_df)), random_state=42).reset_index(drop=True)
    bs       = 256
    all_embs, all_labels, all_meta = [], [], []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(sample), bs):
            batch = sample.iloc[i:i+bs]
            cidx  = torch.tensor(card_enc.transform(batch["card_id"]), dtype=torch.long, device=device)
            ts    = pd.to_datetime(batch["timestamp"])
            raw   = torch.tensor(np.stack([
                np.log1p(batch["amount"].values),
                batch["is_international"].values.astype(float),
                np.sin(2*np.pi*ts.dt.hour/24),
                np.cos(2*np.pi*ts.dt.hour/24),
                np.sin(2*np.pi*ts.dt.dayofweek/7),
                np.cos(2*np.pi*ts.dt.dayofweek/7),
            ], axis=1).astype(np.float32), device=device)
            seqs = torch.tensor(np.stack([
                sequences.get(c, np.zeros((config["seq_len"],6)))
                for c in batch["card_id"]
            ], axis=0).astype(np.float32), device=device)
            _, embs, _ = model(x_dict, ei_dict, cidx, seqs, raw)
            all_embs.append(embs["fused_emb"])
            all_labels.append(torch.tensor(batch["is_fraud"].values, dtype=torch.float))
            all_meta.extend(batch.to_dict("records"))

    explainer.add_to_memory(
        torch.cat(all_embs),
        all_meta,
        torch.cat(all_labels)
    )
    print(f"[Main] RAG memory: {len(all_meta)} examples")


def run_test_eval(model, graph, sequences, test_df, card_enc, config):
    from training.trainer import TransactionDataset, evaluate, PlattCalibrator
    from models.fusion_model import FocalLoss
    from torch.utils.data import DataLoader
    print("\n[Main] Test evaluation...")
    device      = next(model.parameters()).device
    test_ds     = TransactionDataset(test_df, sequences, card_enc, config["seq_len"])
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    # Load best threshold/calibration if available
    thr_path = Path(config["checkpoint_dir"]) / "best_threshold.json"
    threshold = float(config.get("threshold", 0.5))
    calibrator = None
    if thr_path.exists():
        j = json.loads(thr_path.read_text())
        threshold = float(j.get("threshold", threshold))
        cal = j.get("calibration")
        if cal:
            calibrator = PlattCalibrator(a=cal.get("a", 1.0), b=cal.get("b", 0.0))

    metrics, _, _ = evaluate(model, test_loader, graph, FocalLoss(), device, threshold=threshold, calibrator=calibrator)
    print("\n" + "="*50)
    print("  FINAL TEST RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}" if isinstance(v, float) else f"  {k:<20}: {v}")
    print("="*50)
    return metrics


def main():
    print("="*60)
    print("  Credit Card Fraud Detection — GNN + Transformer + RAG")
    print("="*60)

    for d in [CONFIG["output_dir"], CONFIG["checkpoint_dir"], CONFIG["data_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # 1. Data
    txn_df, cards_df, merchants_df, devices_df, customers_df = load_or_generate_data(CONFIG)

    # 2. Graph
    print("\n[Main] Building graph...")
    builder = FraudGraphBuilder()
    builder.fit(txn_df, cards_df, merchants_df, devices_df)
    graph, txn_clean = builder.build_graph(txn_df, cards_df, merchants_df, devices_df)
    builder.save("models/graph_builder.pkl")

    print("[Main] Building sequences...")
    sequences, _, _ = build_temporal_sequences(txn_df, seq_len=CONFIG["seq_len"])

    # 3. Split
    train_df, val_df, test_df = time_split(txn_clean, CONFIG)

    # 4. Model
    print("\n[Main] Building model...")
    model  = build_model(CONFIG)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Main] Parameters: {params:,}")

    # 5. Train
    train_ds = TransactionDataset(train_df, sequences, builder.card_enc, CONFIG["seq_len"])
    val_ds   = TransactionDataset(val_df,   sequences, builder.card_enc, CONFIG["seq_len"])
    print(f"[Main] Train={len(train_ds)} Val={len(val_ds)}")
    trained_model, history = train(model, train_ds, val_ds, graph, CONFIG, CONFIG["checkpoint_dir"])

    # 6. Test
    test_metrics = run_test_eval(trained_model, graph, sequences, test_df, builder.card_enc, CONFIG)

    # 7. RAG
    explainer = FraudExplainer(embedding_dim=CONFIG["hidden_dim"] // 2)
    populate_rag(trained_model, graph, sequences, train_df, builder.card_enc, explainer, CONFIG)

    # 8. Stream
    print("\n[Main] Streaming simulation...")
    pipeline = StreamingFraudPipeline(
        model=trained_model, graph=graph, explainer=explainer,
        card_encoder=builder.card_enc, seq_len=CONFIG["seq_len"],
        threshold=CONFIG["threshold"],
        checkpoint_dir=CONFIG["checkpoint_dir"],
    )
    stream_results, stream_stats = simulate_streaming(
        pipeline, txn_clean,
        n=CONFIG["stream_n_transactions"],
        delay_ms=CONFIG["stream_delay_ms"],
    )

    # 9. Save
    out = CONFIG["output_dir"]
    pd.DataFrame(stream_results).to_csv(f"{out}/stream_results.csv", index=False)
    with open(f"{out}/test_metrics.json",     "w") as f: json.dump(test_metrics,  f, indent=2)
    with open(f"{out}/stream_stats.json",     "w") as f: json.dump(stream_stats,  f, indent=2)
    with open(f"{out}/training_config.json",  "w") as f: json.dump(CONFIG,        f, indent=2)

    fraud_results = [r for r in stream_results if r.get("fraud_prediction")]
    if fraud_results:
        print("\n" + "="*60)
        print("  EXAMPLE FRAUD EXPLANATION")
        print("="*60)
        print(fraud_results[0]["explanation"])

    print(f"\n[Main] ✓ Outputs saved to {out}/")
    print(f"[Main] Stream stats:\n{json.dumps(stream_stats, indent=2)}")
    print("[Main] Done.")


if __name__ == "__main__":
    main()
