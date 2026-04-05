"""
Training Pipeline — Python 3.13 / PyTorch 2.6+ safe.
No C++ extensions required.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve,
)
from pathlib import Path
import json, time, copy, random
from typing import Optional

from models.fusion_model import FusionFraudDetector, FocalLoss, build_model


def build_seq_mask(sequences: torch.Tensor) -> torch.Tensor:
    """Return a boolean padding mask for nn.Transformer (True = PAD)."""
    # sequences: (B, T, F). Padding rows are all zeros in current pipeline.
    return (sequences.abs().sum(dim=-1) == 0)


class TransactionDataset(Dataset):
    def __init__(self, transactions_df, sequences, card_encoder, seq_len=10, per_txn_sequences=None):
        self.seq_len = seq_len
        known = set(card_encoder.classes_)
        df = transactions_df[transactions_df["card_id"].isin(known)].reset_index(drop=True)
        self.df           = df
        self.sequences    = sequences
        self.per_txn_sequences = per_txn_sequences or {}
        self.card_indices = card_encoder.transform(df["card_id"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        card_idx = int(self.card_indices[idx])
        card_id  = row["card_id"]

        txn_id = row.get("transaction_id")
        seq = self.per_txn_sequences.get(txn_id)
        if seq is None:
            seq = self.sequences.get(card_id, np.zeros((self.seq_len, 10), dtype=np.float32))
        seq = np.array(seq, dtype=np.float32)
        seq = torch.tensor(seq, dtype=torch.float32)
        seq_mask = build_seq_mask(seq.unsqueeze(0)).squeeze(0)

        if txn_id in self.per_txn_sequences:
            raw = torch.tensor(seq[-1], dtype=torch.float32)
        else:
            ts = pd.Timestamp(row["timestamp"])
            raw = torch.tensor([
                np.log1p(float(row["amount"])),
                float(row["is_international"]),
                np.sin(2 * np.pi * ts.hour / 24),
                np.cos(2 * np.pi * ts.hour / 24),
                np.sin(2 * np.pi * ts.dayofweek / 7),
                np.cos(2 * np.pi * ts.dayofweek / 7),
                0.0,
                np.log1p(604800.0),
                0.0,
                0.0,
            ], dtype=torch.float32)

        label = torch.tensor(float(row["is_fraud"]), dtype=torch.float32)
        return card_idx, seq, raw, seq_mask, label


class ReplayBuffer:
    def __init__(self, max_fraud=2000, max_normal=2000):
        self.max_fraud  = max_fraud
        self.max_normal = max_normal
        self.fraud_samples  = []
        self.normal_samples = []

    def add(self, df):
        self.fraud_samples.extend(df[df["is_fraud"]==1].to_dict("records"))
        self.normal_samples.extend(df[df["is_fraud"]==0].to_dict("records"))
        if len(self.fraud_samples)  > self.max_fraud:
            self.fraud_samples  = random.sample(self.fraud_samples,  self.max_fraud)
        if len(self.normal_samples) > self.max_normal:
            self.normal_samples = random.sample(self.normal_samples, self.max_normal)

    def get_replay_df(self):
        samples = self.fraud_samples + self.normal_samples
        return pd.DataFrame(samples) if samples else pd.DataFrame()


def make_weighted_sampler(dataset):
    labels        = dataset.df["is_fraud"].values
    counts        = np.bincount(labels.astype(int), minlength=2)
    # Higher weight for minority class
    class_weights = (counts.sum() / (counts + 1e-8))
    w = torch.tensor([class_weights[int(l)] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0
        self.best_score  = None
        self.should_stop = False

    def step(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    n_pos = y_true.sum()
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_pred_prob) if n_pos > 0 and n_pos < len(y_true) else 0.0,
        "pr_auc":    average_precision_score(y_true, y_pred_prob) if n_pos > 0 else 0.0,
        "fpr":       fp / (fp + tn + 1e-9),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def fraud_kpis(y_true: np.ndarray, y_prob: np.ndarray,
               top_k_frac: float = 0.01,
               fpr_target: float = 0.05,
               precision_target: float = 0.5) -> dict:
    """Compute fraud-specific threshold-based KPIs.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_prob: Predicted fraud probabilities.
        top_k_frac: Fraction of top-scored transactions to use for precision@k
                    (default 1%).
        fpr_target: FPR level at which recall is evaluated (default 5%).
        precision_target: Precision level at which recall is evaluated
                          (default 50%).

    Returns:
        Dict with keys:
          - ``precision_at_k``: precision among top ``top_k_frac`` scored rows.
          - ``recall_at_fpr{N}``: recall when FPR ≤ fpr_target (0 if infeasible).
          - ``recall_at_precision{N}``: recall when precision ≥ precision_target
            (0 if infeasible).
          - ``pr_auc``: area under precision-recall curve (for convenience).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    n_pos = y_true.sum()
    n = len(y_true)

    results: dict = {}

    # precision@k: top top_k_frac fraction by score
    k = max(1, int(np.ceil(n * top_k_frac)))
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    if k > 0:
        results["precision_at_k"] = float(y_true[top_k_idx].mean())
    else:
        results["precision_at_k"] = 0.0

    # recall@FPR: walk the ROC curve; find first point where FPR ≤ fpr_target
    if n_pos > 0 and n_pos < n:
        from sklearn.metrics import roc_curve
        fprs, tprs, _ = roc_curve(y_true, y_prob)
        # find the highest TPR (recall) achievable at FPR ≤ fpr_target
        mask = fprs <= fpr_target
        results[f"recall_at_fpr{int(fpr_target * 100)}pct"] = (
            float(tprs[mask].max()) if mask.any() else 0.0
        )
    else:
        results[f"recall_at_fpr{int(fpr_target * 100)}pct"] = 0.0

    # recall@precision: walk the PR curve; find highest recall where precision ≥ target
    if n_pos > 0:
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
        mask = precisions >= precision_target
        results[f"recall_at_precision{int(precision_target * 100)}pct"] = (
            float(recalls[mask].max()) if mask.any() else 0.0
        )
    else:
        results[f"recall_at_precision{int(precision_target * 100)}pct"] = 0.0

    # PR-AUC (convenience duplicate so callers need only this dict)
    results["pr_auc"] = (
        float(average_precision_score(y_true, y_prob)) if n_pos > 0 else 0.0
    )

    return results


def fraud_composite_score(pr_auc: float, roc_auc: float, f1: float) -> float:
    """Fraud-focused composite score (no accuracy, PR-AUC weighted highest).

    Formula:  0.5 × PR-AUC + 0.3 × ROC-AUC + 0.2 × F1

    Rationale:
    - PR-AUC is the primary metric for imbalanced fraud data (0.5 weight).
    - ROC-AUC captures overall ranking quality (0.3 weight).
    - F1 reflects threshold-specific trade-off (0.2 weight).
    - Accuracy is excluded: it is dominated by the majority class and
      misleading on highly imbalanced datasets.

    Safe fallback: any NaN/inf input is treated as 0.
    """
    def _safe(v):
        return float(v) if np.isfinite(v) else 0.0
    return 0.5 * _safe(pr_auc) + 0.3 * _safe(roc_auc) + 0.2 * _safe(f1)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, objective: str = "f1"):
    """Select threshold from PR curve; default maximizes F1 on validation."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns precision/recall of len(thresholds)+1
    thresholds = np.concatenate([thresholds, [1.0]])

    if objective == "f1":
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        ix = int(np.nanargmax(f1))
        return float(thresholds[ix]), float(f1[ix])

    raise ValueError(f"Unknown objective: {objective}")


class PlattCalibrator:
    """Fast logistic calibration: p' = sigmoid(a * logit(p) + b)."""

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = float(a)
        self.b = float(b)

    def fit(self, probs: np.ndarray, y: np.ndarray, max_iter: int = 200, lr: float = 0.05):
        x = np.log(probs.clip(1e-6, 1 - 1e-6) / (1 - probs.clip(1e-6, 1 - 1e-6)))
        a = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
        b = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
        xt = torch.tensor(x, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        opt = torch.optim.LBFGS([a, b], lr=lr, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            logits = a * xt + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
            loss.backward()
            return loss

        opt.step(closure)
        self.a = float(a.detach().cpu())
        self.b = float(b.detach().cpu())
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        x = np.log(probs.clip(1e-6, 1 - 1e-6) / (1 - probs.clip(1e-6, 1 - 1e-6)))
        logits = self.a * x + self.b
        return 1.0 / (1.0 + np.exp(-logits))

    def state_dict(self):
        return {"a": self.a, "b": self.b}


def _graph_to_device(graph_data, device):
    x_dict = {k: v.to(device) for k, v in graph_data.x_dict.items()}
    ei_dict = {k: v.to(device) for k, v in graph_data.edge_index_dict.items()}
    return x_dict, ei_dict


def train_epoch(model, loader, graph_data, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    x_dict, ei_dict = _graph_to_device(graph_data, device)

    for card_indices, sequences, raw_features, seq_mask, labels in loader:
        card_indices = card_indices.to(device)
        sequences    = sequences.to(device)
        raw_features = raw_features.to(device)
        seq_mask     = seq_mask.to(device)
        labels       = labels.to(device)

        optimizer.zero_grad()
        probs, _, _ = model(x_dict, ei_dict, card_indices, sequences, raw_features, seq_mask=seq_mask)
        logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, graph_data, criterion, device, threshold: float = 0.5, calibrator: Optional["PlattCalibrator"] = None):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    x_dict, ei_dict = _graph_to_device(graph_data, device)

    for card_indices, sequences, raw_features, seq_mask, labels in loader:
        card_indices = card_indices.to(device)
        sequences    = sequences.to(device)
        raw_features = raw_features.to(device)
        seq_mask     = seq_mask.to(device)
        labels       = labels.to(device)

        probs, _, _ = model(x_dict, ei_dict, card_indices, sequences, raw_features, seq_mask=seq_mask)
        logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
        total_loss += criterion(logits, labels).item()
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels).astype(int)
    if calibrator is not None:
        y_prob = calibrator.transform(y_prob)

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = total_loss / max(len(loader), 1)
    metrics["threshold"] = float(threshold)
    return metrics, y_true, y_prob


def train(model, train_dataset, val_dataset, graph_data, config, save_dir="models/checkpoints"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")          # MPS can be unstable with PyG; CPU is safest
    print(f"[Trainer] Device: {device}")
    model = model.to(device)

    imbalance_mode = config.get("imbalance_mode", "weighted_sampler")
    if imbalance_mode == "weighted_sampler":
        sampler = make_weighted_sampler(train_dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 256),
        sampler=sampler,
        shuffle=shuffle,
        num_workers=0,
    )
    val_loader   = DataLoader(val_dataset, batch_size=config.get("batch_size", 256),
                              shuffle=False,  num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.get("lr", 1e-3),
                                  weight_decay=config.get("weight_decay", 1e-4))

    # Adaptive scheduler: ReduceLROnPlateau monitors validation PR-AUC and halves LR on
    # plateau, giving the model more room to escape flat regions.  Falls back to
    # CosineAnnealingLR when the config explicitly requests it.
    scheduler_type = config.get("scheduler", "reduce_on_plateau")
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get("epochs", 15), eta_min=1e-5)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",           # maximise validation PR-AUC
            factor=0.5,
            patience=config.get("lr_patience", 3),
            min_lr=1e-6,
        )

    # Loss config
    alpha = float(config.get("focal_alpha", 0.25))
    gamma = float(config.get("focal_gamma", 2.0))
    pos_weight = torch.tensor([float(config.get("pos_weight", 1.0))], dtype=torch.float32, device=device)  # pos_weight=1.0 is intentional when focal_alpha >= 0.75; do not increase both simultaneously
    criterion = FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)

    early_stopper = EarlyStopping(patience=config.get("patience", 5))

    history, best_state, best_pr_auc = [], None, 0.0
    best_threshold = float(config.get("threshold", 0.5))
    calibrator = None

    for epoch in range(1, config.get("epochs", 15) + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, graph_data, optimizer, criterion, device)

        # Evaluate uncalibrated probs first.
        val_m, y_true, y_prob = evaluate(model, val_loader, graph_data, criterion, device, threshold=0.5, calibrator=None)

        # Fit calibrator on validation and re-score metrics.
        if config.get("use_calibration", True) and (y_true.sum() > 0) and (y_true.sum() < len(y_true)):
            calibrator = PlattCalibrator().fit(y_prob, y_true)
            y_prob_cal = calibrator.transform(y_prob)
        else:
            y_prob_cal = y_prob

        best_threshold, _ = find_best_threshold(y_true, y_prob_cal, objective=config.get("threshold_objective", "f1"))
        val_m2 = compute_metrics(y_true, y_prob_cal, threshold=best_threshold)
        val_m2["loss"] = val_m["loss"]
        val_m2["threshold"] = float(best_threshold)
        val_m2["calibration"] = calibrator.state_dict() if calibrator is not None else None

        # Step scheduler: ReduceLROnPlateau needs the monitored metric;
        # CosineAnnealingLR takes no argument.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_m2["pr_auc"])
        else:
            scheduler.step()

        print(
            f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
            f"acc={val_m2['accuracy']:.4f} | f1={val_m2['f1']:.4f} | "
            f"roc_auc={val_m2['roc_auc']:.4f} | pr_auc={val_m2['pr_auc']:.4f} | "
            f"prec={val_m2['precision']:.4f} | rec={val_m2['recall']:.4f} | "
            f"thr={val_m2['threshold']:.3f} | {time.time()-t0:.1f}s"
        )

        history.append({"epoch": epoch, "train_loss": train_loss, **val_m2})

        if early_stopper.step(val_m2["pr_auc"]):
            best_pr_auc = val_m2["pr_auc"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, f"{save_dir}/best_model.pt")
            with open(f"{save_dir}/best_threshold.json", "w") as f:
                json.dump({"threshold": float(best_threshold), "calibration": (calibrator.state_dict() if calibrator else None)}, f, indent=2)
            print(f"  ✓ Best PR-AUC={val_m2['pr_auc']:.4f} — saved")

        if early_stopper.should_stop:
            print(f"[Trainer] Early stop at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    with open(f"{save_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Composite training summary ──────────────────────────────────────────
    best_epoch_m = max(history, key=lambda h: h.get("f1", 0.0)) if history else {}
    best_f1_hist = best_epoch_m.get("f1", 0.0)
    best_roc     = best_epoch_m.get("roc_auc", 0.0)
    best_pr_auc  = best_epoch_m.get("pr_auc", 0.0)
    composite    = fraud_composite_score(best_pr_auc, best_roc, best_f1_hist)
    print(f"[Trainer] Done. Best PR-AUC={best_pr_auc:.4f}")
    print("\n" + "="*60)
    print("  TRAINING FRAUD-COMPOSITE SCORE SUMMARY")
    print("="*60)
    print(f"  {'PR-AUC':<22}: {best_pr_auc:.4f}")
    print(f"  {'ROC-AUC':<22}: {best_roc:.4f}")
    print(f"  {'F1 Score':<22}: {best_f1_hist:.4f}")
    print(f"  {'Fraud Composite':<22}: {composite:.4f}  (0.5×PR-AUC + 0.3×ROC-AUC + 0.2×F1)")
    print("="*60)
    return model, history
