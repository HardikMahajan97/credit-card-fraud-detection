"""
Fusion Model: GNN + Transformer + Classifier
Pure PyTorch — no C++ extension dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.gnn_model import build_gnn
from models.transformer_model import TransactionTransformer, TemporalAnomalyHead

RAW_FEAT_DIM = 10


class FusionFraudDetector(nn.Module):
    def __init__(
        self,
        gnn_variant: str = "graphsage",
        gnn_out_dim: int = 32,
        transformer_out_dim: int = 32,
        raw_feat_dim: int = RAW_FEAT_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        seq_len: int = 10,
        use_anomaly_heads: bool = True,
        card_in_dim: int = 3,
        merchant_in_dim: int = 3,
        device_in_dim: int = 2,
        max_neighbors: int = 30,
    ):
        super().__init__()

        self.gnn = build_gnn(
            variant=gnn_variant,
            card_in_dim=card_in_dim,
            merchant_in_dim=merchant_in_dim,
            device_in_dim=device_in_dim,
            out_dim=gnn_out_dim,
            max_neighbors=max_neighbors,
        )

        self.transformer = TransactionTransformer(
            input_dim=raw_feat_dim,
            out_dim=transformer_out_dim,
            max_seq_len=seq_len,
        )

        self.use_anomaly_heads = use_anomaly_heads
        if use_anomaly_heads:
            self.anomaly_head = TemporalAnomalyHead(in_dim=transformer_out_dim)

        fusion_in = gnn_out_dim + transformer_out_dim + raw_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim // 2, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_dict, edge_index_dict, card_indices,
                sequences, raw_features, seq_mask=None):
        node_embs = self.gnn(x_dict, edge_index_dict)
        gnn_emb   = node_embs["card"][card_indices]

        temporal_emb = self.transformer(sequences, src_key_padding_mask=seq_mask)

        fused    = torch.cat([gnn_emb, temporal_emb, raw_features], dim=-1)
        fused    = self.fusion(fused)
        logit    = self.classifier(fused).squeeze(-1)
        fraud_prob = torch.sigmoid(logit)

        anomaly_scores = {}
        if self.use_anomaly_heads:
            anomaly_scores = self.anomaly_head(temporal_emb)

        embeddings = {
            "gnn_emb":      gnn_emb.detach(),
            "temporal_emb": temporal_emb.detach(),
            "fused_emb":    fused.detach(),
        }
        return fraud_prob, embeddings, anomaly_scores

    def predict(self, *args, threshold=0.5, **kwargs):
        self.eval()
        with torch.no_grad():
            prob, embs, scores = self.forward(*args, **kwargs)
        return (prob >= threshold).float(), prob, embs, scores


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )
        pt  = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == "mean" else loss


def build_model(config: dict) -> FusionFraudDetector:
    return FusionFraudDetector(
        gnn_variant=config.get("gnn_variant", "graphsage"),
        gnn_out_dim=config.get("gnn_out_dim", 32),
        transformer_out_dim=config.get("transformer_out_dim", 32),
        hidden_dim=config.get("hidden_dim", 128),
        dropout=config.get("dropout", 0.3),
        seq_len=config.get("seq_len", 10),
        use_anomaly_heads=config.get("use_anomaly_heads", True),
        max_neighbors=config.get("gnn_max_neighbors", 30),
    )


if __name__ == "__main__":
    model = FusionFraudDetector()
    B = 8
    n_c, n_m, n_d = 50, 20, 80
    x_dict = {"card": torch.randn(n_c,3), "merchant": torch.randn(n_m,3), "device": torch.randn(n_d,2)}
    ei = {
        ("card","pays","merchant"): torch.stack([torch.randint(0,n_c,(100,)), torch.randint(0,n_m,(100,))]),
        ("card","uses","device"):   torch.stack([torch.randint(0,n_c,(100,)), torch.randint(0,n_d,(100,))]),
    }
    probs, embs, scores = model(x_dict, ei, torch.randint(0,n_c,(B,)),
                                torch.randn(B,10,10), torch.randn(B,10))
    print(f"[FusionModel] fraud_prob: {probs.shape}")
    print("[FusionModel] Smoke test passed.")
