"""
Transformer Module for Temporal Pattern Learning — Python 3.13 / PyTorch 2.6+ safe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransactionTransformer(nn.Module):
    """
    Transformer encoder for per-card transaction sequences.
    Uses only built-in nn.TransformerEncoder — no C++ extensions needed.
    """

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        out_dim: int = 32,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ):
        super().__init__()
        self.d_model  = d_model
        self.out_dim  = out_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Pass no extra kwargs — works on all PyTorch versions >= 2.0
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)
            x_pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            x_pooled = x.mean(dim=1)
        return self.output_proj(x_pooled)

    def get_attention_weights(self, x, src_key_padding_mask=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        attn_weights = []
        for layer in self.transformer.layers:
            x_norm = layer.norm1(x)
            _, w = layer.self_attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            attn_weights.append(w.detach())
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return attn_weights


class TemporalAnomalyHead(nn.Module):
    def __init__(self, in_dim: int = 32):
        super().__init__()
        self.burst_head   = nn.Linear(in_dim, 1)
        self.amount_head  = nn.Linear(in_dim, 1)
        self.channel_head = nn.Linear(in_dim, 1)

    def forward(self, temporal_emb):
        return {
            "burst_score":    torch.sigmoid(self.burst_head(temporal_emb)).squeeze(-1),
            "amount_anomaly": torch.sigmoid(self.amount_head(temporal_emb)).squeeze(-1),
            "channel_shift":  torch.sigmoid(self.channel_head(temporal_emb)).squeeze(-1),
        }


if __name__ == "__main__":
    B, seq_len, input_dim = 32, 10, 6
    model = TransactionTransformer(input_dim=input_dim)
    x   = torch.randn(B, seq_len, input_dim)
    out = model(x)
    print(f"[Transformer] {x.shape} → {out.shape}")
    head   = TemporalAnomalyHead(in_dim=32)
    scores = head(out)
    for k, v in scores.items():
        print(f"  {k}: {v.shape}")
    print("[Transformer] Smoke test passed.")
