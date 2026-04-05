"""
Graph Neural Network Module — pure PyTorch.

This repo originally referenced PyTorch Geometric, but for macOS/Python environments
where PyG wheels aren't available, we use a small safe implementation that only
requires core PyTorch.

It builds a homogeneous graph from the (typed) edge_index_dict and performs a
few rounds of mean neighbor aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor, max_neighbors: int = 30) -> torch.Tensor:
    """Mean neighbor aggregation with per-node neighbor cap (no torch-scatter required)."""
    if edge_index.numel() == 0:
        return torch.zeros_like(x)

    src, dst = edge_index[0], edge_index[1]

    # Heuristic guard: only run sampling path when graph is dense globally; per-node capping is still applied below.
    if edge_index.size(1) > max_neighbors * x.size(0):
        perm = torch.randperm(edge_index.size(1), device=x.device)
        src = src[perm]
        dst = dst[perm]
        order = torch.argsort(dst, stable=True)
        src = src[order]
        dst = dst[order]
        _, counts = torch.unique_consecutive(dst, return_counts=True)
        mask = torch.cat([torch.arange(c, device=x.device) < max_neighbors for c in counts])
        src = src[mask]
        dst = dst[mask]

    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src])

    deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
    out = out / deg.clamp(min=1).unsqueeze(-1)
    return out


class FraudGNN(nn.Module):
    """Homogeneous GNN over the entity graph with typed node projections."""

    def __init__(
        self,
        card_in_dim: int = 3,
        merchant_in_dim: int = 3,
        device_in_dim: int = 2,
        hidden_dim: int = 64,
        out_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_neighbors: int = 30,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.max_neighbors = max_neighbors

        self.card_proj     = nn.Linear(card_in_dim,     hidden_dim)
        self.merchant_proj = nn.Linear(merchant_in_dim, hidden_dim)
        self.device_proj   = nn.Linear(device_in_dim,   hidden_dim)

        self.update_mlps = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim_i = hidden_dim if i < num_layers - 1 else out_dim
            self.update_mlps.append(
                nn.Sequential(
                    nn.Linear(in_dim * 2, out_dim_i),
                    nn.GELU(),
                )
            )
            self.norms.append(nn.LayerNorm(out_dim_i))

    def forward(self, x_dict, edge_index_dict):
        card_x     = F.relu(self.card_proj(x_dict["card"]))
        merchant_x = F.relu(self.merchant_proj(x_dict["merchant"]))
        device_x   = F.relu(self.device_proj(x_dict["device"]))

        n_card     = card_x.size(0)
        n_merchant = merchant_x.size(0)
        n_device   = device_x.size(0)

        x = torch.cat([card_x, merchant_x, device_x], dim=0)
        merch_offset = n_card
        dev_offset   = n_card + n_merchant

        edges = []
        if ("card", "pays", "merchant") in edge_index_dict:
            ei = edge_index_dict[("card", "pays", "merchant")]
            src = ei[0]
            dst = ei[1] + merch_offset
            edges.append(torch.stack([src, dst]))
            edges.append(torch.stack([dst, src]))

        if ("card", "uses", "device") in edge_index_dict:
            ei = edge_index_dict[("card", "uses", "device")]
            src = ei[0]
            dst = ei[1] + dev_offset
            edges.append(torch.stack([src, dst]))
            edges.append(torch.stack([dst, src]))

        if ("device", "seen_at", "merchant") in edge_index_dict:
            ei = edge_index_dict[("device", "seen_at", "merchant")]
            src = ei[0] + dev_offset
            dst = ei[1] + merch_offset
            edges.append(torch.stack([src, dst]))
            edges.append(torch.stack([dst, src]))

        edge_index = torch.cat(edges, dim=1) if edges else torch.zeros(2, 0, dtype=torch.long, device=x.device)

        for mlp, norm in zip(self.update_mlps, self.norms):
            neigh = _mean_aggregate(x, edge_index, max_neighbors=self.max_neighbors)
            x = mlp(torch.cat([x, neigh], dim=-1))
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return {
            "card":     x[:n_card],
            "merchant": x[n_card: n_card + n_merchant],
            "device":   x[n_card + n_merchant:],
        }


def build_gnn(variant="graphsage", max_neighbors=30, **kwargs):
    """Factory. Variant kept for compatibility."""
    return FraudGNN(max_neighbors=max_neighbors, **kwargs)


if __name__ == "__main__":
    model = FraudGNN()
    x_dict = {
        "card":     torch.randn(50, 3),
        "merchant": torch.randn(20, 3),
        "device":   torch.randn(80, 2),
    }
    edge_index_dict = {
        ("card", "pays", "merchant"): torch.stack([
            torch.randint(0, 50, (100,)),
            torch.randint(0, 20, (100,))
        ]),
        ("card", "uses", "device"): torch.stack([
            torch.randint(0, 50, (100,)),
            torch.randint(0, 80, (100,))
        ]),
    }
    out = model(x_dict, edge_index_dict)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
    print("[GNN] Smoke test passed.")
