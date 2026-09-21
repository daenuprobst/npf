"""Message passing on the place graph, one edge per input and output place of a transition."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import N_PLACE_FEAT, N_TRANS_FEAT, mlp, place_features, scatter_sum, trans_features


class GNN(nn.Module):
    """Baseline without transition nodes. Places exchange messages along input to output pairs of every transition,
    firing counts are read from the pooled input and output places of a transition."""

    def __init__(self, task, hidden=64, rounds=4):
        super().__init__()
        self.task = task
        self.enc = mlp(N_PLACE_FEAT, hidden, hidden)
        n_edge = 2 + N_TRANS_FEAT
        self.layers = nn.ModuleList(
            nn.ModuleDict({
                "fwd": mlp(2 * hidden + n_edge, hidden, hidden),
                "bwd": mlp(2 * hidden + n_edge, hidden, hidden),
                "upd": mlp(3 * hidden, hidden, hidden),
            })
            for _ in range(rounds)
        )
        self.pool_in, self.pool_out = mlp(hidden + 1, hidden, hidden), mlp(hidden + 1, hidden, hidden)
        self.out = mlp(2 * hidden + N_TRANS_FEAT if task == "transitions" else hidden, hidden, 1)

    def forward(self, b, m=None):
        m = b.m if m is None else m
        h = self.enc(place_features(b, m))
        edge = torch.cat([b.edge_wu[:, None], b.edge_wv[:, None], trans_features(b)[b.edge_t]], -1)
        for layer in self.layers:
            x = torch.cat([h[b.edge_u], h[b.edge_v], edge], -1)
            m_in = scatter_sum(layer["fwd"](x), b.edge_v, b.n_places)
            m_out = scatter_sum(layer["bwd"](x), b.edge_u, b.n_places)
            h = h + layer["upd"](torch.cat([h, m_in, m_out], -1))

        if self.task == "next":
            return m + self.out(h).squeeze(-1)

        z_in = scatter_sum(self.pool_in(torch.cat([h[b.pre_p], b.pre_w[:, None]], -1)), b.pre_t, b.n_trans)
        z_out = scatter_sum(self.pool_out(torch.cat([h[b.pos_p], b.pos_w[:, None]], -1)), b.pos_t, b.n_trans)

        return F.softplus(self.out(torch.cat([z_in, z_out, trans_features(b)], -1)).squeeze(-1))
