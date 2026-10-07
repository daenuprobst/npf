"""Message passing on the valence net of a molecule."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .featurisation import N_ATOM_FEAT

# bond types none, single, double, triple, aromatic
N_BOND = 5


def mlp(n_in, n_hidden, n_out):
    return nn.Sequential(
        nn.Linear(n_in, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_out)
    )


class PetriLayer(nn.Module):
    """One round of place and transition message passing on the valence net, as in PGNN Eqs. 9 to 11.
    Every marked bond place B_ij links the atoms i and j, with one message function per bond type.
    """

    def __init__(self, d, channels=N_BOND - 1):
        super().__init__()
        self.channels = channels
        self.rel = nn.Linear(d, channels * d, bias=False)
        self.upd = mlp(2 * d, 2 * d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, adj):
        B, N, d = h.shape
        m = torch.einsum(
            "bijt,bjtd->bid", adj, self.rel(h).view(B, N, self.channels, d)
        )

        return self.norm(h + self.upd(torch.cat([h, m], -1)))


class Encoder(nn.Module):
    """rounds of message passing along marked bond places. attention layers also let atoms exchange messages that
    are not bonded, these are the neighbourhoods of the bond forming transitions."""

    def __init__(self, d=128, rounds=4, attention=0, n_extra=0):
        super().__init__()
        self.inp = nn.Linear(N_ATOM_FEAT + n_extra, d)
        self.layers = nn.ModuleList(PetriLayer(d) for _ in range(rounds))
        self.attn = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d, 8, 2 * d, dropout=0.0, batch_first=True, norm_first=True
            )
            for _ in range(attention)
        )

    def forward(self, x, bonds, mask, all_depths=False):
        adj = F.one_hot(bonds, N_BOND)[..., 1:].to(x.dtype)
        h = self.inp(x) * mask[..., None]
        depths = [h]
        for k, layer in enumerate(self.layers):
            h = layer(h, adj) * mask[..., None]

            if k < len(self.attn):
                h = self.attn[k](h, src_key_padding_mask=~mask) * mask[..., None]

            depths.append(h)

        return depths if all_depths else h
