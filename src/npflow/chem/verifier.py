"""A verifier for the candidates of the token game."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder, mlp
from .featurisation import BOND_ORDER


class Verifier(nn.Module):
    """Scores a candidate firing vector sigma of the token game by a state-equation readout. The marking m_A + C sigma
    lives on the precursor atoms, so no correspondence is needed, and with a local encoder the readout is exactly zero
    on every atom whose R-ball the firing does not change. The score adds to the beam log probability, so an untrained
    verifier keeps the beam order."""

    def __init__(self, d=128, rounds=4, attention=0):
        super().__init__()
        self.encoder = Encoder(d, rounds, attention, n_extra=8)
        self.atom = mlp(d, 2 * d, d)
        self.out = mlp(d + 1, 2 * d, 1)

        # zero output at the start, so that training begins from the order of the beam
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self.register_buffer("order", torch.tensor(BOND_ORDER, dtype=torch.float32))

    def marking(self, b, cur, fired):
        # slack tokens follow from the valence invariant, as in the token game
        hydrogens = b["h_a"] - (self.order[cur] - self.order[b["ba"]]).sum(2)
        touched = fired.any(2, keepdim=True).float()
        x = torch.cat(
            [
                b["xa"],
                F.one_hot((hydrogens.round().long() + 2).clamp(0, 6), 7).float(),
                touched,
            ],
            -1,
        )

        return self.atom(self.encoder(x, cur, b["mask_a"])) * b["mask_a"][..., None]

    def forward(self, b, edits, log_p):
        """edits [B, N, N] holds 0 for unchanged and k + 1 for the new bond type k, log_p [B] comes from the beam."""
        fired = edits > 0
        cur = torch.where(fired, edits - 1, b["ba"])
        readout = (
            self.marking(b, cur, fired)
            - self.marking(b, b["ba"], torch.zeros_like(fired))
        ).sum(1)

        return log_p + self.out(torch.cat([readout, log_p[:, None]], -1)).squeeze(-1)
