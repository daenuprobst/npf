"""One shot counterpart of the token game. Every bond place is labelled independently."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import N_BOND, Encoder, mlp
from .featurisation import BOND_ORDER


class Forward(nn.Module):
    """Every bond place is labelled independently in one pass. petri adds enabling masks and a valence repair."""

    def __init__(self, d=128, rounds=5, attention=3, petri=True):
        super().__init__()
        self.petri = petri
        self.encoder = Encoder(d, rounds, attention)

        # classes are unchanged, or re-type to none, single, double, triple, aromatic
        self.pair = mlp(2 * d + N_BOND + 1, 2 * d, N_BOND + 1)

    def forward(self, b):
        h = self.encoder(b["xa"], b["ba"], b["mask_a"])
        same_fragment = (b["frag_a"][:, :, None] == b["frag_a"][:, None, :]).float()[..., None]
        z = torch.cat([h[:, :, None] + h[:, None, :], h[:, :, None] * h[:, None, :], F.one_hot(b["ba"], N_BOND).float(), same_fragment], -1)
        logits = self.pair(z)

        # a transition that would leave the marking of B_ij unchanged does not exist
        if self.petri:
            logits = logits.masked_fill(F.one_hot(b["ba"] + 1, N_BOND + 1).bool(), -1e4)

        return logits

    @staticmethod
    def loss(logits, b):
        n = logits.shape[1]
        valid = (b["mask_a"][:, :, None] & b["mask_a"][:, None, :]) & torch.triu(torch.ones(n, n, dtype=torch.bool, device=logits.device), 1)

        return F.cross_entropy(logits[valid], b["edits"][valid])

    def decode(self, logits, b, reactions):
        """Most probable firing vector. With petri it is repaired greedily until every slack place is non-negative,
        h_i - sum_j (order_after(ij) - order_before(ij)) >= 0. A violated atom drops its least likely bond forming
        firing or fires its most likely bond breaking transition, whichever is cheaper."""
        logp = torch.log_softmax(logits.float(), -1).cpu().numpy()
        out = []
        for k, r in enumerate(reactions):
            n = len(r["a"]["x"])
            lp, before = logp[k, :n, :n], np.zeros((n, n), np.int64)

            if len(r["a"]["bonds"]):
                i, j, t = r["a"]["bonds"].T
                before[i, j] = t
                before[j, i] = t

            choice = lp.argmax(-1)
            choice = np.triu(choice, 1)

            # one decision per unordered pair
            choice = choice + choice.T

            if self.petri:
                for _ in range(8):
                    after = np.where(choice > 0, choice - 1, before)
                    hydrogens = r["a"]["h"] - (BOND_ORDER[after] - BOND_ORDER[before]).sum(1)
                    bad = np.nonzero(hydrogens < -1e-6)[0]
                    if not len(bad):
                        break

                    i = bad[np.argmin(hydrogens[bad])]
                    gain = BOND_ORDER[after[i]] - BOND_ORDER[before[i]]

                    # options are (cost in log probability, atom j, new choice)
                    options = []

                    # undo a firing that adds tokens to B_ij
                    for j in np.nonzero(gain > 0)[0]:
                        options.append((lp[i, j, choice[i, j]] - lp[i, j, 0], j, 0))

                    # or fire a transition that removes tokens
                    for j in np.nonzero((before[i] > 0) & (choice[i] == 0))[0]:
                        lower = [c for c in range(1, N_BOND + 1) if BOND_ORDER[c - 1] < BOND_ORDER[before[i, j]]]
                        c = max(lower, key=lambda c: lp[i, j, c])
                        options.append((lp[i, j, 0] - lp[i, j, c], j, c))

                    if not options:
                        break

                    _, j, c = min(options, key=lambda o: o[0])
                    choice[i, j] = choice[j, i] = c

            i, j = np.nonzero(np.triu(choice, 1))
            out.append(np.stack([i, j, choice[i, j] - 1], 1))

        return out
