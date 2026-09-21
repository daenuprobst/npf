"""Atom mapping as the equilibrium of an assignment net."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .encoder import N_BOND, Encoder, PetriLayer, mlp


def sinkhorn(log_s, n_iter=10):
    """Log of the equilibrium marking of the assignment net for energies -log_s.

    Rows are product atoms with mass 1 plus a last slack row that takes the precursor atoms which leave, columns are
    precursor atoms with mass 1. The partner swaps Z_ij + Z_kl <-> Z_il + Z_kj conserve exactly the row and column
    sums, and the equilibrium with these marginals is the matrix scaling exp(log_s + u_i + v_j).
    """
    B, R, C = log_s.shape
    log_r = torch.zeros(B, R, device=log_s.device)
    log_r[:, -1] = np.log(max(C - R + 1, 1))
    u, v = torch.zeros(B, R, device=log_s.device), torch.zeros(B, C, device=log_s.device)
    for _ in range(n_iter):
        u = log_r - torch.logsumexp(log_s + v[:, None, :], 2)
        v = -torch.logsumexp(log_s + u[:, :, None], 1)

    return log_s + u[:, :, None] + v[:, None, :]


class Mapper(nn.Module):
    """Atom mapping as the correspondence between the places of B and of A under which the firing is most probable.

    Every round scores a pair by the bonds it keeps under the current correspondence, checks the correspondence as a
    net morphism and relaxes to the equilibrium of the assignment net. petri False is the generic counterpart, the
    other switches remove one component each for the ablations.
    """

    def __init__(self, d=192, rounds=6, petri=True, consensus=10, equilibrium=True, kept_bonds=True, morphism=True, token_cost=True):
        super().__init__()
        self.petri, self.consensus = petri, consensus if petri else 0
        self.use_equilibrium, self.use_kept, self.use_morphism, self.use_cost = equilibrium, kept_bonds, morphism, token_cost
        self.encoder = Encoder(d, rounds)
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.slack = nn.Parameter(torch.zeros(()))

        # log rate of keeping a bond of one type in A as a bond of another type in B. every bond that is not kept
        # costs a firing, which is minimum chemical distance with learned costs
        self.keep = nn.Parameter(torch.eye(N_BOND - 1))
        self.alpha = nn.Parameter(torch.ones(max(self.consensus, 1)))

        # a hydrogen or a charge that has to move is a firing as well
        self.token_cost = nn.Parameter(torch.ones(2))
        self.agree_b, self.agree_a = mlp(3 * d, 2 * d, d), mlp(3 * d, 2 * d, d)
        self.spread_b, self.spread_a = PetriLayer(d), PetriLayer(d)

    def forward(self, b, all_rounds=False):
        ha = self.encoder(b["xa"], b["ba"], b["mask_a"])
        hb = self.encoder(b["xb"], b["bb"], b["mask_b"])
        moved = self.token_cost[0] * (b["h_b"][:, :, None] - b["h_a"][:, None, :]).abs() \
            + self.token_cost[1] * (b["q_b"][:, :, None] - b["q_a"][:, None, :]).abs() if self.petri and self.use_cost else 0.0
        score = lambda: self.q(hb) @ self.k(ha).transpose(1, 2) / ha.shape[-1] ** 0.5 - moved

        # elements are conserved, and padded product rows behave like slack
        allowed = (b["el_b"][:, :, None] == b["el_a"][:, None, :]) & b["mask_a"][:, None, :]
        pad_row = ~b["mask_b"][:, :, None]

        def equilibrium(s):
            s = torch.where(allowed, s, torch.full_like(s, -1e4))

            # generic readout, every product atom picks its most similar precursor atom
            if not self.petri or not self.use_equilibrium:
                return torch.log_softmax(s, 2)

            s = torch.where(pad_row, torch.zeros_like(s), s)

            return sinkhorn(torch.cat([s, self.slack.expand(len(s), 1, s.shape[2])], 1))[:, :-1]

        rounds = [equilibrium(score())]
        adj_a = F.one_hot(b["ba"], N_BOND)[..., 1:].float()
        adj_b = F.one_hot(b["bb"], N_BOND)[..., 1:].float()
        for t in range(self.consensus):
            p = rounds[-1].exp() * b["mask_b"][:, :, None]

            # expected number of bonds of product atom i that survive if i sits on precursor atom j
            kept = torch.einsum("bikt,bkl,bjlu,tu->bij", adj_b, p, adj_a, self.keep) if self.use_kept else 0.0

            # a mapping should be a net morphism. every atom compares itself with its expected partner and the
            # disagreement spreads along the bond places
            if self.use_morphism:
                partner_b, partner_a = p @ ha, p.transpose(1, 2) @ hb
                hb = hb + self.spread_b(self.agree_b(torch.cat([hb, partner_b, hb - partner_b], -1)), adj_b) * b["mask_b"][..., None]
                ha = ha + self.spread_a(self.agree_a(torch.cat([ha, partner_a, ha - partner_a], -1)), adj_a) * b["mask_a"][..., None]

            rounds.append(equilibrium(score() + self.alpha[t] * kept))

        return rounds if all_rounds else rounds[-1]

    @staticmethod
    def loss(log_p, b):
        """Likelihood of the recorded mapping up to symmetry. Product atom i may sit on any precursor atom that is
        equivalent to the recorded partner of any product atom equivalent to i."""
        partner = torch.gather(b["sym_a"], 1, b["target"].clamp(min=0))
        same = (b["sym_b"][:, :, None] == b["sym_b"][:, None, :]) & b["mask_b"][:, None, :]
        same = ((partner[:, None, :, None] == b["sym_a"][:, None, None, :]) & same[..., None]).any(2)
        ll = torch.logsumexp(torch.where(same & b["mask_a"][:, None, :], log_p, torch.full_like(log_p, -1e4)), 2)

        # a product atom without a seat in the target carries no likelihood
        seated = b["mask_b"] & (b["target"] >= 0)

        return -(ll * seated).sum() / seated.sum().clamp(min=1)

    @staticmethod
    def decode(log_p, b):
        """One to one assignment per reaction by the Hungarian algorithm. Returns arrays product atom to precursor atom."""
        out = []
        for lp, mb, ma in zip(log_p.detach().cpu().numpy(), b["mask_b"].cpu().numpy(), b["mask_a"].cpu().numpy()):
            rows, cols = linear_sum_assignment(-lp[mb][:, ma])
            out.append(cols[np.argsort(rows)])

        return out
