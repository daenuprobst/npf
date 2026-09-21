"""NPF on a net whose arc weights are learned as functions of the place types at the two ends of an arc."""
import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import apply_incidence
from .npf import NPF

# P- and T-invariants are not continuous in the arc weights. A cycle of conversions is a T-invariant only if
# its ratios multiply to exactly one, and learned weights are never exact. The numerical rank of the learned
# incidence matrix is therefore taken with a tolerance
RANK_TOLERANCE = 1e-3


class SheafNPF(NPF):
    """Scalar restriction maps of a cellular sheaf on the net, indexed by the types of the two ends of an arc."""

    def __init__(self, *, n_types=4, consistency=True, flat=False):
        super().__init__("transitions")
        self.consistency, self.flat = consistency, flat

        # flat, the ratio of two types is a difference of type potentials, so the ratios of every cycle of
        # conversions multiply to one by construction. the gain graph is then balanced, the learned incidence
        # matrix is C_theta = D_P C D_T^{-1} with positive diagonal D, its rank cannot drop, and the invariants
        # are exact, x_theta = D_P^{-1} x and y_theta = D_T y, instead of holding only at the true weights
        self.potential = nn.Parameter(torch.zeros(n_types)) if flat else None
        self.raw_ratio = None if flat else nn.Parameter(torch.zeros(n_types, n_types))

    @property
    def log_ratio(self):
        # Adam steps are scale free, so the arc weights move ten times faster than the MLP weights
        if self.flat:
            u = 10.0 * self.potential
            return u[None, :] - u[:, None]

        return 10.0 * self.raw_ratio

    def learned_net(self, b, detach=False):
        kind = b.e[:, 0].long()
        source = torch.zeros(b.n_trans, dtype=torch.long, device=kind.device).index_copy_(0, b.pre_t, kind[b.pre_p])
        log_ratio = self.log_ratio.detach() if detach else self.log_ratio
        pos_w = torch.exp(log_ratio[source[b.pos_t], kind[b.pos_p]])
        G, S, Pmax, Tmax = b.pad_shape

        # arcs of the first sample of every net carry the structure
        first = lambda pad, size: (pad // size) % S == 0
        dense = torch.zeros(G, Pmax, Tmax, dtype=torch.float64, device=kind.device)
        for p, t, w, sign in ((b.pos_p, b.pos_t, pos_w, 1.0), (b.pre_p, b.pre_t, b.pre_w, -1.0)):
            keep = first(b.pad_p[p], Pmax)
            g = b.pad_p[p][keep] // (S * Pmax)
            dense.index_put_((g, b.pad_p[p][keep] % Pmax, b.pad_t[t][keep] % Tmax), sign * w[keep].double(), accumulate=True)

        # a flat parametrisation keeps the rank of the unit net, so the pseudoinverse needs no tolerance
        rtol = 1e-9 if self.flat else RANK_TOLERANCE

        return dataclasses.replace(b, pos_w=pos_w, C=dense.float(), C_pinv=torch.linalg.pinv(dense, rtol=rtol).float())

    def forward(self, b, m=None):
        # with the consistency loss the arc weights are identified by the state equation alone
        return super().forward(self.learned_net(b, detach=self.consistency), m)

    def auxiliary_loss(self, b, sigma):
        # m_B - m_A = C_theta sigma* is a linear regression for the arc weights, independent of the rate law
        if not self.consistency:
            return 0.0

        return F.mse_loss(apply_incidence(sigma, self.learned_net(b)), b.m_b - b.m)
