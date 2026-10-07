"""Petri graph neural network of Ademovic Tahirovic et al. 2025, the baseline, and its variants."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import (
    N_PLACE_FEAT,
    N_TRANS_FEAT,
    kl_project_state_equation,
    mlp,
    place_features,
    project_state_equation,
    scatter_sum,
    trans_features,
)


class PGNN(nn.Module):
    """Firing counts between two markings. aggregate is incidence, the paper with the signed aggregation of Eq. 13, or
    learned, the strengthened PGNN+. state_equation adds our projection as an ablation.
    """

    def __init__(
        self,
        hidden=64,
        rounds=4,
        aggregate="incidence",
        state_equation=False,
        n_ids=0,
        n_env=0,
        kl_options=None,
    ):
        super().__init__()
        self.aggregate, self.state_equation = aggregate, state_equation
        self.enc = mlp(N_PLACE_FEAT, hidden, hidden)
        n_msg = 2 * hidden if aggregate == "learned" else hidden
        self.layers = nn.ModuleList()

        # the last entry only computes the transition messages that are read out
        for i in range(rounds + 1):
            layer = nn.ModuleDict(
                {
                    "phi_in": mlp(hidden + 1, hidden, hidden),
                    "phi_out": mlp(hidden + 1, hidden, hidden),
                    "psi": mlp(2 * hidden + N_TRANS_FEAT, hidden, hidden),
                }
            )

            if i < rounds:
                layer["upd"] = mlp(hidden + n_msg, hidden, hidden)

                if aggregate == "learned":
                    layer["from_in"] = mlp(hidden + 1, hidden, hidden)
                    layer["from_out"] = mlp(hidden + 1, hidden, hidden)

            self.layers.append(layer)

        self.out = mlp(hidden, hidden, 1)

        # the same identities and environment as NPF, an additive term per transition on the readout
        self.n_ids, self.n_env, self.kl_options = n_ids, n_env, kl_options

        if n_ids:
            self.const = nn.Embedding(n_ids, 1)
            nn.init.zeros_(self.const.weight)

        if n_ids and n_env:
            self.orders = nn.Embedding(n_ids, n_env)
            nn.init.zeros_(self.orders.weight)

    def project(self, sigma, b):
        """Our projection on the state equation, the I projection when kl_options are given."""
        if self.kl_options is None:
            return project_state_equation(sigma, b)

        return kl_project_state_equation(sigma, b, **self.kl_options)

    def message(self, layer, h, b):
        z_in = scatter_sum(
            layer["phi_in"](torch.cat([h[b.pre_p], b.pre_w[:, None]], -1)),
            b.pre_t,
            b.n_trans,
        )
        z_out = scatter_sum(
            layer["phi_out"](torch.cat([h[b.pos_p], b.pos_w[:, None]], -1)),
            b.pos_t,
            b.n_trans,
        )

        return layer["psi"](torch.cat([z_in, z_out, trans_features(b)], -1))

    def forward(self, b, m=None):
        m = b.m if m is None else m
        h = self.enc(place_features(b, m))
        for layer in self.layers[:-1]:
            m_t = self.message(layer, h, b)

            if self.aggregate == "incidence":
                m_p = scatter_sum(
                    b.pos_w[:, None] * m_t[b.pos_t], b.pos_p, b.n_places
                ) - scatter_sum(b.pre_w[:, None] * m_t[b.pre_t], b.pre_p, b.n_places)
            else:
                m_p = torch.cat(
                    [
                        scatter_sum(
                            layer["from_in"](
                                torch.cat([m_t[b.pos_t], b.pos_w[:, None]], -1)
                            ),
                            b.pos_p,
                            b.n_places,
                        ),
                        scatter_sum(
                            layer["from_out"](
                                torch.cat([m_t[b.pre_t], b.pre_w[:, None]], -1)
                            ),
                            b.pre_p,
                            b.n_places,
                        ),
                    ],
                    -1,
                )

            h = h + layer["upd"](torch.cat([h, m_p], -1))

        flow = self.out(self.message(self.layers[-1], h, b)).squeeze(-1)

        if self.n_ids:
            flow = flow + 10.0 * self.const(b.t_id).squeeze(-1)

        if self.n_ids and self.n_env:
            flow = flow + (self.orders(b.t_id) * b.env_t).sum(-1)

        return (
            self.project(F.softplus(flow), b)
            if self.state_equation
            else F.softplus(flow)
        )
