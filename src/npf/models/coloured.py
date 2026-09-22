"""Coloured tokens. Tokens carry a feature vector and transitions compute on it."""

import torch
import torch.nn as nn

from ..layers import (
    apply_incidence,
    enabling_factor,
    marking_features,
    mlp,
    scatter_sum,
    token_game_flow,
)


def transition_features(b):
    return torch.cat([b.a, torch.cos(b.angles), torch.sin(b.angles)], -1)


def mean_input_colour(colour, b):
    weight = b.pre_w / scatter_sum(b.pre_w, b.pre_t, b.n_trans)[b.pre_t]

    return scatter_sum(weight[:, None] * colour[b.pre_p], b.pre_t, b.n_trans)


class ColouredNPF(nn.Module):
    """mixing False is the ablation with the same masses and a learned colour update."""

    def __init__(self, hidden=64, rounds=4, mixing=True):
        super().__init__()
        self.rounds, self.mixing = rounds, mixing
        self.arc, self.base = mlp(3 + 1 + 1, hidden, 1), mlp(5, hidden, 1)

        # learned guard of the rate law on the mean input colour, and learned arc expression for the emitted colour
        self.guard = mlp(2 + 5, hidden, 1)
        self.emit = mlp(2 + 5, hidden, 2)
        self.free = mlp(2 + 2 + 2, hidden, 2)

    def step(self, m, c, b):
        x = torch.cat(
            [marking_features(m)[b.pre_p], b.e[b.pre_p], b.pre_w[:, None]], -1
        )
        c_in, feats = mean_input_colour(c, b), transition_features(b)
        log_rate = (
            self.base(feats)
            + scatter_sum(self.arc(x), b.pre_t, b.n_trans)
            + self.guard(torch.cat([c_in, feats], -1))
        )
        d = (
            torch.exp(log_rate.squeeze(-1).clamp(max=8.0))
            * enabling_factor(m, b)
            / self.rounds
        )
        v = token_game_flow(d, m, b)
        out = scatter_sum(b.pre_w * v[b.pre_t], b.pre_p, b.n_places)
        emitted = self.emit(torch.cat([c_in, feats], -1))
        arriving = scatter_sum(
            (b.pos_w * v[b.pos_t])[:, None] * emitted[b.pos_t], b.pos_p, b.n_places
        )
        m_next = m + apply_incidence(v, b)

        if self.mixing:
            # what stays keeps its colour and what arrives brings its own, so c+ is convex and equals c without inflow
            c_next = ((m - out)[:, None] * c + arriving) / m_next.clamp(min=1e-6)[
                :, None
            ]
        else:
            c_next = c + self.free(
                torch.cat([c, arriving, torch.stack([out, m_next - m + out], -1)], -1)
            )

        return m_next, c_next

    def forward(self, b, m=None, c=None):
        m, c = (b.m, b.colour) if m is None else (m, c)
        for _ in range(self.rounds):
            m, c = self.step(m, c, b)

        return m, c


class ColouredPGNN(nn.Module):
    """Place and transition message passing on mass and colour features with a free readout."""

    def __init__(self, hidden=64, rounds=4):
        super().__init__()
        self.enc = mlp(3 + 1 + 2, hidden, hidden)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "phi_in": mlp(hidden + 1, hidden, hidden),
                    "phi_out": mlp(hidden + 1, hidden, hidden),
                    "psi": mlp(2 * hidden + 5, hidden, hidden),
                    "from_in": mlp(hidden + 1, hidden, hidden),
                    "from_out": mlp(hidden + 1, hidden, hidden),
                    "upd": mlp(3 * hidden, hidden, hidden),
                }
            )
            for _ in range(rounds)
        )
        self.out = mlp(hidden, hidden, 3)

    def forward(self, b, m=None, c=None):
        m, c = (b.m, b.colour) if m is None else (m, c)
        h, feats = self.enc(
            torch.cat([marking_features(m), b.e, c], -1)
        ), transition_features(b)
        for layer in self.layers:
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
            m_t = layer["psi"](torch.cat([z_in, z_out, feats], -1))
            h = h + layer["upd"](
                torch.cat(
                    [
                        h,
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
            )

        delta = self.out(h)

        return m + delta[:, 0], c + delta[:, 1:]
