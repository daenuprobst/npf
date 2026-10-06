"""Neural Petri Flow. Only the rate law of a transition is learned."""

from functools import partial

import torch
import torch.nn as nn

from ..layers import (
    N_MARK,
    N_PLACE_FEAT,
    apply_incidence,
    enabling_factor,
    kl_project_state_equation,
    marking_features,
    mlp,
    place_features,
    project_state_equation,
    scatter_sum,
    token_game_flow,
)


class NPF(nn.Module):
    """task is transitions, the firing counts between two markings, or next, the marking after one time unit."""

    def __init__(self, task, hidden=64, rounds=4, n_ids=0, n_env=0, gauge=None, kl_options=None):
        super().__init__()
        self.task, self.rounds = task, rounds

        # without kl_options the Gaussian step with alternating projections, with them the I projection whose first
        # iterate is that step
        self.project = (
            project_state_equation
            if kl_options is None
            else partial(kl_project_state_equation, **kl_options)
        )
        self.damping = nn.Parameter(torch.tensor(0.0))

        # a rate law must not see output places, but given both markings the expected count of a
        # transition depends on them, so the inverse task adds a bounded local correction exp(c_t)
        if task == "transitions":
            self.post_in, self.post_out = mlp(N_PLACE_FEAT, hidden, hidden), mlp(
                N_PLACE_FEAT, hidden, hidden
            )
            self.post = mlp(2 * hidden + 1, hidden, 1)
            nn.init.zeros_(self.post[-1].weight)
            nn.init.zeros_(self.post[-1].bias)

        self.arc = mlp(N_MARK + 1 + 1, hidden, 1)
        self.base = mlp(1, hidden, 1)

        # on a net of fixed transitions, a rate constant per transition and kinetic orders on read arcs from the
        # boundary, generalised mass action, both zero at the start and created only when asked for
        self.n_ids, self.n_env, self.gauge = n_ids, n_env, gauge

        if n_ids:
            self.const = nn.Embedding(n_ids, 1)
            nn.init.zeros_(self.const.weight)

        if n_ids and n_env:
            self.orders = nn.Embedding(n_ids, n_env)
            nn.init.zeros_(self.orders.weight)

    def restrict(self, d, b):
        """With gauge row a per-transition term is projected onto im C^T, where the state equation removes it."""
        if self.gauge != "row":
            return d

        G, S, Pmax, Tmax = b.pad_shape
        x = d.new_zeros(G * S * Tmax).index_copy_(0, b.pad_t, d).view(G, S, Tmax)
        row = b.C_pinv @ b.C

        return torch.einsum("gst,gut->gsu", x, row).reshape(-1)[b.pad_t]

    def rate(self, m, b):
        # local rate law, lambda_t = exp(beta(a_t) + sum over input places of g(m_p, e_p, Pre(p, t)))
        # the product form is generalised mass action, mass action itself is g = Pre log m
        x = torch.cat(
            [marking_features(m)[b.pre_p], b.e[b.pre_p], b.pre_w[:, None]], -1
        )
        log_rate = self.base(b.a) + scatter_sum(self.arc(x), b.pre_t, b.n_trans)

        # the rate constant k_t, scaled so that it learns at the pace of the network
        if self.n_ids:
            log_rate = (
                log_rate
                + self.restrict(10.0 * self.const(b.t_id).squeeze(-1), b)[:, None]
            )

        # enabling, a local positivity preserving rate law has to vanish when an input place is empty
        return torch.exp(log_rate.squeeze(-1).clamp(max=8.0)) * enabling_factor(m, b)

    def step(self, m, b, dt=1.0):
        """One round of the token game. Returns the firing amounts."""
        return token_game_flow(self.rate(m, b) * dt / self.rounds, m, b)

    def infer_transitions(self, m, b):
        """Expected firing counts between the markings A and B."""
        has_consumer = (
            b.pre_w.new_zeros(b.n_places).index_add_(
                0, b.pre_p, torch.ones_like(b.pre_w)
            )
            > 0
        )

        # the path between A and B is latent, play the token game from A and pin its end to B
        states = [m]
        for _ in range(self.rounds):
            states.append(
                states[-1] + apply_incidence(self.step(states[-1], b, b.dt_t), b)
            )

        miss = b.m_b - states[-1]
        path = [
            ((states[k] + states[k + 1]) / 2 + (k + 0.5) / self.rounds * miss).clamp(
                min=0
            )
            + 0.05 * has_consumer
            for k in range(self.rounds)
        ]
        scale = torch.ones_like(m)
        for i in range(self.rounds):
            # E sigma_t is the integral of lambda_t along the path, here a midpoint quadrature
            prior = b.dt_t * sum(self.rate(x * scale, b) for x in path) / self.rounds

            if i == 0:
                # the correction sees both markings but never the elapsed time
                x = place_features(b, m)[:, :-1]
                z_in = scatter_sum(
                    self.post_in(torch.cat([x[b.pre_p], b.pre_w[:, None]], -1)),
                    b.pre_t,
                    b.n_trans,
                )
                z_out = scatter_sum(
                    self.post_out(torch.cat([x[b.pos_p], b.pos_w[:, None]], -1)),
                    b.pos_t,
                    b.n_trans,
                )
                logit = self.post(torch.cat([z_in, z_out, b.a], -1)).squeeze(-1)

                # read arcs from the boundary, x_b enters with a kinetic order per transition and leaves C as it is
                if self.n_ids and self.n_env:
                    logit = logit + self.restrict(
                        (self.orders(b.t_id) * b.env_t).sum(-1), b
                    )

                correction = torch.exp(1.5 * torch.tanh(logit))

            prior = prior * correction

            # after this line C sigma = m_B - m_A holds exactly, the error lives in ker C only
            sigma = self.project(prior, b)

            # occupancy refinement, a place whose consumers must fire more than predicted was fuller than assumed
            needed = scatter_sum(b.pre_w * sigma[b.pre_t], b.pre_p, b.n_places)
            predicted = scatter_sum(b.pre_w * prior[b.pre_t], b.pre_p, b.n_places)
            scale = scale * (
                (needed.clamp(min=0) + 0.1) / (predicted + 0.1)
            ) ** torch.sigmoid(self.damping)

        return sigma

    def forward(self, b, m=None):
        m = b.m if m is None else m

        if self.task == "transitions":
            return self.infer_transitions(m, b)

        # tied weights, depth is time. m + C v conserves every P-invariant and stays non-negative
        for _ in range(self.rounds):
            m = m + apply_incidence(self.step(m, b), b)

        return m
