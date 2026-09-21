"""Neural Petri Flow. Only the rate law of a transition is learned."""
import torch
import torch.nn as nn

from ..layers import (N_MARK, N_PLACE_FEAT, apply_incidence, enabling_factor, marking_features, mlp, place_features,
                      kl_project_state_equation, project_state_equation, scatter_sum, token_game_flow)


class NPF(nn.Module):
    def __init__(self, task, hidden=64, rounds=4, product_form=True, noisy_states=False, refine=True, posterior=True,
                 divergence="gauss"):
        super().__init__()
        self.task, self.rounds, self.product_form, self.refine = task, rounds, product_form, refine

        # gauss, the Poisson posterior mean under a Gaussian approximation followed by alternating projections.
        # kl, the I projection, which is positive and idempotent and of which the gaussian step is the first iterate
        self.project = kl_project_state_equation if divergence == "kl" else project_state_equation
        self.log_obs_var = nn.Parameter(torch.tensor(-2.0)) if noisy_states else None
        self.damping = nn.Parameter(torch.tensor(0.0))

        # a rate law must not see output places, but given both markings the expected count of a
        # transition depends on them, so the inverse task adds a bounded local correction exp(c_t)
        self.posterior = posterior and task == "transitions"

        if self.posterior:
            self.post_in, self.post_out = mlp(N_PLACE_FEAT, hidden, hidden), mlp(N_PLACE_FEAT, hidden, hidden)
            self.post = mlp(2 * hidden + 1, hidden, 1)
            nn.init.zeros_(self.post[-1].weight)
            nn.init.zeros_(self.post[-1].bias)

        self.arc = mlp(N_MARK + 1 + 1, hidden, 1 if product_form else hidden)
        self.base = mlp(1, hidden, 1) if product_form else mlp(hidden + 1, hidden, 1)

    def rate(self, m, b):
        # local rate law, lambda_t = exp(beta(a_t) + sum over input places of g(m_p, e_p, Pre(p, t)))
        # the product form is generalised mass action, mass action itself is g = Pre log m
        x = torch.cat([marking_features(m)[b.pre_p], b.e[b.pre_p], b.pre_w[:, None]], -1)
        pooled = scatter_sum(self.arc(x), b.pre_t, b.n_trans)
        log_rate = self.base(b.a) + pooled if self.product_form else self.base(torch.cat([pooled, b.a], -1))

        # enabling, a local positivity preserving rate law has to vanish when an input place is empty
        return torch.exp(log_rate.squeeze(-1).clamp(max=8.0)) * enabling_factor(m, b)

    def step(self, m, b, dt=1.0):
        """One round of the token game. Returns the firing amounts."""
        return token_game_flow(self.rate(m, b) * dt / self.rounds, m, b)

    def infer_transitions(self, m, b):
        """Expected firing counts between the markings A and B."""
        obs_var = None if self.log_obs_var is None else self.log_obs_var.exp()
        has_consumer = b.pre_w.new_zeros(b.n_places).index_add_(0, b.pre_p, torch.ones_like(b.pre_w)) > 0

        # the path between A and B is latent, play the token game from A and pin its end to B
        states = [m]
        for _ in range(self.rounds):
            states.append(states[-1] + apply_incidence(self.step(states[-1], b, b.dt_t), b))

        miss = b.m_b - states[-1]
        path = [((states[k] + states[k + 1]) / 2 + (k + 0.5) / self.rounds * miss).clamp(min=0) + 0.05 * has_consumer
                for k in range(self.rounds)]
        scale = torch.ones_like(m)
        for i in range(self.rounds if self.refine else 1):
            # E sigma_t is the integral of lambda_t along the path, here a midpoint quadrature
            prior = b.dt_t * sum(self.rate(x * scale, b) for x in path) / self.rounds

            if self.posterior:
                if i == 0:
                    # the correction sees both markings but never the elapsed time
                    x = place_features(b, m)[:, :-1]
                    z_in = scatter_sum(self.post_in(torch.cat([x[b.pre_p], b.pre_w[:, None]], -1)), b.pre_t, b.n_trans)
                    z_out = scatter_sum(self.post_out(torch.cat([x[b.pos_p], b.pos_w[:, None]], -1)), b.pos_t, b.n_trans)
                    correction = torch.exp(1.5 * torch.tanh(self.post(torch.cat([z_in, z_out, b.a], -1)).squeeze(-1)))

                prior = prior * correction

            # after this line C sigma = m_B - m_A holds exactly, the error lives in ker C only
            sigma = self.project(prior, b, obs_var=obs_var)

            # occupancy refinement, a place whose consumers must fire more than predicted was fuller than assumed
            needed = scatter_sum(b.pre_w * sigma[b.pre_t], b.pre_p, b.n_places)
            predicted = scatter_sum(b.pre_w * prior[b.pre_t], b.pre_p, b.n_places)
            scale = scale * ((needed.clamp(min=0) + 0.1) / (predicted + 0.1)) ** torch.sigmoid(self.damping)

            # the refinement feeds the projected counts back into the markings. the I projection moves counts
            # multiplicatively, so there the loop is bounded to keep it from driving the rates away with it
            if self.project is kl_project_state_equation:
                scale = scale.clamp(0.05, 20.0)

        return sigma

    def forward(self, b, m=None, return_firing=False):
        m = b.m if m is None else m

        if self.task == "transitions":
            return self.infer_transitions(m, b)

        sigma = 0.0

        # tied weights, depth is time. m + C v conserves every P-invariant and stays non-negative
        for _ in range(self.rounds):
            v = self.step(m, b)
            m, sigma = m + apply_incidence(v, b), sigma + v

        return (m, sigma) if return_firing else m
