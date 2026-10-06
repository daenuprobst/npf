"""Parameter free Petri net layers and the small helpers shared by all models."""

import torch
import torch.nn as nn

N_MARK = 3

# features of m, features of m_b, m_b - m, place attribute, elapsed time
N_PLACE_FEAT = 2 * N_MARK + 1 + 1 + 1

# transition attribute, elapsed time
N_TRANS_FEAT = 1 + 1


def mlp(n_in, n_hidden, n_out):
    return nn.Sequential(
        nn.Linear(n_in, n_hidden),
        nn.SiLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.SiLU(),
        nn.Linear(n_hidden, n_out),
    )


def scatter_sum(src, index, n):
    return src.new_zeros((n,) + src.shape[1:]).index_add_(0, index, src)


def marking_features(m):
    # mass action rates are log linear in the markings, so log m is a feature of every model
    m = m.clamp(min=0)

    return torch.stack([m / 4.0, torch.log1p(m), torch.log(m + 0.01)], -1)


def place_features(b, m):
    return torch.cat(
        [
            marking_features(m),
            marking_features(b.m_b),
            ((b.m_b - m) / 4.0)[:, None],
            b.e,
            b.dt_p[:, None],
        ],
        -1,
    )


def trans_features(b):
    return torch.cat([b.a, b.dt_t[:, None]], -1)


def apply_incidence(v, b):
    """C v on the batched net, tokens produced minus tokens consumed by the firing amounts v."""
    produced = scatter_sum(b.pos_w * v[b.pos_t], b.pos_p, b.n_places)
    consumed = scatter_sum(b.pre_w * v[b.pre_t], b.pre_p, b.n_places)

    return produced - consumed


def enabling_factor(m, b):
    """min over input places of min(1, m_p / Pre(p, t)). Zero iff an input place is empty."""
    tokens = (m[b.pre_p] / b.pre_w).clamp(0.0, 1.0)

    return tokens.new_ones(b.n_trans).scatter_reduce(0, b.pre_t, tokens, reduce="amin")


def token_game_flow(d, m, b):
    """Firing amounts v for demands d >= 0 at marking m.

    A place with relative demand x = sum_t Pre(p, t) d_t / m_p serves the fraction (1 - exp(-x)) / x, so it hands
    out less than m_p tokens, and a transition fires as far as its scarcest input allows. Hence m + C v >= 0 for
    all d and x^T C v = 0 for every P-invariant x.
    """
    load = scatter_sum(b.pre_w * d[b.pre_t], b.pre_p, b.n_places) / m.clamp(min=1e-30)
    served = torch.where(
        load < 1e-4, 1.0 - load / 2, -torch.expm1(-load) / load.clamp(min=1e-4)
    )

    return d * served.new_ones(b.n_trans).scatter_reduce(
        0, b.pre_t, served[b.pre_p], reduce="amin"
    )


def kl_project_state_equation(sigma, b, n_iter=30, floor=0.05, low=-30.0, high=8.0):
    """Firing counts that satisfy the state equation, as the I projection of sigma onto {s >= 0, C s = dM}.

    The minimiser of D(s || sigma) over the fibre is s = sigma * exp(C^T nu), positive for every nu, with nu the
    minimiser of the strictly convex dual sum_t sigma_t exp((C^T nu)_t) - nu^T dM, whose Hessian is C diag(s) C^T.
    This is Birch's theorem on the transitions and the large count limit of Poisson counts conditioned on the state
    equation. Newton from nu = 0 gives the Gaussian step as its first iterate.
    """
    G, S, Pmax, Tmax = b.pad_shape
    dm = (
        sigma.new_zeros(G * S * Pmax)
        .index_copy_(0, b.pad_p, b.m_b - b.m)
        .view(G, S, Pmax)
        .double()
    )
    prior = (
        sigma.new_zeros(G * S * Tmax)
        .index_copy_(0, b.pad_t, sigma.clamp(min=0) + floor)
        .view(G, S, Tmax)
        .double()
    )
    C = b.C.double()
    eye = torch.eye(Pmax, dtype=torch.float64, device=sigma.device)

    def counts(nu):
        # a count forced to zero needs a large negative exponent, the floor keeps an untrained prior off zero
        return prior * torch.einsum("gsp,gpt->gst", nu, C).clamp(low, high).exp()

    def dual(nu):
        return counts(nu).sum(-1) - (nu * dm).sum(-1)

    def newton(nu):
        s = counts(nu)
        gradient = torch.einsum("gst,gpt->gsp", s, C) - dm
        laplacian = (C[:, None] * s[:, :, None, :]) @ C.transpose(-1, -2)[:, None]

        # P-invariants are null directions of the Hessian and gauges of the dual, so a ridge picks the minimum norm dual
        scale = (
            laplacian.diagonal(dim1=-2, dim2=-1)
            .mean(-1)
            .clamp(min=1.0)[..., None, None]
        )

        return (
            -torch.linalg.solve(laplacian + 1e-6 * scale * eye, gradient),
            gradient,
        )

    def step_size(nu, direction, gradient):
        # the dual is convex, so backtracking on it makes every step a descent step and Newton converges
        value, slope = dual(nu), (gradient * direction).sum(-1)
        t = torch.ones_like(value)
        for _ in range(12):
            worse = dual(nu + t[..., None] * direction) > value + 1e-4 * t * slope
            if not worse.any():
                break

            t = torch.where(worse, t * 0.5, t)

        return t

    nu = torch.zeros(G, S, Pmax, dtype=torch.float64, device=sigma.device)

    with torch.no_grad():
        for _ in range(n_iter):
            direction, gradient = newton(nu)
            nu = nu + step_size(nu, direction, gradient)[..., None] * direction

    # one more step at the solution carries the gradient, the implicit function theorem in Newton form
    direction, gradient = newton(nu)

    with torch.no_grad():
        t = step_size(nu, direction, gradient)

    return counts(nu + t[..., None] * direction).to(sigma.dtype).reshape(-1)[b.pad_t]


def project_state_equation(sigma, b, weighted=True, n_iter=6):
    """Firing counts close to sigma that satisfy the state equation m_B = m_A + C sigma.

    With W = diag(sigma) the Poisson posterior mean given C sigma = dM is sigma + W C^T (C W C^T)^+ (dM - C sigma),
    one solve with C W C^T. The projector C^+ then removes the residual of the ridge, alternating with sigma >= 0
    and ending on the state equation. Only the component in im C^T is fixed by the data, the part in ker C stays
    with the estimator.
    """
    G, S, Pmax, Tmax = b.pad_shape
    dm = (
        sigma.new_zeros(G * S * Pmax)
        .index_copy_(0, b.pad_p, b.m_b - b.m)
        .view(G, S, Pmax)
    )
    s = sigma.new_zeros(G * S * Tmax).index_copy_(0, b.pad_t, sigma).view(G, S, Tmax)

    if weighted:
        C = b.C.double()[:, None]
        w = s.double() + 1e-3
        laplacian = (C * w[:, :, None, :]) @ C.transpose(-1, -2) + 1e-6 * torch.eye(
            Pmax, dtype=torch.float64, device=s.device
        )
        residual = dm.double() - torch.einsum("gst,gpt->gsp", s.double(), b.C.double())
        y = torch.linalg.solve(laplacian, residual)
        s = s + (w * torch.einsum("gsp,gpt->gst", y, b.C.double())).to(s.dtype)

    for i in range(n_iter):
        if i:
            s = s.clamp(min=0)

        residual = dm - torch.einsum("gst,gpt->gsp", s, b.C)
        s = s + torch.einsum("gsp,gtp->gst", residual, b.C_pinv)

    return s.reshape(-1)[b.pad_t]
