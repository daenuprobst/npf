"""Ground truth dynamics of the synthetic benchmarks. Hidden from every model."""

import numpy as np

# system size that scales the propensities of transitions with several input tokens
VOLUME = 4.0


def propensity(net, M):
    """Stochastic mass action rates [S, T] for markings [S, P]."""
    K, k = np.exp(net.e[:, 0]), np.exp(net.a[:, 0])
    w = net.Pre[None]
    Mx = M[:, :, None]
    falling = np.where(w == 2, Mx * (Mx - 1), np.where(w == 1, Mx, 1.0))

    # a transition is enabled iff every input place holds Pre(p, t) tokens
    enabled = np.where(w > 0, Mx >= w - 1e-9, True).all(1)

    return (
        k
        * VOLUME
        * np.prod(falling / (VOLUME * K[None, :, None]) ** w, axis=1)
        * enabled
    )


def gillespie_pairs(net, M0, t_a, t_b, rng):
    """Exact stochastic simulation of S runs. Returns the markings at t_a and t_b and the firing counts between them."""
    S = M0.shape[0]
    M, t = M0.astype(float).copy(), np.zeros(S)
    MA, got_a = M.copy(), np.zeros(S, bool)
    sigma, done = np.zeros((S, net.n_trans)), np.zeros(S, bool)
    while not done.all():
        lam = propensity(net, M)
        total = lam.sum(1)
        t_next = t + rng.exponential(1.0, S) / np.maximum(total, 1e-300)

        # dead marking, nothing can fire any more
        t_next[total <= 0] = np.inf
        snap = ~got_a & (t_next > t_a)
        MA[snap], got_a[snap] = M[snap], True
        fire = ~done & (t_next <= t_b)
        done |= ~fire

        if fire.any():
            idx = np.nonzero(fire)[0]
            cum = np.cumsum(lam[idx], 1)
            j = (
                (cum < (rng.random(len(idx)) * total[idx])[:, None])
                .sum(1)
                .clip(max=net.n_trans - 1)
            )
            M[idx] += net.C[:, j].T
            counted = t_next[idx] > t_a[idx]
            np.add.at(sigma, (idx[counted], j[counted]), 1.0)
            t[idx] = t_next[idx]

    return MA, M, sigma


def flux(net, M, kind):
    """Rate laws of the fluid nets, [S, P] to [S, T]. sat is a saturating product, min is set by the scarcest input."""
    K, k = np.exp(net.e[:, 0]), np.exp(net.a[:, 0])
    mask = net.Pre > 0

    if kind == "sat":
        logu = np.log(np.maximum(M / (K + M), 1e-300))
        return k * np.exp(np.where(mask, logu[:, :, None] * net.Pre, 0.0).sum(1))

    ratio = np.where(mask, (M / K)[:, :, None] / np.where(mask, net.Pre, 1.0), np.inf)

    return k * ratio.min(1)


def simulate(net, M0, kind, n_steps, dt=0.5, substeps=50):
    """RK4 integration of dm/dt = C v(m). Returns states [S, n_steps + 1, P] and firing amounts per step [S, n_steps, T]."""
    v = lambda M: flux(net, np.maximum(M, 0.0), kind)
    h = dt / substeps
    states, fired, M = [M0], [], M0
    for _ in range(n_steps):
        F = np.zeros((M0.shape[0], net.n_trans))
        for _ in range(substeps):
            v1 = v(M)
            v2 = v(M + h / 2 * v1 @ net.C.T)
            v3 = v(M + h / 2 * v2 @ net.C.T)
            v4 = v(M + h * v3 @ net.C.T)
            dF = h / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
            M, F = np.maximum(M + dF @ net.C.T, 0.0), F + dF

        states.append(M)
        fired.append(F)

    return np.stack(states, 1), np.stack(fired, 1)
