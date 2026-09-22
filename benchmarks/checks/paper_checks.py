"""Numerical checks of the propositions as they are stated in the paper (paper/sections, numbered as in the paper).
Wherever possible the check runs against the implementation that produced the results, with random untrained weights,
because a guarantee for all weights must also hold at initialisation.

    CUDA_VISIBLE_DEVICES="" uv run python -m benchmarks.checks.paper_checks

Note that this is an experiment on whether Qwen 3.6 27B can translate propositions into code.
Various corrections have been added.
"""

import numpy as np
import torch
from scipy.linalg import null_space

from npf import batching, chem, datasets, layers, models, nets, simulate
from npf.models.equilibrium import ThermoNPF

from ..synthetic import equilibrium as thermo

rng = np.random.default_rng(11)
torch.manual_seed(11)
FAILED = []


def ok(cond):
    """Text of one check. A failure is remembered, the script then ends with a non-zero status."""
    if not cond:
        FAILED.append(1)

    return "ok" if cond else "FAILED"


def random_incidence(P=9, need_invariant=False):
    while True:
        try:
            net = nets.random_net(
                rng, P, n_base=max(2, int(round(0.6 * P))), max_arity=2
            )

        # the sampled transitions did not cover every place
        except AssertionError:
            continue

        X = null_space(net.C.T)
        if X.shape[1] or not need_invariant:
            return net, net.C, X


# Proposition 1
print("P1  conservation <=> firing form")
net, C, X = random_incidence(need_invariant=True)

# an update of the firing form
F_in = C @ rng.normal(size=C.shape[1])

# the same update plus a component outside im C
F_out = F_in + X @ rng.normal(size=X.shape[1])
v = np.linalg.pinv(C) @ F_in
print(
    f"    dim ker C^T = {X.shape[1]};  |X^T F| for F in im C: {np.abs(X.T @ F_in).max():.1e};  for F not in im C: {np.abs(X.T @ F_out).max():.2f}"
)
print(
    f"    F in im C is reproduced by v = C^+ F: |C v - F| = {np.abs(C @ v - F_in).max():.1e}   {ok(np.abs(C @ v - F_in).max() < 1e-10)}"
)

# Proposition 5
print(
    "P5  token-game layer with ARBITRARY demands d >= 0 (numpy transcription of Eq. 7)"
)


def layer(m, d, Pre, C):
    demand = Pre @ d

    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(m > 0, demand / np.where(m > 0, m, 1.0), np.inf)
        phi = np.where(
            x == 0,
            1.0,
            np.where(np.isinf(x), 0.0, -np.expm1(-x) / np.where(x > 0, x, 1.0)),
        )

    phi = np.where((m == 0) & (demand == 0), 1.0, phi)
    served = np.where(Pre > 0, phi[:, None], np.inf).min(0)

    # a transition without input places
    served = np.where(np.isinf(served), 1.0, served)

    return m + C @ (d * served)


worst, kept_positive, drift = np.inf, True, 0.0
for _ in range(3000):
    net, C, X = random_incidence(P=int(rng.integers(4, 12)))

    # many empty places
    m = rng.integers(0, 4, net.n_places) * rng.uniform(0.0, 3.0, net.n_places)
    d = rng.exponential(rng.choice([0.1, 1.0, 30.0]), net.n_trans) * (
        rng.random(net.n_trans) > 0.2
    )
    m_next = layer(m, d, net.Pre, C)

    with np.errstate(divide="ignore", invalid="ignore"):
        # the proof shows m+ >= m exp(-x)
        bound = np.where(
            m > 0, m * np.exp(-(net.Pre @ d) / np.where(m > 0, m, 1.0)), 0.0
        )

    worst = min(worst, m_next.min())
    kept_positive &= bool((m_next >= bound - 1e-12 * np.maximum(1.0, m)).all())
    drift = max(
        drift, np.abs(X.T @ (m_next - m)).max(initial=0.0) / max(1.0, np.abs(m).max())
    )

print(
    f"    3000 random nets/markings/demands: min m+ = {worst:.2e}, m+ >= m exp(-x) (hence > 0 where m > 0): {kept_positive}, "
    f"invariant drift {drift:.1e}   {ok(worst >= -1e-9 and kept_positive and drift < 1e-9)}"
)

splits = {k: datasets.make_pairs(s, 12, (6, 12), 2, 16) for k, s in (("a", 21),)}
groups = splits["a"]
batch = batching.collate(groups, [np.arange(16)] * len(groups), "cpu")
npf_next = models.NPF("next")

with torch.no_grad():
    m = batch.m.clone()
    lowest = np.inf

    # the implementation, untrained weights, rolled out
    for _ in range(25):
        m = npf_next(batch, m)
        lowest = min(lowest, float(m.min()))

print(
    f"    implementation (models.NPF, random weights, 25 rollout steps): min marking {lowest:.2e}   {ok(lowest >= -1e-6)}"
)

# Proposition 2
print("P2  a rate law that does not vanish at an empty input place leaves the orthant")

# p0 -> t -> p1 with v = 0.3 regardless of m_0
m, h = np.array([0.0, 1.0]), 1e-3
after = m[0] - h * 0.3
print(f"    after one Euler step of size {h}: m_0 = {after:.1e} < 0   {ok(after < 0)}")

# Appendix E.3, the bridge
print(
    "E.3  given both markings the run is a bridge, its intensity is the Doob transform of the rate law"
)

# reversible pair A <-> B with one token and unit rates, conditioned on m_A = m_B = (1, 0) after time 1
# P_AA(s) = (1 + exp(-2s)) / 2 and P_BA(s) = (1 - exp(-2s)) / 2 are the transition probabilities of the chain
grid = (np.arange(20000) + 0.5) / 20000
p_aa, p_ba = (1 + np.exp(-2 * grid)) / 2, (1 - np.exp(-2 * grid)) / 2
end = (1 + np.exp(-2.0)) / 2

# occupation of A on the bridge is P_AA(s) P_AA(1 - s) / P_AA(1), the Doob ratio for the forward transition is
# P_BA(1 - s) / P_AA(1 - s)
doob = float(np.mean(p_aa * p_ba[::-1]) / end)
naive = float(np.mean(p_aa * p_aa[::-1]) / end)
counts, kept = 0, 0

for _ in range(60000):
    clock, state, fired = rng.exponential(), 0, 0
    while clock < 1.0:
        fired += state == 0
        state = 1 - state
        clock += rng.exponential()

    if state == 0:
        kept += 1
        counts += fired

print(
    f"    E[sigma | m_A, m_B] = {counts / kept:.3f} (Monte Carlo), Doob transform {doob:.3f}, rate law integrated along the bridge {naive:.3f}"
    f"   {ok(abs(counts / kept - doob) < 0.02 and naive - doob > 0.3)}"
)

# Proposition 7
print("P7  projection onto the state equation")
worst_i = worst_ii = worst_iii = 0.0
for _ in range(500):
    net, C, X = random_incidence(P=int(rng.integers(4, 12)))
    T = C.shape[1]
    sigma_star = rng.poisson(2.0, T).astype(float)
    b = C @ sigma_star
    guess = rng.normal(2.0, 2.0, T)
    Cp = np.linalg.pinv(C)
    proj = guess + Cp @ (b - C @ guess)
    worst_i = max(
        worst_i,
        np.abs(C @ proj - b).max(),
        np.abs((proj - sigma_star) - (np.eye(T) - Cp @ C) @ (guess - sigma_star)).max(),
        np.linalg.norm(proj - sigma_star) - np.linalg.norm(guess - sigma_star),
    )
    clipped = np.maximum(guess, 0)
    worst_ii = max(
        worst_ii,
        np.linalg.norm(clipped - sigma_star) - np.linalg.norm(guess - sigma_star),
    )
    w = rng.uniform(0.05, 3.0, T)
    W = np.diag(w)
    weighted = guess + W @ C.T @ np.linalg.pinv(C @ W @ C.T) @ (b - C @ guess)

    # brute force, minimise over sigma = particular + Z z
    Z = null_space(C)
    particular = Cp @ b
    z = (
        np.linalg.solve(
            Z.T @ np.diag(1 / w) @ Z, Z.T @ np.diag(1 / w) @ (guess - particular)
        )
        if Z.shape[1]
        else np.zeros(0)
    )
    brute = particular + Z @ z
    wnorm = lambda u: np.sqrt(u @ (u / w))
    worst_iii = max(
        worst_iii,
        np.abs(weighted - brute).max(),
        np.abs(C @ weighted - b).max(),
        wnorm(weighted - sigma_star) - wnorm(guess - sigma_star),
    )

print(
    f"    (i)   feasibility, error identity, monotonicity: worst violation {worst_i:.1e}   {ok(worst_i < 1e-8)}"
)
print(
    f"    (ii)  clipping at zero never moves away from sigma*: worst {worst_ii:.1e}   {ok(worst_ii < 1e-12)}"
)
print(
    f"    (iii) weighted formula = brute-force weighted least squares, monotone in the W^-1 norm: worst {worst_iii:.1e}   {ok(worst_iii < 1e-7)}"
)

with torch.no_grad():
    guess = torch.rand(batch.n_trans) * 3
    out = layers.project_state_equation(guess, batch)
    residual = (batch.m_b - batch.m - layers.apply_incidence(out, batch)).abs().max()

print(
    f"    implementation (project_state_equation on simulated pairs): |m_B - m_A - C sigma^| = {residual:.1e}, min sigma^ = {out.min():.3f}   {ok(residual < 1e-3)}"
)

# Proposition 6
print("P6  locality lower bound on the path net")
L, k, delta = 48, 24, 5.0
net = nets.chain_net(rng, L + 1)
M0 = rng.integers(0, 9, size=(8, L + 1)).astype(float)
MA, MB, sigma = simulate.gillespie_pairs(net, M0, np.full(8, 0.2), np.full(8, 0.9), rng)
formula = -np.cumsum(MB - MA, 1)[:, :-1]
print(
    f"    sigma*_k = -sum_(j<k) (m_B - m_A)_j on simulated runs: max error {np.abs(formula - sigma).max():.1e}   {ok(np.abs(formula - sigma).max() < 1e-9)}"
)
MA2, MB2 = MA.copy(), MB.copy()
MA2[:, 0] += delta
MB2[:, -1] += delta

# the second input is valid, push delta tokens through the chain, then replay the counts of the first
m = MA2.copy()
for t in range(L):
    assert (m[:, t] >= delta).all()
    m[:, t] -= delta
    m[:, t + 1] += delta

print(
    f"    second input reachable: after moving delta through the chain the marking is m_A + delta e_pL >= m_A: {ok(np.allclose(m, MA + delta * np.eye(L + 1)[-1]))}"
)
pgnn = models.PGNN("transitions", rounds=4)

with torch.no_grad():
    dt = np.full(8, 0.7)
    out1 = pgnn(
        batching.collate(
            [datasets.Group(net, MA, MB, dt, sigma)], [np.arange(8)], "cpu"
        )
    ).view(8, L)
    out2 = pgnn(
        batching.collate(
            [datasets.Group(net, MA2, MB2, dt, sigma + delta)], [np.arange(8)], "cpu"
        )
    ).view(8, L)

same = float((out1[:, k] - out2[:, k]).abs().max())
print(
    f"    PGNN (4 rounds) at the middle transition: outputs differ by {same:.1e} on the two inputs, targets differ by {delta}   {ok(same < 1e-6)}"
)

# Proposition 8
print("P8  equilibrium layer")
net = nets.random_net(rng, 10, n_base=6, max_arity=2, p_reverse=1.0)
net.e = rng.uniform(-0.7, 0.7, size=(10, 1))
E = thermo.energy(net.e[:, 0])
m_ref = np.exp(-E)
m0 = rng.uniform(0.3, 3.0, size=(1, 10))
m_star = thermo.equilibrium(net, m0)[0]
mu = np.log(m_star / m_ref)
Fen = lambda m: float((m * (np.log(m / m_ref) - 1)).sum())
print(
    f"    |C^T mu(m*)| = {np.abs(net.C.T @ mu).max():.1e}, |X^T (m* - m0)| = {np.abs(net.X.T @ (m_star - m0[0])).max():.1e}, min m* = {m_star.min():.3f}"
)
better = 0

# other positive points of the same class
for _ in range(2000):
    z = rng.normal(size=net.n_trans) * rng.choice([1e-3, 1e-2, 0.1])
    other = m_star + net.C @ z
    if other.min() > 0:
        better += Fen(other) < Fen(m_star) - 1e-12

print(
    f"    points of the class with lower free energy than m*: {better} of 2000   {ok(better == 0)}"
)

# every reversible pair appears as two transitions, flux form per transition
forward = net.Pre[:, : net.n_trans]
kappa_w = rng.normal(size=(net.n_trans, 10)) * 0.3


def rhs(m):
    mu = np.log(np.maximum(m, 1e-300) / m_ref)

    # arbitrary positive, marking-dependent
    kappa = np.exp(np.tanh(kappa_w @ m))

    return net.C @ (kappa * (np.exp(net.Pre.T @ mu) - np.exp(net.Pos.T @ mu)))


m, Fs = m0[0].copy(), []
for _ in range(60000):
    m = m + 2e-4 * rhs(m)
    Fs.append(Fen(m))

print(
    f"    dynamics with a random positive kappa(m): largest increase of F {max(np.diff(Fs)):.1e}, |m(T) - m*| = {np.abs(m - m_star).max():.1e} "
    f"(convergence is NOT part of the proposition)   {ok(max(np.diff(Fs)) < 1e-9)}"
)

layer_model = ThermoNPF()
group = datasets.Group(net, m0, y=m_star[None])
b = batching.collate([group], [np.arange(1)], "cpu")
m0_t = b.m.clone().requires_grad_(True)
b.m = m0_t
out = layer_model(b)
weights = torch.randn_like(out)
(out * weights).sum().backward()
grad = m0_t.grad.clone()
fd = torch.zeros_like(grad)

with torch.no_grad():
    for p in range(10):
        for sign in (1, -1):
            b.m = m0_t.detach().clone()
            b.m[p] += sign * 1e-3
            fd[p] += sign * (layer_model(b) * weights).sum() / 2e-3

print(
    f"    d m*/d m0 by one Newton step vs finite differences: max abs diff {float((grad - fd).abs().max()):.1e}   {ok(float((grad - fd).abs().max()) < 1e-3)}"
)

with torch.no_grad():
    b.m = m0_t.detach()
    out = layer_model(b).double().numpy()

print(
    f"    layer output (random energies): invariant drift {np.abs(net.X.T @ (out - m0[0])).max():.1e}, min {out.min():.3f}   {ok(out.min() > 0)}"
)

# Proposition 3
print(
    "P3  state-equation readout: cancellation and invariance (random graphs, random local psi)"
)
R = 3


def random_tree_graph(n):
    x, adj = rng.integers(0, 4, n), np.zeros((n, n), int)
    for i in range(1, n):
        adj[i, rng.integers(0, i)] = adj[rng.integers(0, i), i] = 0

    for i in range(1, n):
        j = int(rng.integers(max(0, i - 3), i))
        t = int(rng.integers(1, 4))
        adj[i, j] = adj[j, i] = t

    return x, adj


# a random message-passing psi with radius R
W_in, W_msg = rng.normal(size=(4, 8)), rng.normal(size=(4, 8, 8)) / 3


def psi(x, adj):
    h = W_in[x]
    for _ in range(R):
        h = np.tanh(h + sum((adj == t).astype(float) @ h @ W_msg[t] for t in (1, 2, 3)))

    return h


def distances(adj, sources):
    n = len(adj)
    dist = np.full(n, np.inf)
    dist[list(sources)] = 0
    frontier = list(sources)
    while frontier:
        nxt = []
        for u in frontier:
            for w in np.nonzero(adj[u])[0]:
                if dist[w] == np.inf:
                    dist[w] = dist[u] + 1
                    nxt.append(w)

        frontier = nxt

    return dist


worst_cancel = worst_inv = 0.0
tested = 0

for _ in range(300):
    nA = int(rng.integers(30, 60))
    xA, adjA = random_tree_graph(nA)
    leaving = set(rng.choice(nA, int(rng.integers(0, 3)), replace=False).tolist())

    # product atom k sits on precursor atom keep[k]
    keep = np.array([i for i in range(nA) if i not in leaving])
    xB, adjB = xA[keep].copy(), adjA[np.ix_(keep, keep)].copy()
    changed = set()

    # re-type / break / form a bond, change an attribute
    for _ in range(int(rng.integers(1, 3))):
        i, j = rng.choice(len(keep), 2, replace=False)
        adjB[i, j] = adjB[j, i] = (adjB[i, j] + int(rng.integers(1, 4))) % 4
        changed |= {int(i), int(j)}

    i = int(rng.integers(len(keep)))
    xB[i] = (xB[i] + 1) % 4
    changed.add(i)
    D = set(changed) | {
        k for k in range(len(keep)) if any(adjA[keep[k], u] for u in leaving)
    }
    dist = distances(adjB, D)
    D_R = dist <= R
    full = psi(xB, adjB).sum(0) - psi(xA, adjA).sum(0)
    pB, pA = psi(xB, adjB), psi(xA, adjA)
    restricted = (pB[D_R] - pA[keep[D_R]]).sum(0) - pA[sorted(leaving)].sum(0)
    worst_cancel = max(worst_cancel, np.abs(full - restricted).max())
    far = np.nonzero(dist > 2 * R)[0]
    far = far[np.isfinite(dist[far])] if len(far) else far

    if len(far):
        i0 = int(rng.choice(far))
        tested += 1

        def attach(x, adj, at):
            n = len(x)
            x2 = np.append(x, [1, 2])
            a2 = np.zeros((n + 2, n + 2), int)
            a2[:n, :n] = adj
            a2[at, n] = a2[n, at] = 1
            a2[n, n + 1] = a2[n + 1, n] = 2

            return x2, a2

        xB2, adjB2 = attach(xB, adjB, i0)
        xA2, adjA2 = attach(xA, adjA, keep[i0])
        full2 = psi(xB2, adjB2).sum(0) - psi(xA2, adjA2).sum(0)
        worst_inv = max(worst_inv, np.abs(full2 - full).max())

print(
    f"    r = sum over D_R and unmatched atoms only: max abs diff {worst_cancel:.1e}   {ok(worst_cancel < 1e-9)}"
)
print(
    f"    same subgraph attached > 2R from D on both sides ({tested} cases): max change of r {worst_inv:.1e}   {ok(worst_inv < 1e-9 and tested > 50)}"
)

# Proposition 4
print(
    "P4  token game on the valence net: every visited marking is valid, for random weights"
)
reactions = [r for r in chem.load()["reactions"] if r["target"] is not None][:256]
game = chem.TokenGame()
order, violations, steps = chem.BOND_ORDER, 0, 0

with torch.no_grad():
    for start in range(0, len(reactions), 32):
        rs = reactions[start : start + 32]
        b = chem.collate(rs, "cpu")
        cur, fired = b["ba"].clone(), torch.zeros_like(b["ba"], dtype=torch.bool)
        n = cur.shape[1]

        # the greedy token game of TokenGame.forward, with a check after every firing
        for _ in range(game.max_steps):
            logits, enabled, stop = game.rates(b, cur, fired)
            best, where = logits.masked_fill(~enabled, -1e4).flatten(1).max(1)
            fire = best > stop
            if not fire.any():
                break

            i, j, k = (
                where // (n * chem.N_BOND),
                (where // chem.N_BOND) % n,
                where % chem.N_BOND,
            )
            rows = torch.nonzero(fire).squeeze(1)
            cur[rows, i[rows], j[rows]] = k[rows]
            cur[rows, j[rows], i[rows]] = k[rows]
            fired[rows, i[rows], j[rows]] = True
            fired[rows, j[rows], i[rows]] = True
            steps += len(rows)

            # independent bookkeeping in numpy
            for row, r in enumerate(rs):
                a, size = r["a"], len(r["a"]["x"])
                before, after = chem.dense_bonds(a), cur[row, :size, :size].numpy()
                slack = a["h"] - (order[after] - order[before]).sum(1)
                capacity = (
                    np.array([chem.EXTRA_CAPACITY.get(int(e), 0) for e in a["element"]])
                    + np.maximum(-a["q"], 0)
                    + 0.5
                )
                violations += int((slack < -capacity - 1e-9).any())

print(
    f"    {steps} firings with untrained weights on {len(reactions)} reactions: markings with a slack place below the "
    f"half-token tolerance: {violations}   {ok(violations == 0 and steps > 100)}"
)


# the I-projection of app:kl
print(
    "KL  the I-projection: positive, idempotent, Pythagorean, and invariant under a gauge change of the prior"
)


def i_projection(C, prior, b, iters=200):
    """min D(sigma || prior) over sigma >= 0 with C sigma = b, by Newton on the dual with a ridge along ker C^T."""
    eta = np.zeros(C.shape[0])
    for _ in range(iters):
        sigma = prior * np.exp(C.T @ eta)
        grad = C @ sigma - b
        hess = C @ np.diag(sigma) @ C.T + 1e-10 * np.eye(C.shape[0])
        eta = eta - np.linalg.solve(hess, grad)

    return prior * np.exp(C.T @ eta)


divergence = lambda x, y: float(np.sum(x * np.log(x / y) - x + y))
rng = np.random.default_rng(0)
worst_feas, worst_pos, worst_idem, worst_pyth, worst_gauge = 0.0, 1.0, 0.0, 0.0, 0.0

for _ in range(200):
    n_p, n_t = rng.integers(3, 6), rng.integers(4, 8)

    # every transition sits on a non-negative T-invariant, so a strictly positive feasible point exists
    C = rng.integers(-2, 3, (n_p, n_t)).astype(float)
    C = np.concatenate([C, -C], 1)
    star = rng.random(2 * n_t) + 0.05
    b = C @ star
    prior = rng.random(2 * n_t) + 0.05
    hat = i_projection(C, prior, b)
    worst_feas = max(worst_feas, float(np.abs(C @ hat - b).max()))
    worst_pos = min(worst_pos, float(hat.min()))
    worst_idem = max(worst_idem, float(np.abs(i_projection(C, hat, b) - hat).max()))
    worst_pyth = max(
        worst_pyth,
        abs(divergence(star, prior) - divergence(star, hat) - divergence(hat, prior)),
    )

    # a prior changed by exp(C^T xi) is the same point of the geometry and must give the same output
    xi = rng.normal(size=n_p)
    worst_gauge = max(
        worst_gauge,
        float(np.abs(i_projection(C, prior * np.exp(C.T @ xi), b) - hat).max()),
    )

print(
    f"    200 random reversible nets: |C sigma - b| {worst_feas:.1e}, smallest entry {worst_pos:.1e}, "
    f"idempotence {worst_idem:.1e}   {ok(worst_feas < 1e-7 and worst_pos > 0 and worst_idem < 1e-7)}"
)
print(
    f"    Pythagorean identity D(s*||prior) = D(s*||hat) + D(hat||prior): worst gap {worst_pyth:.1e}   {ok(worst_pyth < 1e-7)}"
)
print(
    f"    prior rescaled by exp(C^T xi) gives the same projection: worst change {worst_gauge:.1e}   {ok(worst_gauge < 1e-6)}"
)

if FAILED:
    raise SystemExit(f"{len(FAILED)} checks failed")
