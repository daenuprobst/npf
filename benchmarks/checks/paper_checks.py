"""Numerical checks of the propositions as stated in paper_v2/sections, numbered as in the paper, and of a few
statements of Appendix D that have no number. Wherever possible the check runs against the implementation that produced
the results with random untrained weights, since a guarantee for all weights must also hold at initialisation. The
counts that the paper quotes are written to results/checks.json.

    CUDA_VISIBLE_DEVICES="" uv run python -m benchmarks.checks.paper_checks

This began as an experiment on whether Qwen 3.6 27B can translate propositions into code, with corrections added since.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import null_space

from npf import batching, chem, datasets, layers, models
from npf.models.equilibrium import ThermoNPF
from npf.nets import Net
from benchmarks.chemistry.experiment import splits
from benchmarks.chemistry.invariance import RADIUS, add_methyl, remote_site

rng = np.random.default_rng(11)
torch.manual_seed(11)
FAILED = []
COUNTS = {}


def ok(cond):
    """Text of one check. A failure is remembered, the script then ends with a non-zero status."""
    if not cond:
        FAILED.append(1)

    return "ok" if cond else "FAILED"


def random_net(P, n_base, p_reverse=0.5):
    """Net whose transitions have one or two input and output places with arc weights 1 or 2, a reversed copy with
    probability p_reverse, and in which every place lies on a transition."""
    pre, pos = [], []

    # unused places are handed out first so that every place takes part
    unused = list(rng.permutation(P))
    t = 0
    for _ in range(n_base):
        n_in, n_out = rng.integers(1, 3, size=2)
        nodes, unused = unused[: n_in + n_out], unused[n_in + n_out :]
        others = [p for p in rng.permutation(P) if p not in nodes]
        nodes = list(rng.permutation(nodes + others[: n_in + n_out - len(nodes)]))
        w = rng.integers(1, 3, size=n_in + n_out)
        ins, outs = list(zip(nodes[:n_in], w[:n_in])), list(zip(nodes[n_in:], w[n_in:]))
        for src, dst in [(ins, outs)] + ([(outs, ins)] if rng.random() < p_reverse else []):
            pre += [(p, t, wt) for p, wt in src]
            pos += [(p, t, wt) for p, wt in dst]
            t += 1

    pre, pos = np.array(pre), np.array(pos)
    assert len(set(pre[:, 0]) | set(pos[:, 0])) == P

    return Net(
        P, t, pre[:, 0], pre[:, 1], pre[:, 2].astype(float), pos[:, 0], pos[:, 1], pos[:, 2].astype(float),
        e=rng.uniform(-0.7, 0.7, size=(P, 1)), a=rng.uniform(-1.0, 1.0, size=(t, 1)),
    )


def random_incidence(P=9, need_invariant=False):
    while True:
        try:
            net = random_net(P, n_base=max(2, int(round(0.6 * P))))

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
    f"    dim ker C^T = {X.shape[1]},  |X^T F| for F in im C: {np.abs(X.T @ F_in).max():.1e},  for F not in im C: {np.abs(X.T @ F_out).max():.2f}"
    f"   {ok(np.abs(X.T @ F_in).max() < 1e-10 and np.abs(X.T @ F_out).max() > 1e-3)}"
)
print(
    f"    F in im C is reproduced by v = C^+ F: |C v - F| = {np.abs(C @ v - F_in).max():.1e}   {ok(np.abs(C @ v - F_in).max() < 1e-10)}"
)

# Proposition 5
print(
    "P5  token-game layer with arbitrary demands d >= 0, a numpy transcription of Eq. 6"
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



def reachable_pairs(n_nets=12, n_samples=16):
    """Markings m_A and m_B = m_A + C sigma of one run on random nets. m_A holds every token that sigma consumes, so
    the transitions can fire their counts in any order."""
    groups = []
    for _ in range(n_nets):
        net, _, _ = random_incidence(P=int(rng.integers(6, 13)))
        sigma = rng.poisson(1.5, (n_samples, net.n_trans)).astype(float)
        m_a = rng.integers(0, 5, (n_samples, net.n_places)) + sigma @ net.Pre.T
        groups.append(datasets.Group(net, m_a, m_a + sigma @ net.C.T, np.ones(n_samples)))

    return groups


groups = reachable_pairs()
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

# the implementation, the local rate law of models.NPF with random weights vanishes wherever an input place is empty
npf_rate = models.NPF("transitions")
with torch.no_grad():
    m = batch.m.clone()
    m[torch.rand(m.shape) < 0.3] = 0.0
    empty = torch.zeros(batch.n_trans).index_add_(0, batch.pre_t, (m[batch.pre_p] == 0).float()) > 0
    rate = npf_rate.rate(m, batch)

print(
    f"    implementation (models.NPF.rate, random weights): {int(empty.sum())} transitions with an empty input place, "
    f"largest rate {float(rate[empty].max()):.1e}, smallest other {float(rate[~empty].min()):.1e}   "
    f"{ok(float(rate[empty].abs().max()) == 0.0 and float(rate[~empty].min()) > 0)}"
)

# Appendix D.3, no proposition, the run between two markings is a bridge, for which the bounded correction stands in
print(
    "D.3  given both markings the run is a bridge, its intensity is the Doob transform of the rate law"
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

# no proposition, the least-squares projection onto the fibre of the projected baselines of Appendix D.4
print("LS  least-squares projection onto the state equation")
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
    f"    implementation (project_state_equation on reachable pairs): |m_B - m_A - C sigma^| = {residual:.1e}, "
    f"min sigma^ = {out.min():.3f}   {ok(residual < 1e-3)}"
)

# Proposition 6
print("P6  locality lower bound on the path net")
L, k, delta = 48, 24, 5.0
ones = np.ones(L)
net = Net(
    L + 1, L, np.arange(L), np.arange(L), ones, np.arange(L) + 1, np.arange(L), ones,
    e=rng.uniform(-0.7, 0.7, size=(L + 1, 1)), a=rng.uniform(-1.0, 1.0, size=(L, 1)),
)


def random_run(m, n_firings):
    """A run of the token game from m that fires an enabled transition chosen at random. The last marking and the
    firing counts."""
    m, sigma = m.copy(), np.zeros(net.n_trans)
    for _ in range(n_firings):
        enabled = np.nonzero((m[:, None] >= net.Pre).all(0))[0]
        if not len(enabled):
            break

        t = rng.choice(enabled)
        m += net.C[:, t]
        sigma[t] += 1

    return m, sigma


MA = np.stack([random_run(m0, 60)[0] for m0 in rng.integers(0, 9, size=(8, L + 1)).astype(float)])
MB, sigma = map(np.stack, zip(*(random_run(m, 200) for m in MA)))
formula = -np.cumsum(MB - MA, 1)[:, :-1]
print(
    f"    sigma*_k = -sum_(j<k) (m_B - m_A)_j on random runs: max error {np.abs(formula - sigma).max():.1e}   {ok(np.abs(formula - sigma).max() < 1e-9)}"
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
    f"    second input reachable: after moving delta through the chain the marking is m_A + delta e_pL >= m_A: "
    f"{ok(np.allclose(m, MA + delta * np.eye(L + 1)[-1]))}"
)
pgnn = models.PGNN(rounds=4)

with torch.no_grad():
    dt = np.full(8, 0.7)
    out1 = pgnn(
        batching.collate(
            [datasets.Group(net, MA, MB, dt)], [np.arange(8)], "cpu"
        )
    ).view(8, L)
    out2 = pgnn(
        batching.collate(
            [datasets.Group(net, MA2, MB2, dt)], [np.arange(8)], "cpu"
        )
    ).view(8, L)

same = float((out1[:, k] - out2[:, k]).abs().max())
print(
    f"    PGNN (4 rounds) at the middle transition: outputs differ by {same:.1e} on the two inputs, targets differ by {delta}   {ok(same < 1e-6)}"
)

# no longer in the paper, the equilibrium layer of an earlier version
print("EQ  equilibrium layer, not in the paper")


def equilibrium(net, m_ref, M0):
    """The minimiser of the free energy on the class of every row of M0, damped Newton on the dual in double precision."""
    X = net.X
    b, lam = M0 @ X, np.zeros((len(M0), X.shape[1]))
    dual = lambda l: (m_ref * np.exp(l @ X.T)).sum(1) - (l * b).sum(1)
    for _ in range(200):
        m = m_ref * np.exp(lam @ X.T)
        grad = m @ X - b
        if np.abs(grad).max() < 1e-11:
            break

        H = np.einsum("pr,sp,pq->srq", X, m, X) + 1e-12 * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad[..., None])[..., 0]
        alpha = np.ones(len(M0))

        # backtracking
        for _ in range(30):
            worse = dual(lam - alpha[:, None] * step) > dual(lam) - 1e-4 * alpha * (grad * step).sum(1)
            if not worse.any():
                break

            alpha[worse] /= 2

        lam = lam - alpha[:, None] * step

    return m_ref * np.exp(lam @ X.T)


net = random_net(10, n_base=6, p_reverse=1.0)
net.e = rng.uniform(-0.7, 0.7, size=(10, 1))
E = 1.5 * np.sin(2.5 * net.e[:, 0]) + net.e[:, 0] ** 2
m_ref = np.exp(-E)
m0 = rng.uniform(0.3, 3.0, size=(1, 10))
m_star = equilibrium(net, m_ref, m0)[0]
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
group = datasets.Group(net, m0)
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
    "P3  state-equation readout, cancellation and invariance on random graphs with a random local psi"
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

# the implementation, the pure readout of chem.Classifier with random weights, a methyl group attached on both sides
# more than 2(K + 1) bonds from every change, which covers the bond terms of radius K + 1
cls_data = chem.load()
_, _, cls_test = splits(cls_data, "classify")
pairs = []
for r in cls_test:
    if r["target"] is None or len(r["a"]["x"]) > 150:
        continue

    site = remote_site(r, 2 * RADIUS + 1)
    if site is not None:
        twin = dict(r, a=add_methyl(r["a"], site[1]), b=add_methyl(r["b"], site[0]),
                    target=np.append(r["target"], len(r["a"]["x"])).astype(r["target"].dtype))
        pairs.append((r, twin))

    if len(pairs) >= 64:
        break

readout = chem.Classifier(len(cls_data["classes"]), petri=True, gate=False).eval()
generic = chem.Classifier(len(cls_data["classes"]), petri=False, gate=False).eval()
with torch.no_grad():
    change = lambda model: float(
        (model(chem.collate([p[0] for p in pairs], "cpu")) - model(chem.collate([p[1] for p in pairs], "cpu")))
        .abs()
        .max()
    )
    pure, pooled = change(readout), change(generic)

print(
    f"    implementation (chem.Classifier, random weights, {len(pairs)} test reactions, methyl >= {2 * RADIUS + 1} bonds "
    f"away): largest logit change {pure:.1e} for the pure readout, {pooled:.1e} for the generic one   "
    f"{ok(pure < 1e-4 and pooled > 1e-3)}"
)

# Proposition 4
print(
    "P4  token game on the valence net: every visited marking is valid, for random weights"
)
reactions = [
    r for r in chem.load()["reactions"] if r["target"] is not None and r["split"] == "test"
][:256]
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


COUNTS["p4"] = {"reactions": len(reactions), "firings": steps, "violations": violations}

# Appendix D.3, no proposition, the I-projection of the counts onto the fibre
print(
    "D.3  the I-projection is positive, idempotent, Pythagorean and invariant under a gauge change of the prior"
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

# the implementation, the I-projection of models.NPF on the reachable pairs
with torch.no_grad():
    prior = torch.rand(batch.n_trans) * 3 + 0.05
    out = layers.kl_project_state_equation(prior, batch)
    residual = (batch.m_b - batch.m - layers.apply_incidence(out, batch)).abs().max()

print(
    f"    implementation (kl_project_state_equation on reachable pairs): |m_B - m_A - C sigma^| = {residual:.1e}, "
    f"min sigma^ = {out.min():.1e}   {ok(residual < 1e-3 and out.min() > 0)}"
)

Path("results/checks.json").write_text(json.dumps(COUNTS, indent=1))

if FAILED:
    raise SystemExit(f"{len(FAILED)} checks failed")
