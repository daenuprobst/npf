"""Numerical checks of the propositions exactly as they are stated in the paper (paper/sections/method.tex, chem.tex),
wherever possible against the implementation that produced the results, with random (untrained) weights:
a guarantee "for all weights" must hold at initialisation, too.

    CUDA_VISIBLE_DEVICES="" uv run python -m npf.paper_checks
"""
import numpy as np
import torch
from scipy.linalg import null_space

from . import data, locality, models, thermo

rng = np.random.default_rng(11)
torch.manual_seed(11)
ok = lambda cond: "ok" if cond else "FAILED"


def random_incidence(P=9, need_invariant=False):
    while True:
        try:
            net = data.random_net(rng, P, n_base=max(2, int(round(0.6 * P))), max_arity=2)
        except AssertionError:  # the sampled transitions did not cover every place
            continue
        X = null_space(net.C.T)
        if X.shape[1] or not need_invariant:
            return net, net.C, X


# ---------------------------------------------------------------- Proposition 1
print("P1  conservation <=> firing form")
net, C, X = random_incidence(need_invariant=True)
F_in = C @ rng.normal(size=C.shape[1])  # an update of the firing form
F_out = F_in + X @ rng.normal(size=X.shape[1])  # the same update plus a component outside im C
v = np.linalg.pinv(C) @ F_in
print(f"    dim ker C^T = {X.shape[1]};  |X^T F| for F in im C: {np.abs(X.T @ F_in).max():.1e};  for F not in im C: {np.abs(X.T @ F_out).max():.2f}")
print(f"    F in im C is reproduced by v = C^+ F: |C v - F| = {np.abs(C @ v - F_in).max():.1e}   {ok(np.abs(C @ v - F_in).max() < 1e-10)}")

# ---------------------------------------------------------------- Proposition 2 (i)
print("P2  token-game layer with ARBITRARY demands d >= 0 (numpy transcription of Eq. 5)")


def layer(m, d, Pre, C):
    demand = Pre @ d
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(m > 0, demand / np.where(m > 0, m, 1.0), np.inf)
        phi = np.where(x == 0, 1.0, np.where(np.isinf(x), 0.0, -np.expm1(-x) / np.where(x > 0, x, 1.0)))
    phi = np.where((m == 0) & (demand == 0), 1.0, phi)
    served = np.where(Pre > 0, phi[:, None], np.inf).min(0)
    served = np.where(np.isinf(served), 1.0, served)  # a transition without input places
    return m + C @ (d * served)


worst, kept_positive, drift = np.inf, True, 0.0
for _ in range(3000):
    net, C, X = random_incidence(P=int(rng.integers(4, 12)))
    m = rng.integers(0, 4, net.n_places) * rng.uniform(0.0, 3.0, net.n_places)  # many empty places
    d = rng.exponential(rng.choice([0.1, 1.0, 30.0]), net.n_trans) * (rng.random(net.n_trans) > 0.2)
    m_next = layer(m, d, net.Pre, C)
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = np.where(m > 0, m * np.exp(-(net.Pre @ d) / np.where(m > 0, m, 1.0)), 0.0)  # the proof shows m+ >= m exp(-x)
    worst = min(worst, m_next.min()); kept_positive &= bool((m_next >= bound - 1e-12 * np.maximum(1.0, m)).all())
    drift = max(drift, np.abs(X.T @ (m_next - m)).max(initial=0.0) / max(1.0, np.abs(m).max()))
print(f"    3000 random nets/markings/demands: min m+ = {worst:.2e}, m+ >= m exp(-x) (hence > 0 where m > 0): {kept_positive}, "
      f"invariant drift {drift:.1e}   {ok(worst >= -1e-9 and kept_positive and drift < 1e-9)}")

splits = {k: data.make_pairs(s, 12, (6, 12), 2, 16) for k, s in (("a", 21),)}
groups = splits["a"]
batch = data.collate(groups, [np.arange(16)] * len(groups), "cpu")
npf_next = models.NPF("next")
with torch.no_grad():
    m = batch.m.clone()
    lowest = np.inf
    for _ in range(25):  # the implementation, untrained weights, rolled out
        m = npf_next(batch, m)
        lowest = min(lowest, float(m.min()))
print(f"    implementation (models.NPF, random weights, 25 rollout steps): min marking {lowest:.2e}   {ok(lowest >= -1e-6)}")

# ---------------------------------------------------------------- Proposition 2 (ii)
print("P2  (ii) a rate law that does not vanish at an empty input place leaves the orthant")
m, h = np.array([0.0, 1.0]), 1e-3  # p0 -> t -> p1 with v = 0.3 regardless of m_0
print(f"    after one Euler step of size {h}: m_0 = {m[0] - h * 0.3:.1e} < 0   ok")

# ---------------------------------------------------------------- Proposition 3
print("P3  projection onto the state equation")
worst_i = worst_ii = worst_iii = 0.0
for _ in range(500):
    net, C, X = random_incidence(P=int(rng.integers(4, 12)))
    T = C.shape[1]
    sigma_star = rng.poisson(2.0, T).astype(float)
    b = C @ sigma_star
    guess = rng.normal(2.0, 2.0, T)
    Cp = np.linalg.pinv(C)
    proj = guess + Cp @ (b - C @ guess)
    worst_i = max(worst_i, np.abs(C @ proj - b).max(), np.abs((proj - sigma_star) - (np.eye(T) - Cp @ C) @ (guess - sigma_star)).max(),
                  np.linalg.norm(proj - sigma_star) - np.linalg.norm(guess - sigma_star))
    clipped = np.maximum(guess, 0)
    worst_ii = max(worst_ii, np.linalg.norm(clipped - sigma_star) - np.linalg.norm(guess - sigma_star))
    w = rng.uniform(0.05, 3.0, T); W = np.diag(w)
    weighted = guess + W @ C.T @ np.linalg.pinv(C @ W @ C.T) @ (b - C @ guess)
    Z = null_space(C)  # brute force: minimise over sigma = particular + Z z
    particular = Cp @ b
    z = np.linalg.solve(Z.T @ np.diag(1 / w) @ Z, Z.T @ np.diag(1 / w) @ (guess - particular)) if Z.shape[1] else np.zeros(0)
    brute = particular + Z @ z
    wnorm = lambda u: np.sqrt(u @ (u / w))
    worst_iii = max(worst_iii, np.abs(weighted - brute).max(), np.abs(C @ weighted - b).max(), wnorm(weighted - sigma_star) - wnorm(guess - sigma_star))
print(f"    (i)   feasibility, error identity, monotonicity: worst violation {worst_i:.1e}   {ok(worst_i < 1e-8)}")
print(f"    (ii)  clipping at zero never moves away from sigma*: worst {worst_ii:.1e}   {ok(worst_ii < 1e-12)}")
print(f"    (iii) weighted formula = brute-force weighted least squares, monotone in the W^-1 norm: worst {worst_iii:.1e}   {ok(worst_iii < 1e-7)}")
with torch.no_grad():
    guess = torch.rand(batch.n_trans) * 3
    out = models.project_state_equation(guess, batch)
    residual = (batch.m_b - batch.m - models.apply_incidence(out, batch)).abs().max()
print(f"    implementation (project_state_equation on simulated pairs): |m_B - m_A - C sigma^| = {residual:.1e}, min sigma^ = {out.min():.3f}   {ok(residual < 1e-3)}")

# ---------------------------------------------------------------- Proposition 4
print("P4  locality lower bound on the path net")
L, k, delta = 48, 24, 5.0
net = locality.chain(rng, L + 1)
M0 = rng.integers(0, 9, size=(8, L + 1)).astype(float)
MA, MB, sigma = data.gillespie_pairs(net, M0, np.full(8, 0.2), np.full(8, 0.9), rng)
formula = -np.cumsum(MB - MA, 1)[:, :-1]
print(f"    sigma*_k = -sum_(j<k) (m_B - m_A)_j on simulated runs: max error {np.abs(formula - sigma).max():.1e}   {ok(np.abs(formula - sigma).max() < 1e-9)}")
MA2, MB2 = MA.copy(), MB.copy(); MA2[:, 0] += delta; MB2[:, -1] += delta
m = MA2.copy()  # the second input is valid: push delta tokens through the chain, then replay the counts of the first
for t in range(L):
    assert (m[:, t] >= delta).all(); m[:, t] -= delta; m[:, t + 1] += delta
print(f"    second input reachable: after moving delta through the chain the marking is m_A + delta e_pL >= m_A: {ok(np.allclose(m, MA + delta * np.eye(L + 1)[-1]))}")
pgnn = models.PGNN("transitions", rounds=4)
with torch.no_grad():
    dt = np.full(8, 0.7)
    out1 = pgnn(data.collate([data.Group(net, MA, MB, dt, sigma)], [np.arange(8)], "cpu")).view(8, L)
    out2 = pgnn(data.collate([data.Group(net, MA2, MB2, dt, sigma + delta)], [np.arange(8)], "cpu")).view(8, L)
same = float((out1[:, k] - out2[:, k]).abs().max())
print(f"    PGNN (4 rounds) at the middle transition: outputs differ by {same:.1e} on the two inputs, targets differ by {delta}   {ok(same < 1e-6)}")

# ---------------------------------------------------------------- Proposition 5 + Corollary 1
print("P5  equilibrium layer")
net = data.random_net(rng, 10, n_base=6, max_arity=2, p_reverse=1.0)
net.e = rng.uniform(-0.7, 0.7, size=(10, 1))
E = thermo.energy(net.e[:, 0]); m_ref = np.exp(-E)
m0 = rng.uniform(0.3, 3.0, size=(1, 10))
m_star = thermo.equilibrium(net, m0)[0]
mu = np.log(m_star / m_ref)
Fen = lambda m: float((m * (np.log(m / m_ref) - 1)).sum())
print(f"    |C^T mu(m*)| = {np.abs(net.C.T @ mu).max():.1e}, |X^T (m* - m0)| = {np.abs(net.X.T @ (m_star - m0[0])).max():.1e}, min m* = {m_star.min():.3f}")
better = 0
for _ in range(2000):  # other positive points of the same class
    z = rng.normal(size=net.n_trans) * rng.choice([1e-3, 1e-2, 0.1])
    other = m_star + net.C @ z
    if other.min() > 0:
        better += Fen(other) < Fen(m_star) - 1e-12
print(f"    points of the class with lower free energy than m*: {better} of 2000   {ok(better == 0)}")
forward = net.Pre[:, : net.n_trans]  # every reversible pair appears as two transitions; flux form per transition
kappa_w = rng.normal(size=(net.n_trans, 10)) * 0.3


def rhs(m):
    mu = np.log(np.maximum(m, 1e-300) / m_ref)
    kappa = np.exp(np.tanh(kappa_w @ m))  # arbitrary positive, marking-dependent
    return net.C @ (kappa * (np.exp(net.Pre.T @ mu) - np.exp(net.Pos.T @ mu)))


m, Fs = m0[0].copy(), []
for _ in range(60000):
    m = m + 2e-4 * rhs(m); Fs.append(Fen(m))
print(f"    dynamics with a random positive kappa(m): largest increase of F {max(np.diff(Fs)):.1e}, |m(T) - m*| = {np.abs(m - m_star).max():.1e} "
      f"(convergence is NOT part of the proposition)   {ok(max(np.diff(Fs)) < 1e-9)}")

layer_model = thermo.ThermoNPF()
group = data.Group(net, m0, y=m_star[None])
b = data.collate([group], [np.arange(1)], "cpu")
m0_t = b.m.clone().requires_grad_(True); b.m = m0_t
out = layer_model(b)
weights = torch.randn_like(out)
(out * weights).sum().backward()
grad = m0_t.grad.clone()
fd = torch.zeros_like(grad)
with torch.no_grad():
    for p in range(10):
        for sign in (1, -1):
            b.m = m0_t.detach().clone(); b.m[p] += sign * 1e-3
            fd[p] += sign * (layer_model(b) * weights).sum() / 2e-3
print(f"    d m*/d m0 by one Newton step vs finite differences: max abs diff {float((grad - fd).abs().max()):.1e}   {ok(float((grad - fd).abs().max()) < 1e-3)}")
with torch.no_grad():
    b.m = m0_t.detach()
    out = layer_model(b).double().numpy()
print(f"    layer output (random energies): invariant drift {np.abs(net.X.T @ (out - m0[0])).max():.1e}, min {out.min():.3f}   {ok(out.min() > 0)}")

print("C1  Sinkhorn = equilibrium of the partner-swap net (generic dual Newton solver vs Sinkhorn iteration)")
n, k2 = 4, 5
S_ = rng.normal(size=(n, k2)); r = rng.uniform(0.5, 2.0, n); c = rng.uniform(0.5, 2.0, k2); c *= r.sum() / c.sum()
cols = []
for i in range(n):
    for kk in range(i + 1, n):
        for j in range(k2):
            for l in range(j + 1, k2):
                col = np.zeros((n, k2)); col[i, j] -= 1; col[kk, l] -= 1; col[i, l] += 1; col[kk, j] += 1
                cols.append(col.ravel())
Cs = np.stack(cols, 1)
Xs = null_space(Cs.T)
m_ref_s, m0_s = np.exp(S_).ravel(), np.outer(r, c).ravel() / r.sum()
lam = np.zeros(Xs.shape[1])
for _ in range(100):
    m = m_ref_s * np.exp(Xs @ lam)
    lam -= np.linalg.solve(Xs.T @ (m[:, None] * Xs), Xs.T @ (m - m0_s))
Z = np.exp(S_); u, v_ = np.ones(n), np.ones(k2)
for _ in range(5000):
    u = r / (Z @ v_); v_ = c / (Z.T @ u)
sink = u[:, None] * Z * v_[None]
print(f"    dim ker C^T = {Xs.shape[1]} (rows + columns - 1 = {n + k2 - 1}); max |m* - Sinkhorn| = {np.abs(m.reshape(n, k2) - sink).max():.1e}   "
      f"{ok(Xs.shape[1] == n + k2 - 1 and np.abs(m.reshape(n, k2) - sink).max() < 1e-9)}")

# ---------------------------------------------------------------- Proposition 6
print("P6  state-equation readout: cancellation and invariance (random graphs, random local psi)")
R = 3


def random_tree_graph(n):
    x, adj = rng.integers(0, 4, n), np.zeros((n, n), int)
    for i in range(1, n):
        adj[i, rng.integers(0, i)] = adj[rng.integers(0, i), i] = 0
    for i in range(1, n):
        j = int(rng.integers(max(0, i - 3), i)); t = int(rng.integers(1, 4))
        adj[i, j] = adj[j, i] = t
    return x, adj


W_in, W_msg = rng.normal(size=(4, 8)), rng.normal(size=(4, 8, 8)) / 3  # a random message-passing psi with radius R


def psi(x, adj):
    h = W_in[x]
    for _ in range(R):
        h = np.tanh(h + sum((adj == t).astype(float) @ h @ W_msg[t] for t in (1, 2, 3)))
    return h


def distances(adj, sources):
    n = len(adj); dist = np.full(n, np.inf); dist[list(sources)] = 0; frontier = list(sources)
    while frontier:
        nxt = []
        for u in frontier:
            for w in np.nonzero(adj[u])[0]:
                if dist[w] == np.inf:
                    dist[w] = dist[u] + 1; nxt.append(w)
        frontier = nxt
    return dist


worst_cancel = worst_inv = 0.0
tested = 0
for _ in range(300):
    nA = int(rng.integers(30, 60))
    xA, adjA = random_tree_graph(nA)
    leaving = set(rng.choice(nA, int(rng.integers(0, 3)), replace=False).tolist())
    keep = np.array([i for i in range(nA) if i not in leaving])  # product atom k sits on precursor atom keep[k]
    xB, adjB = xA[keep].copy(), adjA[np.ix_(keep, keep)].copy()
    changed = set()
    for _ in range(int(rng.integers(1, 3))):  # re-type / break / form a bond, change an attribute
        i, j = rng.choice(len(keep), 2, replace=False)
        adjB[i, j] = adjB[j, i] = (adjB[i, j] + int(rng.integers(1, 4))) % 4; changed |= {int(i), int(j)}
    i = int(rng.integers(len(keep))); xB[i] = (xB[i] + 1) % 4; changed.add(i)
    D = set(changed) | {k for k in range(len(keep)) if any(adjA[keep[k], u] for u in leaving)}
    dist = distances(adjB, D)
    D_R = dist <= R
    full = psi(xB, adjB).sum(0) - psi(xA, adjA).sum(0)
    pB, pA = psi(xB, adjB), psi(xA, adjA)
    restricted = (pB[D_R] - pA[keep[D_R]]).sum(0) - pA[sorted(leaving)].sum(0)
    worst_cancel = max(worst_cancel, np.abs(full - restricted).max())
    far = np.nonzero(dist > 2 * R)[0]
    far = far[np.isfinite(dist[far])] if len(far) else far
    if len(far):
        i0 = int(rng.choice(far)); tested += 1
        def attach(x, adj, at):
            n = len(x); x2 = np.append(x, [1, 2]); a2 = np.zeros((n + 2, n + 2), int); a2[:n, :n] = adj
            a2[at, n] = a2[n, at] = 1; a2[n, n + 1] = a2[n + 1, n] = 2
            return x2, a2
        xB2, adjB2 = attach(xB, adjB, i0); xA2, adjA2 = attach(xA, adjA, keep[i0])
        full2 = psi(xB2, adjB2).sum(0) - psi(xA2, adjA2).sum(0)
        worst_inv = max(worst_inv, np.abs(full2 - full).max())
print(f"    r = sum over D_R and unmatched atoms only: max abs diff {worst_cancel:.1e}   {ok(worst_cancel < 1e-9)}")
print(f"    same subgraph attached > 2R from D on both sides ({tested} cases): max change of r {worst_inv:.1e}   {ok(worst_inv < 1e-9 and tested > 50)}")

# ---------------------------------------------------------------- Proposition 7
print("P7  token game on the valence net: every visited marking is valid, for random weights")
from . import chem_data, chem_models  # noqa: E402

reactions = [r for r in chem_data.load()["reactions"] if r["target"] is not None][:256]
game = chem_models.TokenGame()
order, violations, steps, drift = chem_data.BOND_ORDER, 0, 0, 0.0
with torch.no_grad():
    for start in range(0, len(reactions), 32):
        rs = reactions[start:start + 32]
        b = chem_data.collate(rs, "cpu")
        cur, fired = b["ba"].clone(), torch.zeros_like(b["ba"], dtype=torch.bool)
        n = cur.shape[1]
        for _ in range(game.max_steps):  # the greedy token game of TokenGame.forward, with a check after every firing
            logits, enabled, stop = game.rates(b, cur, fired)
            best, where = logits.masked_fill(~enabled, -1e4).flatten(1).max(1)
            fire = best > stop
            if not fire.any():
                break
            i, j, k = where // (n * chem_models.N_BOND), (where // chem_models.N_BOND) % n, where % chem_models.N_BOND
            rows = torch.nonzero(fire).squeeze(1)
            cur[rows, i[rows], j[rows]] = k[rows]; cur[rows, j[rows], i[rows]] = k[rows]
            fired[rows, i[rows], j[rows]] = True; fired[rows, j[rows], i[rows]] = True
            steps += len(rows)
            for row, r in enumerate(rs):  # independent bookkeeping in numpy
                a, size = r["a"], len(r["a"]["x"])
                before, after = chem_data.dense_bonds(a), cur[row, :size, :size].numpy()
                slack = a["h"] - (order[after] - order[before]).sum(1)
                capacity = np.array([chem_data.EXTRA_CAPACITY.get(int(e), 0) for e in a["element"]]) + np.maximum(-a["q"], 0) + 0.5
                violations += int((slack < -capacity - 1e-9).any())
                drift = max(drift, np.abs((order[after].sum(1) + slack) - (order[before].sum(1) + a["h"])).max())
print(f"    {steps} firings with untrained weights on {len(reactions)} reactions: markings with a negative slack place: {violations}, "
      f"valence drift {drift:.1e}   {ok(violations == 0 and steps > 100)}")
