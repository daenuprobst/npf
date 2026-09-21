"""Numerical checks of the Petri-net mathematics the architecture proposal relies on.

    uv run python -m npf.theory_checks

A. Thermodynamic flux law with an arbitrary positive *neural* conductance:
   P-invariants conserved, free energy is Lyapunov, equilibrium potentials lie in ker C^T.
B. Softmax is the equilibrium of the simplest conservative Petri net (a state machine).
C. Sinkhorn / entropic OT is the equilibrium of the partner-swap Petri net.
D. Discrete 'neural token game' layer: allocation (softmax) -> synchronisation (min) -> state equation
   keeps markings non-negative and conserves P-invariants for arbitrary scores/gates.
E. Rate-independent min-semantics net computes ReLU.
F. Deficiency-zero + weakly reversible mass-action net (NOT detailed balanced): unique equilibrium per
   compatibility class for arbitrary rate constants, pseudo-Helmholtz function decreases.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space

rng = np.random.default_rng(7)
softplus = lambda z: np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)


def random_net(P, T):
    while True:
        Pre = rng.integers(1, 3, size=(P, T)) * (rng.random((P, T)) < 0.35)
        Pos = rng.integers(1, 3, size=(P, T)) * (rng.random((P, T)) < 0.35)
        Pos = np.where(Pre > 0, 0, Pos)
        ok = (Pre.sum(0) > 0).all() and (Pos.sum(0) > 0).all() and ((Pre + Pos).sum(1) > 0).all()
        if ok:
            return Pre.astype(float), Pos.astype(float)


def make_mlp(n_in, n_out, width=16):
    W1, b1 = rng.normal(size=(width, n_in)), rng.normal(size=width)
    W2, b2 = rng.normal(size=(n_out, width)), rng.normal(size=n_out)
    return lambda m: softplus(W2 @ np.tanh(W1 @ np.log(m) + b1) + b2) + 0.05  # strictly positive


def thermo_flow(Pre, Pos, m_ref, kappa):
    C = Pos - Pre

    def rhs(_, m):
        m = np.maximum(m, 1e-300)
        mu = np.log(m / m_ref)
        v = kappa(m) * (np.exp(Pre.T @ mu) - np.exp(Pos.T @ mu))
        return C @ v

    return C, rhs


def free_energy(m, m_ref):
    return np.sum(m * (np.log(m / m_ref) - 1.0) + m_ref)


def integrate(rhs, m0, t_end, n=400):
    ts = np.concatenate([[0.0], np.geomspace(1e-4, t_end, n)])
    sol = solve_ivp(rhs, (0, t_end), m0, method="LSODA", t_eval=ts, rtol=1e-11, atol=1e-13)
    assert sol.success, sol.message
    return sol.t, sol.y.T


# ---------------------------------------------------------------- A
print("A. thermodynamic neural Petri flow on a random net")
P, T = 7, 4
Pre, Pos = random_net(P, T)
m_ref = np.exp(rng.normal(size=P))
kappa = make_mlp(P, T)
C, rhs = thermo_flow(Pre, Pos, m_ref, kappa)
X = null_space(C.T)  # basis of P-invariants (left kernel of C)
m0 = np.exp(rng.normal(size=P))
ts, ms = integrate(rhs, m0, 5e3)
Fs = np.array([free_energy(m, m_ref) for m in ms])
mu_end = np.log(ms[-1] / m_ref)
print(f"   #P-invariants (dim ker C^T)          : {X.shape[1]}")
print(f"   max drift of conserved quantities    : {np.abs((ms - m0) @ X).max():.2e}")
print(f"   min marking along trajectory         : {ms.min():.3e}  (positivity)")
print(f"   max increase of free energy F        : {np.diff(Fs).max():.2e}  (<= ~0 => Lyapunov)")
print(f"   F(0) -> F(end)                       : {Fs[0]:.4f} -> {Fs[-1]:.4f}")
print(f"   ||C^T mu*|| at the end               : {np.linalg.norm(C.T @ mu_end):.2e}  (mu* in P-invariant space)")
resid = mu_end - X @ (X.T @ mu_end)
print(f"   dist(mu*, span of P-invariants)      : {np.linalg.norm(resid):.2e}")

# ---------------------------------------------------------------- B
print("B. softmax as equilibrium of a state-machine Petri net (p_i <-> p_{i+1})")
n = 6
E = rng.normal(size=n) * 2
Pre = np.zeros((n, n - 1)); Pos = np.zeros((n, n - 1))
for i in range(n - 1):
    Pre[i, i] = 1; Pos[i + 1, i] = 1
_, rhs = thermo_flow(Pre, Pos, np.exp(-E), make_mlp(n, n - 1))
m0 = rng.random(n); m0 /= m0.sum()
_, ms = integrate(rhs, m0, 5e3)
sm = np.exp(-E) / np.exp(-E).sum()
print(f"   max |m* - softmax(-E)|               : {np.abs(ms[-1] - sm).max():.2e}")

# ---------------------------------------------------------------- C
print("C. Sinkhorn as equilibrium of the partner-swap net  QK_ij + QK_kl <-> QK_il + QK_kj")
n = 3
idx = lambda i, j: i * n + j
moves = [(i, k, j, l) for i in range(n) for k in range(i + 1, n) for j in range(n) for l in range(j + 1, n)]
Pre = np.zeros((n * n, len(moves))); Pos = np.zeros((n * n, len(moves)))
for t, (i, k, j, l) in enumerate(moves):
    Pre[idx(i, j), t] = 1; Pre[idx(k, l), t] = 1
    Pos[idx(i, l), t] = 1; Pos[idx(k, j), t] = 1
Eij = rng.normal(size=(n, n)) * 1.5
K = np.exp(-Eij)
C, rhs = thermo_flow(Pre, Pos, K.ravel(), make_mlp(n * n, len(moves)))
M0 = rng.random((n, n)) + 0.1
r, c = M0.sum(1), M0.sum(0)
_, ms = integrate(rhs, M0.ravel(), 5e4)
Mstar = ms[-1].reshape(n, n)
a, b = np.ones(n), np.ones(n)
for _ in range(20000):  # Sinkhorn scaling of K to marginals (r, c)
    a = r / (K @ b); b = c / (K.T @ a)
S = a[:, None] * K * b[None, :]
print(f"   dim ker C^T (expect 2n-1 = {2*n-1})        : {null_space(C.T).shape[1]}")
print(f"   max |m* - Sinkhorn(K; r, c)|         : {np.abs(Mstar - S).max():.2e}")

# ---------------------------------------------------------------- D
print("D. discrete neural token game: allocate (softmax) -> synchronise (min) -> state equation")
P, T = 9, 7
Pre, Pos = random_net(P, T)
C = Pos - Pre
X = null_space(C.T)
m = rng.random(P) * 5
m_init = m.copy()
min_seen, fired = np.inf, 0.0
for step in range(2000):
    scores = rng.normal(size=(P, T)) * 3           # stand-in for a neural conflict-resolution policy
    gate = 1 / (1 + np.exp(-rng.normal(size=T) * 3))  # stand-in for neural guards in (0,1)
    A = np.where(Pre > 0, np.exp(scores), 0.0)
    A = A / np.maximum(A.sum(1, keepdims=True), 1e-300)  # each place splits its tokens among its consumers
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(Pre > 0, A * m[:, None] / Pre, np.inf)
    v = gate * ratio.min(0)                        # enabling degree under the allocation
    m = m + C @ v
    min_seen = min(min_seen, m.min()); fired += v.sum()
print(f"   #P-invariants                        : {X.shape[1]}")
print(f"   min marking over 2000 random steps   : {min_seen:.3e}  (>= 0 up to fp)")
print(f"   drift of conserved quantities        : {np.abs(X.T @ (m - m_init)).max():.2e}")
print(f"   total firing (net is not trivially dead): {fired:.2f}")

# ---------------------------------------------------------------- E
print("E. ReLU from synchronisation:  X1 -> Y ;  X2 + Y -> W   (rate-independent)")
Pre = np.array([[1, 0], [0, 1], [0, 1], [0, 0]], float)  # places X1, X2, Y, W
Pos = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], float)
C = Pos - Pre
worst = 0.0
for _ in range(200):
    x1, x2 = rng.random(2) * 10
    m = np.array([x1, x2, 0.0, 0.0])
    for _ in range(400):
        g = rng.random(2) * 0.9 + 0.05             # arbitrary (adversarial-ish) rates
        with np.errstate(divide="ignore"):
            ratio = np.where(Pre > 0, m[:, None] / Pre, np.inf)
        m = m + C @ (g * ratio.min(0))
    worst = max(worst, abs(m[2] - max(x1 - x2, 0.0)))
print(f"   max |Y - relu(x1 - x2)| over 200 runs: {worst:.2e}")

# ---------------------------------------------------------------- F
print("F. deficiency-zero, weakly reversible, NOT detailed balanced:  A->B->C->A,  2A<->D")
# places A,B,C,D ; transitions: A->B, B->C, C->A, 2A->D, D->2A
Pre = np.array([[1, 0, 0, 2, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]], float)
Pos = np.array([[0, 0, 1, 0, 2], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0]], float)
C = Pos - Pre
n_complexes, n_linkage = 5, 2
delta = n_complexes - n_linkage - np.linalg.matrix_rank(C)
x_inv = np.array([1, 1, 1, 2.0])
print(f"   deficiency                           : {delta}")
for trial in range(3):
    k = np.exp(rng.normal(size=5) * 1.5)           # arbitrary positive rate constants
    rhs = lambda _, m, k=k: C @ (k * np.prod(np.maximum(m, 0)[:, None] ** Pre, axis=0))
    ends = []
    for _ in range(4):                             # different initial markings, same conserved total
        m0 = rng.random(4) + 0.05
        m0 *= 3.0 / (x_inv @ m0)
        ts, ms = integrate(rhs, m0, 1e4)
        ends.append(ms[-1])
    ends = np.array(ends)
    mstar = ends[0]
    G = np.array([np.sum(m * (np.log(m / mstar) - 1) + mstar) for m in ms])
    print(f"   k#{trial}: spread of equilibria across inits = {np.ptp(ends, axis=0).max():.2e},"
          f" max increase of pseudo-Helmholtz = {np.diff(G).max():.2e},"
          f" steady flux in ker C: ||C v*|| = {np.linalg.norm(rhs(0, mstar)):.1e},"
          f" |v*| = {np.linalg.norm(k * np.prod(mstar[:, None] ** Pre, axis=0)):.2f}")
