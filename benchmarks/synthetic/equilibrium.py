"""Thermodynamic equilibrium layer (THEORY.md, P8) and its evaluation.

A reversible net with place energies E_p = f(e_p), f hidden, relaxes from a marking m_0. The task is to predict where
it ends up. The answer is the unique minimiser of the free energy on the compatibility class of m_0,

    m* = argmin sum_p m_p (log(m_p / m_ref,p) - 1)   subject to   X^T m = X^T m_0   with   m_ref = exp(-E),

where X is a basis of the P-invariants, that is of ker C^T. ThermoNPF learns only the energies E_theta(e_p). The layer
solves the strictly convex dual with Newton's method and is differentiated by one Newton step at the solution, which
is the implicit function theorem. Conservation is exact by construction, positivity follows from the exponential form
and uniqueness from convexity. A message-passing readout has none of the three, and the conserved totals are global,
so a fixed number of rounds cannot compute m* on nets larger than the receptive field.

    uv run python -m benchmarks.synthetic.equilibrium    # writes results/equilibrium.json
"""
import json
from pathlib import Path

import numpy as np
import torch

from npf import datasets, models, nets
from npf.models.equilibrium import ThermoNPF

from .experiment import predict, train

SMALL, LARGE = (6, 12), (20, 30)

# hidden from every model
energy = lambda e: 1.5 * np.sin(2.5 * e) + e ** 2


def equilibrium(net, M0):
    """Ground truth by damped Newton on the dual, in double precision. M0 [S, P] -> m* [S, P]."""
    X, m_ref = net.X, np.exp(-energy(net.e[:, 0]))
    if X.shape[1] == 0:
        return np.tile(m_ref, (len(M0), 1))

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


def make_split(seed, n_nets, places, n_samples=32, scale=4.0):
    rng = np.random.default_rng(seed)
    groups = []
    for _ in range(n_nets):
        n_places = int(rng.integers(places[0], places[1] + 1))
        net = nets.random_net(rng, n_places, n_base=max(2, int(round(0.6 * n_places))), max_arity=2, p_reverse=1.0)
        M0 = rng.uniform(0.2, scale, size=(n_samples, n_places))
        groups.append(datasets.Group(net, M0, y=equilibrium(net, M0)))

    return groups




@torch.no_grad()
def evaluate(model, groups, device):
    preds = predict(model, groups, device)
    err = np.concatenate([(p - g.y).ravel() for g, p in zip(groups, preds)])
    move = np.concatenate([(g.y - g.m).ravel() for g in groups])
    drift = [np.abs((p - g.m) @ g.net.X).max() / np.abs(g.m @ g.net.X).max() for g, p in zip(groups, preds) if g.net.X.shape[1]]

    # 1.0 = "nothing happens"
    return {"nrmse": float(np.sqrt((err ** 2).mean() / (move ** 2).mean())),
            "conservation_drift": float(np.mean(drift)), "negative": float(np.mean(np.concatenate([p.ravel() for p in preds]) < -1e-6))}


def main(root="results", iters=3000, seeds=2, model_names=("gnn", "pgnn", "pgnn+", "pgnn+se", "npf@16", "npf-thermo")):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = {"train": make_split(1, 300, SMALL), "val": make_split(2, 40, SMALL), "test": make_split(3, 100, SMALL),
              "test-large": make_split(4, 100, LARGE), "test-tokens": make_split(5, 100, SMALL, scale=12.0)}
    path = Path(root, "equilibrium.json")
    for name in model_names:
        out = {}
        for seed in range(seeds):
            torch.manual_seed(seed)
            model = (ThermoNPF() if name == "npf-thermo" else models.build(name, "next")).to(device)
            train(model, "next", splits, device, iters, seed)
            model.eval()
            row = {k: evaluate(model, v, device) for k, v in splits.items() if k.startswith("test")}
            out.setdefault(name, []).append(row | {"params": sum(p.numel() for p in model.parameters())})
            print(f"{name:>10} seed {seed}: " + "  ".join(f"{k}: nrmse {v['nrmse']:.4f} drift {v['conservation_drift']:.1e}" for k, v in row.items()), flush=True)

        # several processes may run different models
        merged = json.loads(path.read_text()) if path.exists() else {}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(merged | out, indent=1))


if __name__ == "__main__":
    import sys
    main(**({"model_names": tuple(sys.argv[1:])} if len(sys.argv) > 1 else {}))
