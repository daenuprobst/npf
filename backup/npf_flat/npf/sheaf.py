"""Learned incidence: the arc weights (conversion ratios) are hidden and have to be learned.

The paper's motivating example is flow with conversion between semantic layers (1 GBP -> 1.32 USD). Here every place
has a type ("currency"), every transition moves one unit out of its input place and R[type_in, type_out] units into
its output place, and R is *not* given to any model. NPF learns R as a table of restriction maps indexed by the types
of the two ends of an arc (the one-dimensional case of a cellular sheaf on the net: stalks R, restriction maps Pre and
Pos); the dense incidence matrix C_theta is rebuilt from it in every forward pass, and the token game, the projection
and the Petri Laplacian C W C^T all run on the learned net. Whatever R_theta is, the model is a Petri net, its
conservation laws are ker C_theta^T, and they can be read off (here: the price vector, up to scale).

    uv run python -m npf.sheaf            # writes results/sheaf.json

  npf-learned   NPF with learned conversion table                 npf-unit     NPF that assumes all ratios are 1
  npf-oracle    NPF with the true ratios                          pgnn, pgnn+  no incidence matrix at all
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data, models
from .experiment import predict, train, transition_metrics

N_TYPES = 4
PRICE = np.array([1.0, 1.32, 0.76, 2.05])  # hidden; R[a, b] = PRICE[a] / PRICE[b]
SMALL, LARGE = (6, 12), (20, 30)
# P- and T-invariants are not continuous in the arc weights: a cycle of conversions is a T-invariant only if its ratios
# multiply to exactly one. Learned weights are never exact, so the numerical rank of C_theta is taken with a tolerance.
RANK_TOLERANCE = 1e-3


def make_split(seed, n_nets, places, n_samples=64):
    """Returns (seen, true): the same samples on nets with unit arc weights (what a model sees) and with the true ones."""
    rng = np.random.default_rng(seed)
    seen, true = [], []
    for _ in range(n_nets):
        n_places = int(rng.integers(places[0], places[1] + 1))
        unit = data.random_net(rng, n_places, n_base=max(2, int(round(0.6 * n_places))), max_arity=1)
        kind = rng.integers(0, N_TYPES, n_places)
        e = kind[:, None].astype(float)  # the place type is the static place attribute
        source = np.empty(unit.n_trans, int); source[unit.pre_t] = unit.pre_p
        ratio = PRICE[kind[source[unit.pos_t]]] / PRICE[kind[unit.pos_p]]
        nets = [data.Net(unit.n_places, unit.n_trans, unit.pre_p, unit.pre_t, unit.pre_w, unit.pos_p, unit.pos_t, w, e=e, a=unit.a)
                for w in (unit.pos_w, ratio)]
        M0 = rng.integers(0, 9, size=(n_samples, n_places)) * (rng.random((n_samples, n_places)) > 0.25)
        t_a, dt = rng.uniform(0.0, 0.5, n_samples), rng.uniform(0.2, 1.0, n_samples)
        MA, MB, sigma = data.gillespie_pairs(nets[1], M0.astype(float), t_a, t_a + dt, rng)
        seen.append(data.Group(nets[0], MA, MB, dt, sigma)); true.append(data.Group(nets[1], MA, MB, dt, sigma))
    return seen, true


class SheafNPF(models.NPF):
    def __init__(self, consistency=True):
        super().__init__("transitions")
        self.consistency = consistency  # False: the arc weights only receive gradient through the projection (fails, see REPORT)
        self.raw_ratio = nn.Parameter(torch.zeros(N_TYPES, N_TYPES))  # restriction maps, indexed by the types of the two ends

    @property
    def log_ratio(self):
        return 10.0 * self.raw_ratio  # Adam steps are scale-free: the arc weights move ten times faster than the MLP weights

    def learned_net(self, b, detach=False):
        kind = b.e[:, 0].long()
        source = torch.zeros(b.n_trans, dtype=torch.long, device=kind.device).index_copy_(0, b.pre_t, kind[b.pre_p])
        log_ratio = self.log_ratio.detach() if detach else self.log_ratio
        pos_w = torch.exp(log_ratio[source[b.pos_t], kind[b.pos_p]])
        G, S, Pmax, Tmax = b.pad_shape
        first = lambda pad, size: (pad // size) % S == 0  # arcs of the first sample of every net carry the structure
        dense = torch.zeros(G, Pmax, Tmax, dtype=torch.float64, device=kind.device)
        for p, t, w, sign in ((b.pos_p, b.pos_t, pos_w, 1.0), (b.pre_p, b.pre_t, b.pre_w, -1.0)):
            keep = first(b.pad_p[p], Pmax)
            g = b.pad_p[p][keep] // (S * Pmax)
            dense.index_put_((g, b.pad_p[p][keep] % Pmax, b.pad_t[t][keep] % Tmax), sign * w[keep].double(), accumulate=True)
        return dataclasses.replace(b, pos_w=pos_w, C=dense.float(), C_pinv=torch.linalg.pinv(dense, rtol=RANK_TOLERANCE).float())

    def forward(self, b, m=None):
        # with the consistency loss the arc weights are identified by the state equation alone (a linear regression);
        # the gradient of the counting loss through the projection is not needed and, on its own, fails to find them
        return super().forward(self.learned_net(b, detach=self.consistency), m)

    def auxiliary_loss(self, b, sigma):
        """State-equation consistency: with the recorded firing counts, the learned net has to take A to B.
        This identifies the arc weights directly (a linear regression of dM on sigma), independently of the rate law."""
        if not self.consistency:
            return 0.0
        learned = self.learned_net(b)
        return F.mse_loss(models.apply_incidence(sigma, learned), b.m_b - b.m)


def main(root="results", iters=3000, seeds=2, model_names=("pgnn", "pgnn+", "npf-unit", "npf-learned-projection-only", "npf-learned", "npf-oracle")):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = ("train", "val", "test", "test-large")
    made = {n: make_split(k + 1, c, LARGE if n == "test-large" else SMALL) for k, (n, c) in enumerate(zip(names, (300, 40, 100, 100)))}
    seen, true = {n: v[0] for n, v in made.items()}, {n: v[1] for n, v in made.items()}
    true_ratio = PRICE[:, None] / PRICE[None, :]
    path = Path(root, "sheaf.json")
    out = json.loads(path.read_text()) if path.exists() else {}
    for name in model_names:
        out.pop(name, None)
        for seed in range(seeds):
            torch.manual_seed(seed)
            model = (SheafNPF(consistency=name == "npf-learned") if name.startswith("npf-learned")
                     else models.build(name.replace("-unit", "").replace("-oracle", ""), "transitions")).to(device)
            view = true if name == "npf-oracle" else seen
            train(model, "transitions", view, device, iters, seed)
            model.eval()
            row = {}
            for split in ("test", "test-large"):
                m = transition_metrics(true[split], predict(model, view[split], device))  # always scored on the true net
                row[split] = {k: m[k] for k in ("rmse", "mae", "rmse_row", "consistent", "unexplained_tokens")}
            if name.startswith("npf-learned"):
                learned = model.log_ratio.detach().exp().cpu().numpy()
                off = ~np.eye(N_TYPES, dtype=bool)
                row["ratio_relative_error"] = float(np.abs(learned[off] / true_ratio[off] - 1).max())
                cycle = learned[:, :, None] * learned[None, :, :] / learned[:, None, :]  # R[a,b] R[b,c] / R[a,c] = 1 without arbitrage
                row["arbitrage_gap"] = float(np.abs(cycle[off][:, :][np.isfinite(cycle[off])] - 1).max())
                row["learned_ratios"] = learned.round(3).tolist()
            out.setdefault(name, []).append(row)
            print(f"{name:>12} seed {seed}: " + "  ".join(f"{k}: rmse {v['rmse']:.3f} takes A to B {v['consistent']:.3f}" for k, v in row.items() if isinstance(v, dict))
                  + (f"  max ratio error {row['ratio_relative_error']:.2%}" if "ratio_relative_error" in row else ""), flush=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys
    main(**({"model_names": tuple(sys.argv[1:])} if len(sys.argv) > 1 else {}))
