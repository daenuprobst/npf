"""Learned incidence. The arc weights, here conversion ratios, are hidden and have to be learned.

The motivating example of the PGNN paper is flow with conversion between semantic layers (1 GBP gives 1.32 USD). Here
every place has a type, its currency. Every transition moves one unit out of its input place and R[type_in, type_out]
units into its output place, and R is not given to any model. NPF learns R as a table of restriction maps indexed by
the types of the two ends of an arc. This is the one-dimensional case of a cellular sheaf on the net with stalks R and
restriction maps Pre and Pos. The dense incidence matrix C_theta is rebuilt from the table in every forward pass, and
the token game, the projection and the Laplacian C W C^T all run on the learned net. Whatever R_theta is, the model is
a Petri net, its conservation laws are ker C_theta^T, and they can be read off. Here they are the price vector up to
scale.

    uv run python -m benchmarks.synthetic.learned_incidence    # writes results/sheaf.json

npf-learned        NPF with the learned conversion table
npf-learned-flat   the same table written as a difference of type potentials, so cycles multiply to one
npf-unit      NPF that assumes all ratios are 1
npf-oracle    NPF with the true ratios
pgnn, pgnn+   no incidence matrix at all
"""
import json
from pathlib import Path

import numpy as np
import torch

from npf import datasets, models, nets, simulate
from npf.models.learned_incidence import SheafNPF

from .experiment import predict, train, transition_metrics

N_TYPES = 4

# hidden, R[a, b] = PRICE[a] / PRICE[b]
PRICE = np.array([1.0, 1.32, 0.76, 2.05])
SMALL, LARGE = (6, 12), (20, 30)


def make_split(seed, n_nets, places, n_samples=64):
    """Returns (seen, true), the same samples on nets with unit arc weights (what a model sees) and with the true ones."""
    rng = np.random.default_rng(seed)
    seen, true = [], []
    for _ in range(n_nets):
        n_places = int(rng.integers(places[0], places[1] + 1))
        unit = nets.random_net(rng, n_places, n_base=max(2, int(round(0.6 * n_places))), max_arity=1)
        kind = rng.integers(0, N_TYPES, n_places)

        # the place type is the static place attribute
        e = kind[:, None].astype(float)
        source = np.empty(unit.n_trans, int)
        source[unit.pre_t] = unit.pre_p
        ratio = PRICE[kind[source[unit.pos_t]]] / PRICE[kind[unit.pos_p]]
        pair = [nets.Net(unit.n_places, unit.n_trans, unit.pre_p, unit.pre_t, unit.pre_w, unit.pos_p, unit.pos_t, w, e=e, a=unit.a)
                for w in (unit.pos_w, ratio)]
        M0 = rng.integers(0, 9, size=(n_samples, n_places)) * (rng.random((n_samples, n_places)) > 0.25)
        t_a, dt = rng.uniform(0.0, 0.5, n_samples), rng.uniform(0.2, 1.0, n_samples)
        MA, MB, sigma = simulate.gillespie_pairs(pair[1], M0.astype(float), t_a, t_a + dt, rng)
        seen.append(datasets.Group(pair[0], MA, MB, dt, sigma))
        true.append(datasets.Group(pair[1], MA, MB, dt, sigma))

    return seen, true




def main(root="results", iters=3000, seeds=2, model_names=("pgnn", "pgnn+", "npf-unit", "npf-learned-projection-only", "npf-learned", "npf-learned-flat", "npf-oracle")):
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
            model = (SheafNPF(consistency=name != "npf-learned-projection-only", flat=name == "npf-learned-flat") if name.startswith("npf-learned")
                     else models.build(name.replace("-unit", "").replace("-oracle", ""), "transitions")).to(device)
            view = true if name == "npf-oracle" else seen
            train(model, "transitions", view, device, iters, seed)
            model.eval()
            row = {}
            for split in ("test", "test-large"):
                # always scored on the true net
                m = transition_metrics(true[split], predict(model, view[split], device))
                row[split] = {k: m[k] for k in ("rmse", "mae", "rmse_row", "consistent", "unexplained_tokens")}

            if name.startswith("npf-learned"):
                learned = model.log_ratio.detach().exp().cpu().numpy()
                off = ~np.eye(N_TYPES, dtype=bool)
                row["ratio_relative_error"] = float(np.abs(learned[off] / true_ratio[off] - 1).max())

                # R[a,b] R[b,c] / R[a,c] = 1 without arbitrage
                cycle = learned[:, :, None] * learned[None, :, :] / learned[:, None, :]
                row["arbitrage_gap"] = float(np.abs(cycle[off][:, :][np.isfinite(cycle[off])] - 1).max())
                row["learned_ratios"] = learned.round(3).tolist()

            out.setdefault(name, []).append(row)
            print(f"{name:>12} seed {seed}: " + "  ".join(f"{k}: rmse {v['rmse']:.3f} takes A to B {v['consistent']:.3f}" for k, v in row.items() if isinstance(v, dict))
                  + (f"  max ratio error {row['ratio_relative_error']:.2%}" if "ratio_relative_error" in row else ""), flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys
    main(**({"model_names": tuple(sys.argv[1:])} if len(sys.argv) > 1 else {}))
