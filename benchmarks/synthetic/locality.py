"""Locality lower bound, P3 of THEORY.md. Message passing cannot replace the state-equation projection.

The chain net p_0 -> p_1 -> ... -> p_L has no T-invariants, so the state equation fixes the firing counts between two
states, sigma_k = -sum_{j<k} dM_j, but only through sums along the whole chain. Models are trained on the usual random
nets with 6 to 12 places and tested on chains of growing length.

    uv run python -m benchmarks.synthetic.locality    # writes results/locality.json
"""

import json
from pathlib import Path

import numpy as np
import torch

from npf import datasets, models, nets, simulate

from .experiment import evaluate_transitions, load_splits, train

LENGTHS = (3, 6, 12, 24, 48)


def chains(length, n_nets=40, n_samples=64, seed=0):
    rng = np.random.default_rng(seed + length)
    groups = []
    for _ in range(n_nets):
        net = nets.chain_net(rng, length + 1)
        M0 = rng.integers(0, 9, size=(n_samples, net.n_places)) * (
            rng.random((n_samples, net.n_places)) > 0.25
        )
        t_a, dt = rng.uniform(0.0, 0.5, n_samples), rng.uniform(0.2, 1.0, n_samples)
        MA, MB, sigma = simulate.gillespie_pairs(net, M0, t_a, t_a + dt, rng)
        groups.append(datasets.Group(net, MA, MB, dt, sigma))

    return groups


def main(root="results", iters=3000, seeds=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = load_splits("transitions", "graph", root)
    tests = {length: chains(length) for length in LENGTHS}
    out = {}
    for name in ("gnn", "pgnn", "pgnn+", "pgnn+se", "npf"):
        for seed in range(seeds):
            torch.manual_seed(seed)
            model = models.build(name, "transitions").to(device)
            train(model, "transitions", splits, device, iters, seed)
            model.eval()
            row = {
                str(length): evaluate_transitions(model, groups, device)
                for length, groups in tests.items()
            }
            row["random graphs (in distribution)"] = evaluate_transitions(
                model, splits["test"], device
            )
            out.setdefault(name, []).append(
                {
                    k: {"rmse": v["rmse"], "consistent": v["consistent"]}
                    for k, v in row.items()
                }
            )
            print(
                f"{name:>8} seed {seed}: "
                + "  ".join(
                    f"L={k[:4]}: {v['rmse']:.3f}" for k, v in out[name][-1].items()
                ),
                flush=True,
            )

    Path(root, "locality.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
