"""Does the token game without the enabling mask stay valence-valid away from its training data.

On the Schneider 50k forward split the game trained without the mask is valence-valid for 99.98 % of the test
reactions, so there the mask adds a guarantee but no measurable validity. Here the Schneider 50k forward models with
and without the mask decode one fixed random sample of USPTO-MIT test reactions greedily, other patents, other
reagents and larger molecules, and every run writes its share of valence-valid products and of found products.

    uv run python -m benchmarks.chemistry.mask_transfer    # writes results/chem/mask_transfer/<model>-<seed>.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from npf import chem

from .experiment import evaluate

# the Schneider 50k forward runs of Table 4, with and without the enabling mask
MODELS = {"npf-nettargets": True, "npf-noenabling-nettargets": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="USPTO-MIT test reactions in the sample")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results/chem/mask_transfer")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = chem.load("data/uspto_mit.pkl")
    test = [r for r in data["reactions"] if r["split"] == "test"]

    # one sample for every model and seed, drawn before any result is seen
    pick = np.sort(np.random.default_rng(0).choice(len(test), args.n, replace=False))
    sample = [test[i] for i in pick]
    del data, test

    for name, enabling in MODELS.items():
        for seed in map(int, args.seeds.split(",")):
            path = out / f"{name}-{seed}.json"

            if path.exists():
                continue

            weights = Path(f"results/chem/forward/{name}-{seed}.pt")
            model = chem.TokenGame(d=128, rounds=6, attention=4, enabling=enabling).to(device)
            model.load_state_dict(torch.load(weights, map_location=device))
            start = time.time()
            metrics = evaluate(model, "forward", sample, device)
            path.write_text(
                json.dumps(
                    {
                        "model": name,
                        "seed": seed,
                        "enabling": enabling,
                        "weights": str(weights),
                        "dataset": "uspto_mit",
                        "n": len(sample),
                        "sample_seed": 0,
                        "seconds": time.time() - start,
                        "metrics": metrics,
                    },
                    indent=1,
                )
            )
            print(f"{name} seed {seed}: {metrics}", flush=True)


if __name__ == "__main__":
    main()
