"""Fit the one decode-time scalar of the token game, a constant on the log rate of STOP, on validation reactions.

The error analysis shows that the greedy game stops early five times as often as it stops late. The constant is
chosen on validation reactions that played no part in training or model selection, and the test set is scored once
with the chosen value.

    uv run python -m benchmarks.chemistry.calibrate results/uspto_mit/forward/npf-deep-0.pt --dataset uspto_mit --width 256 --rounds 8 --attention 8
"""
import argparse
import json
from pathlib import Path

import torch

from npflow import chem

from .experiment import USPTO_MIT_TEST_LINES, batches, product_found, splits

GRID = (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0)


@torch.no_grad()
def accuracy(model, reactions, device):
    hits = 0

    for rs in batches(reactions, 64):
        b = chem.collate(rs, device)
        hits += sum(product_found(r, pred) for r, pred in zip(rs, model.decode(model(b), b, rs)))

    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--dataset", default="schneider50k")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--attention", type=int, default=4)
    ap.add_argument("--validation", type=int, default=6000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val, test = splits(chem.load(f"data/{args.dataset}.pkl"), "forward")

    # the first 1500 validation reactions chose the epoch, so they are left out here
    held_out = val[1500:1500 + args.validation]
    model = chem.TokenGame(d=args.width, rounds=args.rounds, attention=args.attention).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    curve = {}
    for bias in GRID:
        model.stop_bias = bias
        curve[bias] = accuracy(model, held_out, device) / len(held_out)
        print(f"stop bias {bias:+.1f}: validation top-1 {curve[bias]:.4f}", flush=True)

    best = max(curve, key=curve.get)
    n = USPTO_MIT_TEST_LINES if args.dataset == "uspto_mit" else len(test)
    result = {"validation": curve, "chosen": best, "n_validation": len(held_out), "n_test": n}
    for name, bias in (("product_top1_uncalibrated", 0.0), ("product_top1_calibrated", best)):
        model.stop_bias = bias
        result[name] = accuracy(model, test, device) / n
        print(f"{name} (stop bias {bias:+.1f}): {result[name]:.4f} of {n} test reactions", flush=True)

    out = Path(args.weights).parent.parent / "calibration" / (Path(args.weights).stem + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
