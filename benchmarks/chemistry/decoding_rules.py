"""Top-1 accuracy of one trained token game under the three generations of the rules that decode a marking into
molecules (npf.chem.decode.RECONSTRUCTION). The firings are the same, only the reading of slack tokens as hydrogens
and charges differs. The ceiling is the share of test reactions whose recorded firings decode to the recorded product.

    uv run python -m benchmarks.chemistry.decoding_rules results/uspto_mit/forward/npf-deep-0.pt --dataset uspto_mit --width 256 --rounds 8 --attention 8
"""
import argparse
import json
from pathlib import Path

import torch

from npf import chem
from npf.chem import decode

from .experiment import USPTO_MIT_TEST_LINES, batches, product_found, splits


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--dataset", default="schneider50k")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--attention", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test = splits(chem.load(f"data/{args.dataset}.pkl"), "forward")
    model = chem.TokenGame(d=args.width, rounds=args.rounds, attention=args.attention).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    predictions = []

    for rs in batches(test, 64):
        b = chem.collate(rs, device)
        predictions += list(zip(rs, model.decode(model(b), b, rs)))

    n = USPTO_MIT_TEST_LINES if args.dataset == "uspto_mit" else len(test)
    result = {"n": n}
    for version in (1, 2, 3):
        # the switch is a global of the decode module
        decode.RECONSTRUCTION = version
        top1 = sum(product_found(r, pred) for r, pred in predictions)
        ceiling = sum(r["edits"] is not None and product_found(r, r["edits"]) for r, _ in predictions)
        result[f"rules_{version}"] = {"product_top1": top1 / n, "ceiling": ceiling / n}
        print(f"decoding rules {version}: top-1 {top1 / n:.4f}, ceiling {ceiling / n:.4f}  ({n} reactions in the denominator)", flush=True)

    decode.RECONSTRUCTION = 3
    out = Path(args.weights).parent.parent / "decoding_rules" / (Path(args.weights).stem + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
