"""Validity for all weights on the FlowER mechanism benchmark: untrained arrow and electron nets, with and without the
octet rule as enabling, predict test steps with the beam search of the evaluation (width 1).

Two rollouts per model. "random" keeps the initial STOP head, so an untrained game stops wherever its random rate says,
"forced" sets the STOP bias to -30, so STOP wins only when no arrow is enabled and the game fires until the arrow budget
or the enabling rule stops it. Every end marking is checked for a valid molecule (its scoring form exists, the valid
SMILES criterion of the evaluation), the octet capacities, whole charges and, on the electron net, whole pairs on every
bond place. A game that ends with no event enabled has no prediction, it is counted apart and not as invalid.

    uv run python -m benchmarks.chemistry.mechanism_validity                # results/mechanism/validity.json
    uv run python -m benchmarks.chemistry.mechanism_validity --steps 100 --seeds 1 --nets electron
"""

import argparse
import json
import random

import numpy as np
import torch

from . import mechanism as M


def checks(m, arrows, octet, window):
    net = M.net()
    end = net.apply(m, arrows)
    cap, _, _ = net.limits(m, octet, window)
    shell = net.shell(end)
    pairs = shell if "q" in end else np.ceil(shell / 2)

    try:
        valid = net.scoring_form(net.to_mol(end)) is not None
    except Exception:
        valid = False

    return {
        "valid": valid,
        "octet": bool((pairs <= cap).all()),
        "whole_charges": True if "q" in end else bool((end["q2"] % 2 == 0).all()),
        "paired_bonds": True if "q" in end else all(e % 2 == 0 for e in end["be"].values()),
    }


@torch.no_grad()
def rollout(data, index, seed, enabling, forced, device, batch=8):
    torch.manual_seed(seed)
    model = M.NET["game"](enabling=enabling).to(device).eval()

    if forced:
        model.stop[-1].bias.fill_(-30.0)

    rows = []
    for k0 in range(0, len(index), batch):
        ks = index[k0 : k0 + batch]
        items = []
        for k in ks:
            m = data.marking(k)
            cap, lo, hi = M.net().limits(m, data.octet, data.window)
            fixed = {"cap": cap, "lo": lo, "hi": hi}
            items.append(
                {name: fixed.get(name, m.get(name)) for name in M.NET["state"]}
                | {M.NET["bonds"]: m[M.NET["bonds"]]}
            )

        t = {name: v.to(device) for name, v in M.collate(items).items()}
        for k, found in zip(ks, model.beam(t, width=1, top=1)):
            rows.append(
                checks(data.marking(k), found[0][0], data.octet, data.window)
                if found
                else None
            )

    ended = [r for r in rows if r is not None]
    share = lambda name: float(np.mean([r[name] for r in ended])) if ended else None

    return {"ended": len(ended) / len(rows)} | {
        name: share(name) for name in ("valid", "octet", "whole_charges", "paired_bonds")
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="arrow,electron")
    ap.add_argument("--steps", type=int, default=1000, help="random test steps that an enabled order covers")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for name in args.nets.split(","):
        M.NET = M.NETS[name]
        data = M.Steps("test")
        covered = [k for k in range(len(data)) if data.a["status"][k] == M.OK]
        index = sorted(random.Random(0).sample(covered, args.steps))

        for enabling in (True, False):
            for forced in (False, True):
                key = f"{name}, {'octet rule' if enabling else 'no enabling'}, {'forced' if forced else 'random'} STOP"
                runs = [rollout(data, index, s, enabling, forced, device) for s in range(args.seeds)]
                out[key] = {k: [r[k] for r in runs] for k in runs[0]}
                print(key, out[key], flush=True)

    M.RESULTS.mkdir(parents=True, exist_ok=True)
    (M.RESULTS / "validity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
