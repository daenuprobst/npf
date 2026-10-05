"""Metabolic flux inference, one model on one split, scored on the measured 13C fluxes of held-out cultures.

Every model receives the same net and the same two markings, which carry the measured exchange rates, the growth rate,
the ATP maintenance and the knockouts, and predicts the firing counts of all transitions over one hour, the fluxes in
mmol/gDW/h. The learned models are those of npf.models. On this net the transitions are the same in every culture, so
the -k and -gma variants give NPF and PGNN alike a rate constant per transition and, for -gma, the boundary of the
culture as read arcs. Every projection is the I projection, whose output is the positive firing count vector of a real
run, sigma > 0 with C sigma = m_B - m_A, unless --gauss keeps the Gaussian projection, which meets the state equation
but not sigma >= 0. pfba minimises the summed flux on the same net, the standard constraint-based baseline, and se-only
is the state equation alone, the maximum entropy counts of the fibre.

    uv run python -m benchmarks.fluxes.flux --model npf-gma --split cv --seed 0

Splits. cv is five folds over the glucose cultures, dataset holds out one study at a time and trains on the others,
carbon trains on every glucose culture and tests on the cultures of Gerosa et al. on other carbon sources. The folds
are the same for every seed and every method.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog

from npf import batching, models
from npf.datasets import Group

from . import data

HELD_OUT = ("ishii2007", "long2019", "haverkorn2011")

# the I projection needs a wider range of the dual exponent and more Newton steps than its defaults on these nets
KL = dict(low=-60.0, high=30.0, n_iter=60)


def splits(insts, split, seed=0):
    """Train and test instances of every fold."""
    glucose = [i for i in insts if i.culture.carbon == "glucose"]

    if split == "cv":
        order = np.random.default_rng(seed).permutation(len(glucose))
        folds = np.array_split(order, 5)

        return [
            ([glucose[j] for j in order if j not in set(f)], [glucose[j] for j in f]) for f in folds
        ]

    if split == "dataset":
        return [
            ([i for i in glucose if i.culture.dataset != d], [i for i in glucose if i.culture.dataset == d])
            for d in HELD_OUT
        ]

    return [(glucose, [i for i in insts if i.culture.carbon != "glucose"])]


def environment(inst):
    """The boundary of a culture as seen through read arcs, its marking change per unit of total exchange and the log
    of that total, the same vector for every model that reads it."""
    _, places, boundary = data.vocabulary()
    dm = (inst.m_b - inst.m_a)[[places.index(p) for p in boundary]]
    scale = np.abs(dm).sum()

    return np.append(dm / scale, np.log(scale))


def group(inst):
    return Group(inst.net, m=inst.m_a[None], m_b=inst.m_b[None], dt=np.ones(1), t_id=inst.t_id,
                 env=environment(inst)[None])


def run(model, insts, device):
    """Firing counts of every instance."""
    batch = batching.collate([group(i) for i in insts], [np.zeros(1, int)] * len(insts), device)
    flat = model(batch)

    return torch.split(flat, [i.net.n_trans for i in insts])


def loss_of(model, insts, device):
    total = 0.0
    for inst, sigma in zip(insts, run(model, insts, device)):
        W = torch.as_tensor(inst.W, dtype=torch.float32, device=device)
        y = torch.as_tensor(inst.y, dtype=torch.float32, device=device)
        total = total + ((W @ sigma - y) ** 2).mean()

    return total / len(insts)


def metrics(inst, sigma):
    """Relative error on the measurements and on those the markings leave open, the flux balance residual, and how far
    the output is from a firing count vector."""
    pred = inst.W @ sigma
    open_ = ~data.identified(inst)
    rel = lambda mask: float(np.linalg.norm(pred[mask] - inst.y[mask]) / max(np.linalg.norm(inst.y[mask]), 1e-9))

    # a direction is wrong when prediction and measurement differ in sign on a flux above 5 % of the carbon uptake
    scale = 0.05 * max(abs(inst.culture.pinned.get("EX_glc__D_e", 0.0)), 1.0)
    big = np.abs(inst.y) > scale

    return {
        "culture": f"{inst.culture.dataset}/{inst.culture.name}",
        "rel_error": rel(np.ones(len(pred), bool)),
        "rel_error_open": rel(open_),
        "balance_l1": float(np.abs(inst.net.C @ sigma - (inst.m_b - inst.m_a)).sum()),
        "negative_mass": float(np.clip(-sigma, 0, None).sum() / max(np.abs(sigma).sum(), 1e-9)),
        "min_sigma": float(sigma.min()),
        "wrong_direction": int((np.sign(pred[big]) != np.sign(inst.y[big])).sum()),
        "n_measured": int(len(pred)),
        "n_open": int(open_.sum()),
        "pred": pred.tolist(),
        "measured": inst.y.tolist(),
        "ids": inst.ids,
    }


def pfba(inst):
    """Least total flux on the net, whose maintenance place already holds ATPM at its bound."""
    n = inst.net.n_trans
    res = linprog(np.ones(n), A_eq=inst.net.C, b_eq=inst.m_b - inst.m_a, bounds=(0, None), method="highs")

    return res.x if res.status == 0 else np.zeros(n)


def train(model, insts, device, iters, seed, log_every=100, batch=32):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(insts))
    n_val = max(2, len(insts) // 7)
    val, fit = [insts[i] for i in order[:n_val]], [insts[i] for i in order[n_val:]]
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=2e-5)
    best, best_state, curve = np.inf, None, []
    for it in range(1, iters + 1):
        chosen = [fit[i] for i in rng.choice(len(fit), min(batch, len(fit)), replace=False)]
        loss = loss_of(model, chosen, device)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % log_every == 0 or it == iters:
            model.eval()

            with torch.no_grad():
                sigmas = [s.double().cpu().numpy() for s in run(model, val, device)]

            score = float(np.mean([metrics(i, s)["rel_error"] for i, s in zip(val, sigmas)]))
            model.train()
            curve.append((it, loss.item(), score))

            if score < best:
                best, best_state = score, copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return curve


def summarise(rows):
    """Means over cultures, the median, and the mean of the per-study means, so that no study dominates."""
    keys = ("rel_error", "rel_error_open", "balance_l1", "negative_mass", "wrong_direction")
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["median_rel_error"] = float(np.median([r["rel_error"] for r in rows]))
    studies = sorted({r["culture"].split("/")[0] for r in rows})
    per_study = {s: float(np.mean([r["rel_error"] for r in rows if r["culture"].startswith(s + "/")])) for s in studies}
    out["macro_rel_error"] = float(np.mean(list(per_study.values())))
    out["per_study"] = per_study

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="npf, npf-k, npf-gma, npf-gma-row, pgnn, pgnn-gma, pgnn+, pgnn+se, pgnn+se-gma, se-only or pfba")
    ap.add_argument("--split", choices=("cv", "dataset", "carbon"), default="cv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--gauss", action="store_true", help="the Gaussian projection, sigma >= 0 not held")
    ap.add_argument("--root", default="results/metabolism")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-cap", action="store_true", help="run a seed above the cap of <root>/max_seed")
    args = ap.parse_args()

    # a queue that is already running reads the highest seed it may still start from this file, so a grid is cut
    # short without stopping it
    cap = Path(args.root) / "max_seed"
    if not args.no_cap and cap.exists() and args.seed > int(cap.read_text()):
        print(f"seed {args.seed} is above the cap in {cap}, not run")
        return

    torch.set_num_threads(args.threads)
    device = args.device
    insts, dropped = data.instances()
    ids, _, boundary = data.vocabulary()
    rows, curves, n_params, start = [], [], 0, time.time()
    for k, (train_set, test_set) in enumerate(splits(insts, args.split)):
        if args.model == "pfba":
            rows += [metrics(i, pfba(i)) for i in test_set]
            continue

        torch.manual_seed(args.seed * 100 + k)
        model = models.build(args.model, "transitions", n_ids=len(ids), n_env=len(boundary) + 1,
                             kl_options=None if args.gauss else KL).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if n_params:
            curves.append(train(model, train_set, device, args.iters, args.seed * 100 + k))

        model.eval()

        with torch.no_grad():
            sigmas = [s.double().cpu().numpy() for s in run(model, test_set, device)]

        rows += [metrics(i, s) for i, s in zip(test_set, sigmas)]

    name = args.model + ("-gauss" if args.gauss else "")
    summary = summarise(rows)
    result = {
        "model": name, "split": args.split, "seed": args.seed, "params": n_params, "iters": args.iters,
        "seconds": time.time() - start, "cultures": len(rows),
        "dropped": [f"{i.culture.dataset}/{i.culture.name}" for i in dropped],
        "reconciled": {f"{i.culture.dataset}/{i.culture.name}": [i.culture.pinned[data.BIOMASS], i.growth]
                       for i in insts if i.growth < i.culture.pinned[data.BIOMASS] - 1e-9},
        "summary": summary, "curves": curves, "rows": rows,
    }
    out = Path(args.root) / args.split / f"{name}-{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"{args.split} {name} seed {args.seed}: " + "  ".join(
        f"{k}={v:.4g}" for k, v in summary.items() if isinstance(v, float)) + f"  ({n_params} params, "
          f"{result['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
