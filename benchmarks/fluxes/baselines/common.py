"""Shared pieces of the baselines. The reaction view of a culture net, the linear and quadratic programs on it, and the
loop over the splits of flux.py that writes its result files.

A culture net has one transition per direction of a reaction, N maps firing counts to the net flux of every reaction,
and every solution below satisfies C sigma = m_B - m_A with sigma >= 0 unless the method says otherwise. The ATP
maintenance bound follows pfba in flux.py, 8.39 where the culture allows it and 0 otherwise.
"""

import argparse
import json
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from .. import data, flux

TAKEN = {"pfba", "mean", "se-only", "npf", "pgnn", "pgnn+", "pgnn+se"}
SPLITS = ("cv", "dataset", "carbon")

# flux bound of the core model in either direction
BIG = 1000.0


def parser(doc):
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("--split", choices=SPLITS + ("all",), default="cv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default="results/metabolism")

    return ap


def key(inst):
    return f"{inst.culture.dataset}/{inst.culture.name}"


def reactions(inst):
    """Reaction ids of the net in order of first appearance and N with N sigma the net flux of each."""
    ids = list(dict.fromkeys(inst.reaction))
    index = {r: k for k, r in enumerate(ids)}
    N = np.zeros((len(ids), inst.net.n_trans))
    N[[index[r] for r in inst.reaction], np.arange(inst.net.n_trans)] = inst.sign

    return ids, N


def net_flux(inst, sigma):
    out = {}
    for r, s, x in zip(inst.reaction, inst.sign, sigma):
        out[r] = out.get(r, 0.0) + s * float(x)

    return out


def to_sigma(inst, v):
    """Firing counts from net fluxes, each direction takes its part and a missing reaction does not fire. The drain of
    the maintenance place takes what ATPM fires above the bound."""
    v = dict(v, ATPM_drain=v.get("ATPM", 0.0) - inst.atpm)

    return np.array([max(s * v.get(r, 0.0), 0.0) for r, s in zip(inst.reaction, inst.sign)])


def signed_sigma(inst, v):
    """Signed counts with N sigma = v exactly, the whole net flux on the first transition of each reaction.

    For a learner that ignores the net. W sigma is then its prediction and C sigma its own flux balance, even where
    it runs a reaction against its direction, which no firing count could do.
    """
    sigma, seen = np.zeros(inst.net.n_trans), set()
    v = dict(v, ATPM_drain=v.get("ATPM", 0.0) - inst.atpm)
    for t, (r, s) in enumerate(zip(inst.reaction, inst.sign)):
        if r not in seen:
            sigma[t] = s * v.get(r, 0.0)
            seen.add(r)

    return sigma


def cancel(inst, sigma):
    """Removes the futile firing of both directions of a reaction, the two columns of C cancel so C sigma stays."""
    return to_sigma(inst, net_flux(inst, sigma))


def rhs(inst):
    return inst.m_b - inst.m_a


def lower(inst, atpm):
    lo = np.zeros(inst.net.n_trans)
    lo[[t for t, r in enumerate(inst.reaction) if r == "ATPM"]] = atpm

    return lo


def maintained(inst, solve):
    """solve(lower bounds) under ATP maintenance, which the maintenance place of the net already forces, at the core
    bound where the culture allows it and 0 otherwise (data.instances)."""
    return solve(lower(inst, 0.0))


def lp(c, lo, A_eq=None, b_eq=None, A_ub=None, b_ub=None, hi=None):
    n = len(c)
    hi = [None] * n if hi is None else hi
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lo, hi)), method="highs")

    return res.x if res.status == 0 else None


def tie(value, rel=1e-7):
    """Slack on an optimum that a second stage keeps."""
    return value + rel * max(1.0, abs(value))


def least_total(lo, A_eq, b_eq, A_ub=None, b_ub=None, n=None):
    """Among the solutions of a first stage, the one with the least total firing, the tie rule of pFBA."""
    n = len(lo) if n is None else n
    c = np.zeros(len(lo))
    c[:n] = 1.0

    return lp(c, lo, A_eq, b_eq, A_ub, b_ub)


def project(inst, A, target, lo, alpha=1e-4, weights=None):
    """argmin ||A sigma - target||^2 + alpha sum(sigma) on the flux balance of the culture, a convex QP.

    The small linear term picks the least total firing among the equally close solutions. Firing is capped at the
    flux bound of the core model, 1000, which keeps far off targets well posed. Needs cvxpy.
    """
    import cvxpy as cp

    x = cp.Variable(inst.net.n_trans)
    w = np.ones(len(target)) if weights is None else weights
    objective = cp.sum_squares(cp.multiply(np.sqrt(w), A @ x - target)) + alpha * cp.sum(x)
    problem = cp.Problem(cp.Minimize(objective), [inst.net.C @ x == rhs(inst), x >= lo, x <= BIG])
    settings = (
        ("CLARABEL", {}),
        ("OSQP", {"max_iter": 200000, "eps_abs": 1e-9, "eps_rel": 1e-9}),
        ("SCS", {"max_iters": 200000, "eps_abs": 1e-9, "eps_rel": 1e-9}),
    )
    for solver, options in settings:
        try:
            problem.solve(solver=solver, **options)
        except cp.error.SolverError:
            continue

        if problem.status in ("optimal", "optimal_inaccurate"):
            return np.maximum(x.value, lo)

    return None


@lru_cache(maxsize=1)
def model():
    return data.core()


def carbons(met):
    formula = {m["id"]: m.get("formula", "") for m in model()["metabolites"]}[met]
    hit = re.search(r"C(\d*)(?![a-z])", formula)

    return 0 if hit is None else int(hit.group(1) or 1)


def substrate_scale(inst):
    """Carbon uptake of the culture in glucose equivalents, the unit of fluxes relative to uptake."""
    reactions_by_id = {r["id"]: r for r in model()["reactions"]}
    total = 0.0

    for rid, rate in inst.culture.pinned.items():
        if rid.startswith("EX_") and rate < 0:
            (met,) = reactions_by_id[rid]["metabolites"]
            total += -rate * carbons(met) / 6.0

    return total


def summary_of(rows):
    out = {
        k: float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None
        for k in ("rel_error", "rel_error_open", "balance_l1", "wrong_direction")
    }
    out["median_rel_error"] = float(np.median([r["rel_error"] for r in rows])) if rows else None

    return out


def evaluate(method, args, fit, note="", training_free=False):
    """Runs fit(train) -> predict(inst) -> sigma or None on every fold of the chosen splits and writes the results.

    A culture whose prediction is None is left out of the file and listed under skipped. A predict function may carry
    a dict info with what was chosen on the training folds, stored per fold, its params entry counts parameters.
    """
    assert method not in TAKEN, method
    insts, dropped = data.instances()
    cache = {}
    for split in SPLITS if args.split == "all" else (args.split,):
        rows, skipped, folds, start = [], [], [], time.time()

        # the folds of flux.py, which are fixed whatever the seed of a learner
        for train, test in flux.splits(insts, split):
            predict = fit(train)
            folds.append(getattr(predict, "info", {}))

            for inst in test:
                if training_free and key(inst) in cache:
                    sigma = cache[key(inst)]
                else:
                    sigma = predict(inst)
                    cache[key(inst)] = sigma

                if sigma is None:
                    skipped.append(key(inst))
                    continue

                rows.append(flux.metrics(inst, np.asarray(sigma, float)))

        if not rows:
            print(f"{split} {method}: no culture to score, nothing written", flush=True)
            continue

        summary = summary_of(rows)
        result = {
            "model": method, "split": split, "seed": args.seed, "iters": 0,
            "params": int(np.mean([f.get("params", 0) for f in folds])) if folds else 0,
            "seconds": time.time() - start, "cultures": len(rows), "dropped": [key(i) for i in dropped],
            "skipped": skipped, "note": note, "folds": folds, "summary": summary, "curves": [], "rows": rows,
        }
        out = Path(args.root) / split / f"{method}-{args.seed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))
        print(f"{split} {method} seed {args.seed}: " + "  ".join(
            f"{k}={v:.4g}" for k, v in summary.items() if v is not None) + f"  ({len(rows)} cultures, "
              f"{len(skipped)} skipped, {result['seconds']:.0f}s)", flush=True)
