"""Atom maps computed from the net alone, with no recorded mapping anywhere.

For a reaction the two markings are observed and every injection of the product atoms into the precursor atoms gives
a firing vector with m_B = m_A + C sigma. The cheapest such firing vector is a property of the net, so it can replace
the recorded mapping as the source of training targets. The file it writes has the same format as the one that a
trained mapper writes, so every head reads it with the existing switch.

    uv run python -m benchmarks.chemistry.net_maps                      # data/net_maps_schneider50k.pkl
    uv run python -m benchmarks.chemistry.net_maps --dataset uspto_mit  # data/net_maps_uspto_mit.pkl
"""
import argparse
import pickle
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from npf import chem

from .experiment import token_descent

BUDGET = 20000


def seed_map(reaction):
    """A feasible seating that uses only the element and the atom environment, never a recorded map."""
    doms = chem.domains(reaction)
    out, used = np.zeros(len(reaction["b"]["x"]), np.int64), set()
    for i, d in enumerate(doms):
        free = [int(j) for j in d if int(j) not in used]
        if not free:
            free = [int(j) for j in np.nonzero(reaction["a"]["element"] == reaction["b"]["element"][i])[0] if int(j) not in used]

        if not free:
            return None

        out[i] = free[0]
        used.add(free[0])

    return out


def _work(reaction):
    # products larger than the precursors cannot be seated, the record is incomplete
    if len(reaction["b"]["x"]) > len(reaction["a"]["x"]):
        return reaction["id"], None, False

    try:
        start = seed_map(reaction)
        if start is None:
            return reaction["id"], None, False

        best, _, exact = chem.branch_and_bound(reaction, token_descent(reaction, start), budget=BUDGET)
        return reaction["id"], best.astype(np.int16), exact
    except Exception:
        return reaction["id"], None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="schneider50k")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    reactions = chem.load(f"data/{args.dataset}.pkl")["reactions"]
    start = time.time()

    with Pool(args.workers) as pool:
        rows = pool.map(_work, reactions, chunksize=32)

    maps = {rid: m for rid, m, _ in rows if m is not None}
    out = args.out or f"data/net_maps_{args.dataset}.pkl"
    Path(out).write_bytes(pickle.dumps(maps))
    stats = Counter()
    for _, m, exact in rows:
        stats["seated"] += m is not None
        stats["exhaustive"] += exact

    n = len(rows)
    print(f"{n} reactions, {stats['seated'] / n:.2%} seated, search exhaustive for {stats['exhaustive'] / n:.2%}, "
          f"{time.time() - start:.0f}s, wrote {out}")

    # how often the net agrees with the record, reported only as a diagnostic, never used for training
    agree = total = 0

    for r in reactions:
        if r["target"] is not None and r["id"] in maps:
            total += 1
            agree += np.array_equal(maps[r["id"]].astype(np.int64), r["target"].astype(np.int64))

    if total:
        print(f"agrees with the recorded mapping for {agree / total:.2%} of the {total} reactions that have one")


if __name__ == "__main__":
    main()
