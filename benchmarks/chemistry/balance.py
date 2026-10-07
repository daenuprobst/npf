"""Balanced equations from the open valence net, for one reaction or for the reactions of a data set that lack atoms.

uv run python -m benchmarks.chemistry.balance "CC(=O)Cl.NCCN>>CC(=O)NCCNC(C)=O"
uv run python -m benchmarks.chemistry.balance --data golden          # results/chem/open_net/golden.json
"""

import argparse
import json
import pickle
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from npflow import chem
from npflow.chem import cost, mapper, open_net


def work(job):
    reaction, seconds = job
    start = time.time()

    try:
        mapping, places, proved, solved = mapper.solve_open(
            reaction,
            seconds=4 * seconds,
            deterministic=seconds,
            **cost.CHOSEN,
        )
    except Exception:
        mapping = None

    if mapping is None:
        return {"id": reaction["id"], "solved": False}

    out = open_net.balance(solved, mapping)
    copies = cost.entered(solved, mapping) - len(out["entered"])

    return {
        "id": reaction["id"],
        "solved": True,
        "proved": bool(proved),
        "cost": places,
        "copies": copies,
        "atoms": len(out["entered"]),
        "balanced": out["balanced"],
        "seconds": time.time() - start,
    }


def one(smiles, seconds):
    r = chem.featurise(
        {"original_rxn": smiles, "rxn": smiles, "label": 0, "split": "test", "id": 0}
    )
    mapping, places, proved, solved = mapper.solve_open(
        r, seconds=4 * seconds, deterministic=seconds, **cost.CHOSEN
    )
    out = open_net.balance(solved, mapping)
    print(
        f"places that change   {places}  ({'proved minimal' if proved else 'not proved'})"
    )
    print(f"equivalents          {out['equivalents']}")
    print(f"by-products          {out['by_products']}")
    print(f"spectators           {out['spectators']}")
    print(f"atoms from outside   {out['entered']}")
    print(f"balanced             {out['balanced']}")


def data_set(name, seconds, processes):
    reactions = pickle.loads(Path(f"data/{name}.pkl").read_bytes())["reactions"]
    short = [r for r in reactions if open_net.deficit(r).any()]

    with Pool(processes) as pool:
        rows = pool.map(work, [(r, seconds) for r in short], chunksize=1)

    done = [x for x in rows if x["solved"]]
    how = Counter(
        (
            "copies and atoms"
            if x["copies"] and x["atoms"]
            else "copies only" if x["copies"] else "atoms only"
        )
        for x in done
    )

    summary = {
        "data": name,
        "reactions": len(reactions),
        "lack_atoms": len(short),
        "solved": len(done),
        "proved": float(np.mean([x["proved"] for x in done])),
        **{k: v / len(done) for k, v in how.items()},
        "median_seconds": float(np.median([x["seconds"] for x in done])),
    }

    out = Path("results/chem/open_net")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(summary, indent=1))
    (out / f"{name}.rows.json").write_text(json.dumps(rows, indent=1))

    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("smiles", nargs="?")
    ap.add_argument("--data")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--processes", type=int, default=4)
    args = ap.parse_args()
    if args.data:
        data_set(args.data, args.seconds, args.processes)
    else:
        one(args.smiles, args.seconds)
