"""Forward training targets from the net alone, with no recorded atom map anywhere.

For a training reaction exact.cheapest_mappings lists every mapping of minimum cost, and each gives a firing vector
with m_B = m_A + C sigma. A vector is kept when it moves at most MAX_TOKENS tokens, has an enabled order and decodes
to the recorded product, so the target set is a function of the precursors, the product and the net. Vectors that
differ by a symmetry are all kept and the loss sums over the set. Without a proof in time the cheapest vector found
stands alone.

    uv run python -m benchmarks.chemistry.net_targets --dataset uspto_mit --subset 40900   # data/net_targets_uspto_mit-sub40900.pkl
    uv run python -m benchmarks.chemistry.net_targets --dataset uspto_mit                  # data/net_targets_uspto_mit.pkl
    uv run python -m benchmarks.chemistry.net_targets --dataset schneider50k               # data/net_targets_schneider50k.pkl
"""

import argparse
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from npf import chem
from npf.chem import exact

from .experiment import splits, subset_lines


def targets(job):
    """One reaction of the pool, (reaction, seconds, limit)."""
    return chem.targets(*job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="uspto_mit")
    ap.add_argument(
        "--subset",
        type=int,
        help="only the reactions of this USPTO-MIT subset (experiment --subset)",
    )
    ap.add_argument(
        "--sample",
        type=int,
        help="a random sample of this many reactions, a timing run that writes nothing",
    )
    ap.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="deterministic solver time for the proof and again for listing the ties",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=64,
        help="most optimal mappings listed per reaction",
    )
    ap.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out")
    args = ap.parse_args()

    # workers start before the data is loaded, so they do not hold a copy of it
    pool = Pool(args.processes)
    train = splits(chem.load(f"data/{args.dataset}.pkl"), "forward", recorded=False)[0]
    name = args.dataset + (f"-sub{args.subset}" if args.subset else "")

    if args.subset:
        chosen = set(subset_lines(args.subset).tolist())
        train = [r for r in train if r["id"] in chosen]

    if args.sample:
        train = [
            train[i]
            for i in np.random.default_rng(0).choice(
                len(train), args.sample, replace=False
            )
        ]

    out = Path(args.out or f"data/net_targets_{name}.pkl")
    done = (
        pickle.loads(out.read_bytes())
        if out.exists() and not args.sample
        else {"targets": {}, "rows": {}}
    )
    todo = [r for r in train if r["id"] not in done["rows"]]
    print(
        f"{len(train)} training reactions, {len(todo)} to solve with {args.processes} processes",
        flush=True,
    )
    start, config = time.time(), {
        "options": exact.CHOSEN,
        "seconds": args.seconds,
        "limit": args.limit,
        "max_vectors": chem.MAX_VECTORS,
    }

    def save():
        if not args.sample:
            out.write_bytes(pickle.dumps(done | {"config": config}))

    for k, row in enumerate(
        pool.imap_unordered(
            targets, [(r, args.seconds, args.limit) for r in todo], chunksize=1
        )
    ):
        done["rows"][row["id"]] = {key: v for key, v in row.items() if key != "vectors"}

        if row["vectors"]:
            done["targets"][row["id"]] = row["vectors"]

        if (k + 1) % 1000 == 0:
            rows = list(done["rows"].values())
            print(
                f"{k + 1}/{len(todo)} with targets {len(done['targets']) / len(rows):.4f} proved {np.mean([x['proved'] for x in rows]):.4f} "
                f"{time.time() - start:.0f}s",
                flush=True,
            )

        if (k + 1) % 20000 == 0:
            save()

    pool.close()
    save()
    rows = list(done["rows"].values())
    counts = [len(done["targets"].get(x["id"], [])) for x in rows]
    print(
        f"{len(rows)} reactions in {time.time() - start:.0f}s, mean {np.mean([x['seconds'] for x in rows]):.2f}s median "
        f"{np.median([x['seconds'] for x in rows]):.2f}s per reaction"
    )
    print(
        f"proved {np.mean([x['proved'] for x in rows]):.4f}, with targets {np.mean([c > 0 for c in counts]):.4f}, "
        f"vectors per reaction with targets {np.mean([c for c in counts if c]):.2f}, listing hit the limit "
        f"{np.mean([x['mappings'] >= args.limit for x in rows]):.4f}, errors {sum('error' in x for x in rows)}"
    )

    # agreement with the record, a diagnostic that is printed and never used for training
    if args.sample:
        by_id = {r["id"]: r for r in train}
        recorded = [
            r
            for r in (by_id[x["id"]] for x in rows)
            if r["target"] is not None and r["edits"] is not None
        ]
        among = [
            any(np.array_equal(v, r["edits"]) for v in done["targets"].get(r["id"], []))
            for r in recorded
        ]
        shorter = [
            min(len(v) for v in done["targets"][r["id"]]) < len(r["edits"])
            for r in recorded
            if r["id"] in done["targets"]
        ]
        print(
            f"recorded vector among the targets {np.mean(among):.4f}, a target shorter than the record {np.mean(shorter):.4f}"
        )


if __name__ == "__main__":
    main()
