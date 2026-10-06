"""EnzymeMap (Heid et al., Chem. Sci. 2023), the single-product enzymatic reactions with their rule-derived atom maps,
as a benchmark file for the exact mapper. The maps come from the Broadbelt rule set, so agreement with them measures
whether the minimum firing vector reproduces a rule-based mapping, not a curated one.

    uv run python -m benchmarks.chemistry.enzymemap prepare               # data/enzymemap.pkl, data/enzymemap_unmapped.txt
    uv run python -m benchmarks.chemistry.enzymemap prepare --sample 3000  # data/enzymemap_3k.pkl, a random subset first
    uv run python -m benchmarks.chemistry.exact_map --data enzymemap        # results/chem/exact_map/enzymemap-*.json
    uv run --no-project --python 3.11 --with rxnmapper --with rdkit --with "setuptools<81" --with "numpy<2" \\
        python benchmarks/chemistry/baselines/rxnmapper_golden.py data/enzymemap_unmapped.txt results/chem/rxnmapper_enzymemap.json
    uv run python -m benchmarks.chemistry.exact_map_report results/chem/exact_map/enzymemap-1-1-1-nolabile-ch.pkl results/chem/rxnmapper_enzymemap.json

The files come from github.com/hesther/enzymemap, data/brenda_singleprod.csv and data/brendadirect_singleprod.csv.
"""
import argparse
import csv
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from npflow import chem

FOLDER = Path("data/enzymemap")


def featurise(job):
    k, mapped, plain, direct = job

    try:
        r = chem.featurise({"original_rxn": mapped, "rxn": plain, "label": int(direct), "split": "test", "id": k}, max_atoms=None)
    except Exception:
        return None

    return r


def prepare(sample=None, seed=0):
    rows = list(csv.DictReader(open(FOLDER / "brenda_singleprod.csv")))
    direct = {row["rxn_smiles"] for row in csv.DictReader(open(FOLDER / "brendadirect_singleprod.csv"))}

    # the file lists a reaction once per source, the same reaction counts once
    seen, jobs = set(), []
    for row in rows:
        plain = f"{row['reac_smiles']}>>{row['prod_smiles']}"
        if plain not in seen:
            seen.add(plain)
            jobs.append((len(jobs), row["rxn_smiles"], plain, row["rxn_smiles"] in direct))

    if sample:
        jobs = [jobs[i] for i in np.random.default_rng(seed).choice(len(jobs), sample, replace=False)]

    with Pool(12) as pool:
        reactions = [r for r in pool.map(featurise, jobs, chunksize=64) if r is not None]

    name = f"enzymemap_{sample // 1000}k" if sample else "enzymemap"
    Path(f"data/{name}.pkl").write_bytes(pickle.dumps({"reactions": reactions, "classes": ["-"]}))
    Path(f"data/{name}_unmapped.txt").write_text("\n".join(f"{r['id']}\t{r['smiles']}" for r in reactions))
    mapped = sum(r["target"] is not None for r in reactions)
    print(f"{len(rows)} rows, {len(jobs)} distinct reactions, {len(reactions)} featurised, {mapped} with a map we can read, "
          f"{sum(r['label'] for r in reactions)} from direct entries, wrote data/{name}.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare"])
    ap.add_argument("--sample", type=int, help="a random subset of this many reactions")
    args = ap.parse_args()
    prepare(args.sample)
