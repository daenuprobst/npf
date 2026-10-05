"""Atom mapping on EnzymeMap under the metric of SynRXN: the mapper of the paper against RXNMapper, both scored against
the curated maps of EnzymeMap with SynKit's AAMValidator and its defaults, as benchmarks.chemistry.synrxn_map scores
the published mappers.

Every reaction with a curated map of every product atom is scored. Each mapped reaction is written by
npf.chem.mapping.mapped_smiles from a product -> precursor map in the atom order of the featurised reaction: the
reference from the curated map, ours from the maps of exact_map --third-level, RXNMapper's from its maps read the same
way. RXNMapper's raw output is scored as well, as a check that reading its maps back loses nothing. The paired
comparison counts the reactions only one of two mappers gets right, with an exact sign test.

    uv run python -m benchmarks.chemistry.enzymes prepare enzymemap   # and the maps, see the README
    uv run --with "synkit>=1.5,<1.6" python -m benchmarks.chemistry.synrxn_enzymemap   # results/enzymemap_ec/synrxn_map.json
    uv run --with "synkit>=1.5,<1.6" python -m benchmarks.chemistry.synrxn_enzymemap --limit 300   # printed, not written
"""

import argparse
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from npf.chem.mapping import mapped_smiles

DATA = Path("data/enzymemap_ec.pkl")
MAPS = Path("data/enzyme_maps")
OURS = MAPS / "mapper_maps_enzymemap_ec.pkl"
RXN = MAPS / "rxnmapper_maps_enzymemap_ec.pkl"
RXN_RAW = MAPS / "rxnmapper_enzymemap_ec.json"
OUT = Path("results/enzymemap_ec/synrxn_map.json")
COLUMNS = ("npf", "rxnmapper", "rxnmapper_raw")


def write(smiles, mapping):
    try:
        return None if mapping is None else mapped_smiles(smiles, np.asarray(mapping, np.int64))
    except Exception:
        return None


def chunk(rows):
    import pandas as pd
    from synkit.Chem.Reaction.Mapper import AAMValidator

    validator = AAMValidator()
    out = {}
    for col in COLUMNS:
        df = pd.DataFrame({"ground_truth": [r["ground_truth"] for r in rows], col: [r[col] or "" for r in rows]})
        first = validator.validate_smiles(data=df, ground_truth_col="ground_truth", mapped_cols=[col],
                                          ignore_tautomers=False)[0]
        first = first[0] if isinstance(first, (tuple, list)) else first
        out[col] = [bool(b) for b in first["results"]]

    return [r["id"] for r in rows], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first reactions, printed and not written")
    ap.add_argument("--processes", type=int, default=16)
    args = ap.parse_args()

    reactions = [r for r in pickle.loads(DATA.read_bytes())["reactions"]
                 if r["target"] is not None and (np.asarray(r["target"]) >= 0).all()]
    ours, rxn = pickle.loads(OURS.read_bytes()), pickle.loads(RXN.read_bytes())
    raw = json.loads(RXN_RAW.read_text())
    rows = []
    for r in reactions[: args.limit]:
        rows.append({"id": r["id"],
                     "ground_truth": write(r["smiles"], r["target"]),
                     "npf": write(r["smiles"], ours.get(r["id"])),
                     "rxnmapper": write(r["smiles"], rxn.get(r["id"])),
                     "rxnmapper_raw": (raw.get(str(r["id"])) or {}).get("mapped_rxn")})

    rows = [r for r in rows if r["ground_truth"]]
    with Pool(args.processes) as pool:
        results = pool.map(chunk, [rows[k : k + 200] for k in range(0, len(rows), 200)])

    ids = [i for part_ids, _ in results for i in part_ids]
    verdict = {col: np.array([v for _, out in results for v in out[col]]) for col in COLUMNS}
    summary = {"reactions": len(ids)} | {col: round(100 * float(v.mean()), 2) for col, v in verdict.items()}
    for other in ("rxnmapper", "rxnmapper_raw"):
        a, b = int((verdict["npf"] & ~verdict[other]).sum()), int((~verdict["npf"] & verdict[other]).sum())
        summary[f"npf_vs_{other}"] = {"only_npf": a, f"only_{other}": b,
                                      "p": float(binomtest(a, a + b).pvalue) if a + b else None}

    print(json.dumps(summary, indent=1))
    if args.limit is None:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(summary | {"verdicts": {"id": ids} | {c: v.tolist() for c, v in verdict.items()}}))


if __name__ == "__main__":
    main()
