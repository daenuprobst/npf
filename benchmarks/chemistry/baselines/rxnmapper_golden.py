"""RXNMapper on the unmapped Golden reactions, the pretrained mapper the exact mapper is compared with.

RXNMapper needs an environment of its own with transformers and an older torch, so this script imports nothing of npf.
It reads data/golden_unmapped.txt from benchmarks.chemistry.golden prepare and writes the mapped reaction SMILES and
confidence per reaction id, the file benchmarks.chemistry.exact_map_report reads.

    uv run --no-project --python 3.11 --with rxnmapper --with rdkit --with "setuptools<81" --with "numpy<2" python benchmarks/chemistry/baselines/rxnmapper_golden.py
"""

import json
import sys
import time
from pathlib import Path


def main(
    unmapped="data/golden_unmapped.txt",
    out="results/chem/rxnmapper_golden.json",
    batch=32,
):
    # imported here so the script imports without the rxnmapper environment
    from rxnmapper import RXNMapper

    rows = [
        line.split("\t")
        for line in Path(unmapped).read_text().splitlines()
        if line.strip()
    ]
    mapper, done, start = RXNMapper(), {}, time.time()
    for k in range(0, len(rows), int(batch)):
        part = rows[k : k + int(batch)]

        # a reaction RXNMapper cannot map is left out, the report scores it as wrong
        try:
            results = mapper.get_attention_guided_atom_maps(
                [smiles for _, smiles in part]
            )
        except Exception:
            results = []
            for _, smiles in part:
                try:
                    results.append(mapper.get_attention_guided_atom_maps([smiles])[0])
                except Exception:
                    results.append(None)

        for (rid, _), result in zip(part, results):
            if result is not None:
                done[rid] = {
                    "mapped_rxn": result["mapped_rxn"],
                    "confidence": str(result["confidence"]),
                }

        print(f"{k} {time.time() - start:.0f}s", flush=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(done, indent=1))

    print(f"done {len(done)} of {len(rows)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
