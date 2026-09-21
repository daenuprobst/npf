"""Blinded review package for the atom-mapping disagreements (to be judged by a chemist).

    uv run python -m npf.chem_review build     # writes results/chem/mapping_review/{index.html, answers.csv, key.json}
    uv run python -m npf.chem_review score     # after answers.csv has been filled in

For a stratified sample of test reactions on which the NPF mapper, RXNMapper and the recorded mapping do not all
imply the same firing vector, every *distinct* candidate mapping is drawn (atom-map numbers on the mapped atoms,
the bonds that change under that mapping highlighted) and shown under a random letter. The reviewer marks each
candidate as correct (1), wrong (0) or unclear (?). `key.json` holds which source(s) produced which letter.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from . import chem_data, chem_models
from .chem_experiment import batches, firing, splits, tokens_moved
from .chem_rxnmapper import as_reaction

OUT = Path("results/chem/mapping_review")
PER_STRATUM = {"NPF differs, RXNMapper = record": 40, "NPF = RXNMapper, both differ from record": 40,
               "NPF = record, RXNMapper differs": 40, "all three differ": 30}


def centre_atoms(reaction, mapping):
    """Precursor atoms whose bonds change under the mapping (the reaction centre it implies)."""
    before = chem_data.dense_bonds(reaction["a"])
    after = np.zeros_like(before); after[np.ix_(mapping, mapping)] = chem_data.dense_bonds(reaction["b"])
    kept = np.zeros(len(before), bool); kept[mapping] = True
    i, j = np.nonzero((after != before) & (kept[:, None] | kept[None, :]))
    return sorted({int(x) for x in np.concatenate([i, j])})


def draw_side(g, atoms, numbers, highlight, width):
    """One side of the reaction: only the atoms in `atoms`, map numbers only where given, highlighted centre."""
    atoms = list(atoms)
    new = {old: k for k, old in enumerate(atoms)}
    bonds = chem_data.dense_bonds(g)[np.ix_(atoms, atoms)]
    mol = chem_data._mol(g["element"][atoms], bonds, g["h"][atoms], g["q"][atoms])
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)
    for old, number in numbers.items():
        if old in new:
            mol.GetAtomWithIdx(new[old]).SetProp("atomNote", str(number))
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, 380)
    drawer.drawOptions().annotationFontScale = 0.9
    drawer.DrawMolecule(mol, highlightAtoms=[new[a] for a in highlight if a in new])
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def depict(reaction, mapping):
    """Participating precursor molecules -> product, numbered and highlighted around the reaction centre."""
    a, b = reaction["a"], reaction["b"]
    mapping = np.asarray(mapping)
    centre = centre_atoms(reaction, mapping)
    near = set(centre)
    bonds_a = chem_data.dense_bonds(a)
    for c in centre:  # label the centre and its neighbours, everything else stays unlabelled
        near |= {int(x) for x in np.nonzero(bonds_a[c])[0]}
    number = {int(j): k + 1 for k, j in enumerate(mapping)}  # precursor atom -> map number (= product atom + 1)
    participating = [i for i in range(len(a["element"])) if a["fragment"][i] in set(a["fragment"][mapping])]
    left = draw_side(a, participating, {i: number[i] for i in near if i in number}, centre, 700)
    inverse = {int(j): k for k, j in enumerate(mapping)}
    right = draw_side(b, range(len(b["element"])), {inverse[i]: number[i] for i in near if i in inverse},
                      [inverse[i] for i in centre if i in inverse], 480)
    return f"<div style='display:flex;align-items:center'>{left}<span style='font-size:40px'>&rarr;</span>{right}</div>"


@torch.no_grad()
def build(seed=0):
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = chem_data.load()
    _, _, test = splits(data, "map")
    theirs = json.loads(Path("results/chem/rxnmapper_test.json").read_text())
    mapper = chem_models.Mapper().to(device)
    mapper.load_state_dict(torch.load("results/chem/map/npf-0.pt", map_location=device)); mapper.eval()
    strata = defaultdict(list)
    for rs in batches(test, 64):
        b = chem_data.collate(rs, device)
        for r, ours in zip(rs, mapper.decode(mapper(b), b)):
            other = as_reaction(theirs[str(r["id"])]["mapped_rxn"]) if theirs.get(str(r["id"]), {}).get("mapped_rxn") else None
            if other is None or other["target"] is None or len(other["a"]["x"]) != len(r["a"]["x"]):
                continue
            f_rec, f_npf, f_rxm = firing(r, r["target"]), firing(r, ours), firing(other, other["target"])
            if f_npf == f_rec == f_rxm:
                continue
            name = ("NPF differs, RXNMapper = record" if f_rxm == f_rec else "NPF = RXNMapper, both differ from record" if f_npf == f_rxm
                    else "NPF = record, RXNMapper differs" if f_npf == f_rec else "all three differ")
            strata[name].append((r, ours, other))
    OUT.mkdir(parents=True, exist_ok=True)
    html = ["<html><head><meta charset='utf-8'><title>Atom-mapping review</title><style>body{font-family:sans-serif;max-width:1150px;margin:auto}"
            ".case{border-top:2px solid #888;margin-top:2em;padding-top:1em}.cand{margin:0.5em 0}</style></head><body>",
            "<h1>Atom-mapping review</h1><p>For every candidate: is this atom mapping chemically correct? Numbers are atom-map numbers; "
            "mark 1 (correct), 0 (wrong) or ? (unclear) in <code>answers.csv</code>. Only the precursor molecules that contribute atoms to the product are drawn; the full reaction is given as text. Several candidates of a case can be correct "
            "(equivalent atoms, resonance).</p>"]
    key, rows, case = {}, [], 0
    for name, wanted in PER_STRATUM.items():
        pool = strata[name]
        for k in rng.permutation(len(pool))[:wanted]:
            r, ours, other = pool[k]
            # candidates in our atom indexing; RXNMapper's mapping lives on its own parse, so it is drawn from its own record
            candidates = {"recorded": (r, r["target"]), "NPF": (r, ours), "RXNMapper": (other, other["target"])}
            distinct = []
            for source, (rec, mapping) in candidates.items():
                f = firing(rec, mapping)
                for entry in distinct:
                    if entry["firing"] == f:
                        entry["sources"].append(source); break
                else:
                    distinct.append({"firing": f, "sources": [source], "record": rec, "mapping": mapping})
            case += 1
            html.append(f"<div class='case'><h3>Case {case}</h3><p><code>{r['smiles']}</code><br>class {data['classes'][r['label']]}</p>")
            for letter, idx in zip("ABC", rng.permutation(len(distinct))):
                entry = distinct[idx]
                try:
                    svg = depict(entry["record"], entry["mapping"])
                except Exception as error:  # a depiction problem must not drop the case
                    svg = f"<pre>could not draw: {error}</pre>"
                html.append(f"<div class='cand'><b>Candidate {letter}</b> (highlighted: atoms whose bonds change; numbers: atom maps near them)<br>{svg}</div>")
                key[f"{case}{letter}"] = {"sources": entry["sources"], "stratum": name, "reaction_id": int(r["id"]),
                                           "tokens_moved": tokens_moved(entry["record"], entry["mapping"])}
                rows.append({"case": case, "candidate": letter, "correct (1 / 0 / ?)": "", "comment": ""})
            html.append("</div>")
    html.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(html))
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    with open(OUT / "answers.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"{case} cases, {len(rows)} candidates -> {OUT}/index.html ; strata available: " + ", ".join(f"{k}: {len(v)}" for k, v in strata.items()))


def score():
    key = json.loads((OUT / "key.json").read_text())
    verdict = {f"{row['case']}{row['candidate']}": row["correct (1 / 0 / ?)"].strip() for row in csv.DictReader(open(OUT / "answers.csv"))}
    tally = defaultdict(Counter)
    for cand, info in key.items():
        if verdict.get(cand) in ("0", "1"):
            for source in info["sources"]:
                tally[info["stratum"], source][verdict[cand]] += 1
    for (stratum, source), c in sorted(tally.items()):
        print(f"{stratum:45s} {source:10s} correct {c['1']:3d} / {c['0'] + c['1']:3d}")


if __name__ == "__main__":
    build() if len(sys.argv) < 2 or sys.argv[1] == "build" else score()
