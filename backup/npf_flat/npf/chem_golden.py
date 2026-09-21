"""Atom mapping on the Golden dataset (Lin et al., Mol. Inform. 2022): 1,851 manually curated reactions, the accepted
benchmark for atom-to-atom mapping (RXNMapper, GraphormerMapper, LocalMapper, SAMMNet report on it).

Criterion, as in Lin et al.: a mapping is correct if its condensed graph of reaction (CGR) is identical to the CGR of
the curated mapping. We build the CGR as a labelled graph (atoms: element; bonds: order before -> order after) over
all precursor atoms and test isomorphism, so symmetry-equivalent mappings count as correct. Our mapper is applied as
trained on Schneider 50k (zero-shot on this set); RXNMapper is scored with exactly the same code.

    uv run python -m npf.chem_golden prepare <golden_dataset.rdf>       # -> data/golden.pkl, data/golden_unmapped.txt
    uv run python -m npf.chem_golden evaluate [rxnmapper_output.json]    # -> results/chem/golden.json
"""
import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from . import chem_data, chem_models

RDLogger.DisableLog("rdApp.*")


def read_rdf(path):
    """Mapped reaction SMILES 'precursors>>products' for every $RXN block that RDKit can read."""
    blocks = Path(path).read_text(errors="ignore").split("$RFMT")[1:]
    out = []
    for k, block in enumerate(blocks):
        rxn_block = "$RXN" + block.split("$RXN", 1)[1].split("$DTYPE", 1)[0]
        try:
            rxn = AllChem.ReactionFromRxnBlock(rxn_block, sanitize=False)
            mols = lambda ms: ".".join(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(m))) for m in ms)
            out.append((k, f"{mols(rxn.GetReactants())}>>{mols(rxn.GetProducts())}"))
        except Exception:
            out.append((k, None))
    return out


def unmapped(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def prepare(rdf, out="data/golden.pkl"):
    rows, skipped = [], 0
    for k, mapped in read_rdf(rdf):
        try:
            precursors, products = mapped.split(">>")
            row = {"original_rxn": f"{precursors}>>{products}", "rxn": f"{unmapped(precursors)}>>{unmapped(products)}", "label": 0, "split": "test", "id": k}
            r = chem_data.featurise(row)
        except Exception:
            r = None
        if r is None:
            skipped += 1
        else:
            rows.append(r | {"mapped": mapped})
    usable = [r for r in rows if r["target"] is not None]
    Path(out).write_bytes(pickle.dumps({"reactions": rows, "classes": ["-"]}))
    Path(out).with_name("golden_unmapped.txt").write_text("\n".join(f"{r['id']}\t{r['smiles']}" for r in rows))
    print(f"{len(rows)} reactions read ({skipped} unreadable), {len(usable)} with a complete curated mapping of all product atoms")


def cgr(reaction, mapping):
    """Condensed graph of reaction under a product -> precursor mapping, over all precursor atoms."""
    a, bb = reaction["a"], chem_data.dense_bonds(reaction["b"])
    before = chem_data.dense_bonds(a)
    after = np.zeros_like(before); after[np.ix_(mapping, mapping)] = bb
    kept = np.zeros(len(before), bool); kept[mapping] = True
    g = nx.Graph()
    for i, el in enumerate(a["element"]):
        g.add_node(i, label=(int(el), bool(kept[i])))
    for i, j in zip(*np.nonzero(np.triu((before > 0) | (after > 0), 1))):
        # bonds between two atoms that both leave are not part of the product: their fate is not recorded
        new = int(after[i, j]) if (kept[i] and kept[j]) else (0 if (kept[i] or kept[j]) else int(before[i, j]))
        g.add_edge(int(i), int(j), label=(int(before[i, j]), new))
    return g


def same_cgr(reaction, mapping, reference):
    g1, g2 = cgr(reaction, mapping), cgr(reaction, reference)
    centre = lambda g: sorted(d["label"] for _, _, d in g.edges(data=True) if d["label"][0] != d["label"][1])
    if centre(g1) != centre(g2):
        return False
    return nx.is_isomorphic(g1, g2, node_match=lambda x, y: x["label"] == y["label"], edge_match=lambda x, y: x["label"] == y["label"])


def mapping_from_smiles(reaction, mapped_rxn):
    """product atom -> precursor atom from a mapped reaction SMILES over the same molecules (e.g. RXNMapper output)."""
    precursors, products = mapped_rxn.split(">>")
    row = {"original_rxn": f"{precursors}>>{products}", "rxn": reaction["smiles"], "label": 0, "split": "test", "id": reaction["id"]}
    r = chem_data.featurise(row)
    if r is None or r["target"] is None:
        return None
    # featurise orders atoms canonically after removing the maps, so indices agree with `reaction` if the molecules do
    same = np.array_equal(r["a"]["element"], reaction["a"]["element"]) and np.array_equal(r["b"]["element"], reaction["b"]["element"]) \
        and np.array_equal(chem_data.dense_bonds(r["a"]), chem_data.dense_bonds(reaction["a"])) and np.array_equal(chem_data.dense_bonds(r["b"]), chem_data.dense_bonds(reaction["b"]))
    return r["target"].astype(np.int64) if same else None


@torch.no_grad()
def evaluate(rxnmapper_json=None, weights="results/chem/map/npf-0.pt", out="results/chem/golden.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reactions = pickle.loads(Path("data/golden.pkl").read_bytes())["reactions"]
    n_total = len(reactions)
    usable = [r for r in reactions if r["target"] is not None and len(r["b"]["x"]) <= len(r["a"]["x"])]
    model = chem_models.Mapper().to(device)
    model.load_state_dict(torch.load(weights, map_location=device)); model.eval()
    correct, per_reaction = 0, {}
    for start in range(0, len(usable), 16):
        rs = usable[start:start + 16]
        b = chem_data.collate(rs, device)
        for r, pred in zip(rs, model.decode(model(b), b)):
            ok = len(pred) == len(r["target"]) and same_cgr(r, pred.astype(np.int64), r["target"].astype(np.int64))
            correct += ok; per_reaction[r["id"]] = bool(ok)
    result = {"n_reactions": n_total, "n_with_complete_curated_mapping": len(usable),
              "npf_correct_on_usable": correct / len(usable), "npf_correct_on_all": correct / n_total}
    print(f"NPF mapper (trained on Schneider 50k, zero-shot): {correct}/{len(usable)} = {correct / len(usable):.2%} of the usable reactions, "
          f"{correct / n_total:.2%} of all {n_total}")
    if rxnmapper_json:
        mapped = json.loads(Path(rxnmapper_json).read_text())
        hits = comparable = both = 0
        for r in usable:
            m = mapping_from_smiles(r, mapped[str(r["id"])]["mapped_rxn"]) if str(r["id"]) in mapped else None
            if m is None:
                continue
            comparable += 1
            ok = same_cgr(r, m, r["target"].astype(np.int64))
            hits += ok; both += ok and per_reaction[r["id"]]
        result |= {"rxnmapper_correct_on_usable": hits / len(usable), "rxnmapper_outputs_comparable": comparable, "both_correct": both / len(usable)}
        print(f"RXNMapper, same code: {hits}/{len(usable)} = {hits / len(usable):.2%} (outputs that could be aligned: {comparable})")
    Path(out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "evaluate": evaluate}[sys.argv[1]](*sys.argv[2:])
