"""Atom mapping on the Golden dataset of Lin et al., Mol. Inform. 2022, with 1,851 manually curated reactions, the
accepted benchmark for atom-to-atom mapping that RXNMapper, GraphormerMapper, LocalMapper and SAMMNet report on.

The criterion is the one of Lin et al. A mapping is correct if its condensed graph of reaction equals the CGR of the
curated mapping, a labelled graph over all precursor atoms with the element on the atoms and the bond order before
and after on the bonds, compared by isomorphism so that symmetry-equivalent mappings count. RXNMapper is scored with
the same code.

    uv run python -m benchmarks.chemistry.golden prepare <golden_dataset.rdf>       # writes data/golden.pkl and data/golden_unmapped.txt

The exact mapper is scored by benchmarks.chemistry.exact_map, the comparison with RXNMapper by exact_map_report.
"""

import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from npflow import chem
from npflow.chem.cgr import cgr, same_cgr  # noqa: F401

RDLogger.DisableLog("rdApp.*")


def read_rdf(path):
    """Mapped reaction SMILES 'precursors>>products' for every $RXN block that RDKit can read."""
    blocks = Path(path).read_text(errors="ignore").split("$RFMT")[1:]
    out = []
    for k, block in enumerate(blocks):
        rxn_block = "$RXN" + block.split("$RXN", 1)[1].split("$DTYPE", 1)[0]

        try:
            rxn = AllChem.ReactionFromRxnBlock(rxn_block, sanitize=False)
            mols = lambda ms: ".".join(
                Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(m))) for m in ms
            )
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
            row = {
                "original_rxn": f"{precursors}>>{products}",
                "rxn": f"{unmapped(precursors)}>>{unmapped(products)}",
                "label": 0,
                "split": "test",
                "id": k,
            }
            r = chem.featurise(row)
        except Exception:
            r = None

        if r is None:
            skipped += 1
        else:
            rows.append(r | {"mapped": mapped})

    usable = [r for r in rows if r["target"] is not None]
    Path(out).write_bytes(pickle.dumps({"reactions": rows, "classes": ["-"]}))
    Path(out).with_name("golden_unmapped.txt").write_text(
        "\n".join(f"{r['id']}\t{r['smiles']}" for r in rows)
    )
    print(
        f"{len(rows)} reactions read ({skipped} unreadable), {len(usable)} with a complete curated mapping of all product atoms"
    )


def mapping_from_smiles(reaction, mapped_rxn):
    """product atom -> precursor atom from a mapped reaction SMILES over the same molecules, as RXNMapper writes it."""
    precursors, products = mapped_rxn.split(">>")
    row = {
        "original_rxn": f"{precursors}>>{products}",
        "rxn": reaction["smiles"],
        "label": 0,
        "split": "test",
        "id": reaction["id"],
    }
    r = chem.featurise(row)
    if r is None or r["target"] is None:
        return None

    # featurise orders atoms canonically after removing the maps, so indices agree with reaction if the molecules do
    same = (
        np.array_equal(r["a"]["element"], reaction["a"]["element"])
        and np.array_equal(r["b"]["element"], reaction["b"]["element"])
        and np.array_equal(chem.dense_bonds(r["a"]), chem.dense_bonds(reaction["a"]))
        and np.array_equal(chem.dense_bonds(r["b"]), chem.dense_bonds(reaction["b"]))
    )

    if same:
        return r["target"].astype(np.int64)

    # the canonical order counts double-bond stereo, which the two SMILES can write differently, so the molecules are
    # matched as graphs without stereo and the mapping is carried over
    pa, pb = match(r["a"], reaction["a"]), match(r["b"], reaction["b"])
    if pa is None or pb is None:
        return None

    out = np.empty(len(pb), np.int64)
    out[pb] = pa[r["target"].astype(np.int64)]

    return out


def match(g, h):
    """Atom i of graph g onto an atom of graph h under an isomorphism by element, hydrogens, charge and bond type, or
    None. Symmetric choices give condensed graphs of reaction that are isomorphic, so any one serves.
    """

    def labelled(x):
        out, bonds = nx.Graph(), chem.dense_bonds(x)
        for i in range(len(x["element"])):
            out.add_node(
                i, label=(int(x["element"][i]), int(x["h"][i]), int(x["q"][i]))
            )

        for i, j in zip(*np.nonzero(np.triu(bonds, 1))):
            out.add_edge(int(i), int(j), label=int(bonds[i, j]))

        return out

    same = lambda x, y: x["label"] == y["label"]
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        labelled(g), labelled(h), node_match=same, edge_match=same
    )
    if not matcher.is_isomorphic():
        return None

    out = np.empty(len(g["element"]), np.int64)
    for i, j in matcher.mapping.items():
        out[i] = j

    return out


if __name__ == "__main__":
    {"prepare": prepare}[sys.argv[1]](*sys.argv[2:])
