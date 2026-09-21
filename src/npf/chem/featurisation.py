"""Reactions as pairs of attributed graphs. State A is the precursor graph, state B the product graph."""
import csv
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B", "Sn", "Mg", "Li", "Na", "K", "Cu", "Pd", "Zn", "Se"]
BOND_TYPE = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}

# bond order of every bond type index, aromatic bonds count 1.5
BOND_ORDER = np.array([0.0, 1.0, 2.0, 3.0, 1.5])

# capacity of the slack place, the number of valence tokens an atom can take up beyond its hydrogens by changing
# charge or expanding its valence. measured on the training data, carbon needs none, N and O one, S and P two
EXTRA_CAPACITY = {7: 1, 8: 1, 15: 2, 16: 2, 17: 1, 35: 1, 53: 1}

# valence expansion in steps of two tokens, as in S=O and P=O
HYPERVALENT = {15: 2, 16: 4, 34: 4, 53: 4}
N_ATOM_FEAT = len(ELEMENTS) + 1 + 6 + 4 + 5 + 2
MAX_PRECURSOR_ATOMS, MAX_PRODUCT_ATOMS = 100, 70
RD_BOND = {v: k for k, v in BOND_TYPE.items()}


def one_hot(index, size):
    v = np.zeros(size, np.uint8)
    v[min(index, size - 1)] = 1

    return v


def atom_features(atom):
    symbol = atom.GetSymbol()

    return np.concatenate([
        one_hot(ELEMENTS.index(symbol) if symbol in ELEMENTS else len(ELEMENTS), len(ELEMENTS) + 1),
        one_hot(atom.GetDegree(), 6),
        one_hot({-1: 0, 0: 1, 1: 2}.get(atom.GetFormalCharge(), 3), 4),
        one_hot(atom.GetTotalNumHs(), 5),
        [atom.GetIsAromatic(), atom.IsInRing()],
    ]).astype(np.uint8)


def skeleton_classes(mol):
    """Symmetry classes of the bare skeleton, elements and connectivity only. Atoms that differ only by bond orders,
    hydrogens or charges, such as the two oxygens of a carboxylic acid or of a nitro group, are interchangeable
    in an atom mapping, and these classes treat them as such."""
    bare = Chem.RWMol(mol)
    for bond in bare.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)

    for atom in bare.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    bare.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(bare)

    return np.array(list(Chem.CanonicalRankAtoms(bare, breakTies=False, includeChirality=False)), np.int16)


def graph(smiles):
    """Molecular graph in canonical atom order. Returns None if RDKit cannot parse the SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    maps = np.array([a.GetAtomMapNum() for a in mol.GetAtoms()])
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)

    order = np.argsort(list(Chem.CanonicalRankAtoms(mol, breakTies=True)))
    new = np.empty_like(order)
    new[order] = np.arange(len(order))
    bonds = np.array([(new[b.GetBeginAtomIdx()], new[b.GetEndAtomIdx()], BOND_TYPE[b.GetBondType()]) for b in mol.GetBonds()
                      if b.GetBondType() in BOND_TYPE], dtype=np.int16).reshape(-1, 3)
    fragment = np.zeros(len(order), np.int16)
    for k, atoms in enumerate(Chem.GetMolFrags(mol)):
        fragment[new[list(atoms)]] = k

    return {
        "x": np.stack([atom_features(mol.GetAtomWithIdx(int(i))) for i in order]),
        "element": np.array([mol.GetAtomWithIdx(int(i)).GetAtomicNum() for i in order], np.int16),
        "h": np.array([mol.GetAtomWithIdx(int(i)).GetTotalNumHs() for i in order], np.int16),
        "q": np.array([mol.GetAtomWithIdx(int(i)).GetFormalCharge() for i in order], np.int16),
        "bonds": bonds,
        "fragment": fragment,
        # atoms with equal rank are equivalent
        "symmetry": np.array(list(Chem.CanonicalRankAtoms(mol, breakTies=False)), np.int16)[order],
        "skeleton": skeleton_classes(mol)[order],
        "maps": maps[order],
    }


def dense_bonds(g):
    n = len(g["x"])
    b = np.zeros((n, n), np.int8)

    if len(g["bonds"]):
        i, j, t = g["bonds"].T
        b[i, j] = t
        b[j, i] = t

    return b


def featurise(row):
    """Both graphs for every parseable reaction. target and edits are set only if the recorded atom mapping is a
    clean injection of product atoms into precursor atoms. Classification does not need them."""
    reactants, reagents, product = row["original_rxn"].split(">")
    a, b = graph(".".join(s for s in (reactants, reagents) if s)), graph(product)
    if a is None or b is None or len(a["x"]) > 200 or len(b["x"]) > 130:
        return None

    out = {"a": a, "b": b, "target": None, "edits": None, "label": row["label"], "split": row["split"], "id": row["id"], "smiles": row["rxn"]}
    maps_a, maps_b = a.pop("maps"), b.pop("maps")
    where = {m: j for j, m in enumerate(maps_a) if m}
    clean = len(where) == (maps_a > 0).sum() and (maps_b > 0).all() and len(set(maps_b)) == len(maps_b) and all(m in where for m in maps_b)
    if not clean or len(a["x"]) > MAX_PRECURSOR_ATOMS or len(b["x"]) > MAX_PRODUCT_ATOMS:
        return out

    # target maps every product atom to its precursor atom
    target = np.array([where[m] for m in maps_b], np.int16)
    if (a["element"][target] != b["element"]).any():
        return out

    # the firing vector, a new bond type for every precursor pair whose bond differs between A and B
    ba, bb = dense_bonds(a), dense_bonds(b)
    after = np.zeros_like(ba)
    after[np.ix_(target, target)] = bb
    kept = np.zeros(len(ba), bool)
    kept[target] = True
    changed = (after != ba) & (kept[:, None] | kept[None, :])
    i, j = np.nonzero(np.triu(changed, 1))

    # rows are (atom, atom, new bond type) with type 0 for a broken bond
    out["edits"] = np.stack([i, j, after[i, j]], 1).astype(np.int16)
    out["target"] = target

    # P-invariant of the valence net per atom, bond tokens + hydrogens - charge is conserved by every firing
    valence = lambda g, bonds: BOND_ORDER[bonds].sum(1) + g["h"] - g["q"]
    out["conserved"] = np.isclose(valence(a, ba)[target], valence(b, bb))

    return out


def build(tsv="data/schneider50k.tsv", out="data/schneider50k.pkl"):
    """Schneider 50k with its 50 reaction classes. Prints how often the valence P-invariant holds in the records."""
    rows = list(csv.DictReader(open(tsv), delimiter="\t"))
    classes = sorted({r["rxn_class"] for r in rows})
    for k, r in enumerate(rows):
        r["label"], r["id"] = classes.index(r["rxn_class"]), k

    with Pool(12) as pool:
        data = [d for d in pool.map(featurise, rows, chunksize=256) if d is not None]

    Path(out).write_bytes(pickle.dumps({"reactions": data, "classes": classes}))
    mapped = [d for d in data if d["target"] is not None]
    n_edits = np.array([len(d["edits"]) for d in mapped])
    atoms_ok = np.concatenate([d["conserved"] for d in mapped])
    print(f"{len(data)}/{len(rows)} reactions parsed, {len(mapped)} with a clean atom mapping and <= {MAX_PRECURSOR_ATOMS}/{MAX_PRODUCT_ATOMS} atoms")
    print(f"valence P-invariant (bond orders + H - charge) conserved for {atoms_ok.mean():.4f} of product atoms, "
          f"for every atom in {np.mean([d['conserved'].all() for d in mapped]):.4f} of reactions")
    print(f"atoms: precursors {np.mean([len(d['a']['x']) for d in mapped]):.1f}, product {np.mean([len(d['b']['x']) for d in mapped]):.1f}; "
          f"edits per reaction: mean {n_edits.mean():.2f}, median {np.median(n_edits):.0f}, max {n_edits.max()}")


def _featurise_mit(args):
    line, split, k = args
    rxn = line.split()[0]
    precursors, product = rxn.split(">>")
    plain = lambda smi: Chem.MolToSmiles(_without_maps(Chem.MolFromSmiles(smi)))

    try:
        row = {"original_rxn": f"{precursors}>>{product}", "label": 0, "split": split, "id": k, "rxn": f"{plain(precursors)}>>{plain(product)}"}
        return featurise(row)
    except Exception:
        return None


def _without_maps(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return mol


def build_uspto_mit(folder="data/uspto_mit/data", out="data/uspto_mit.pkl"):
    """USPTO-MIT of Jin et al. 2017, the standard forward prediction benchmark with 409k, 30k and 40k reactions."""
    jobs = []
    for split, name in (("train", "train.txt"), ("val", "valid.txt"), ("test", "test.txt")):
        jobs += [(line, split, len(jobs) + k) for k, line in enumerate(open(Path(folder) / name)) if line.strip()]

    with Pool(12) as pool:
        data = [d for d in pool.map(_featurise_mit, jobs, chunksize=512) if d is not None]

    Path(out).write_bytes(pickle.dumps({"reactions": data, "classes": ["-"]}, protocol=4))

    for split in ("train", "val", "test"):
        rs = [d for d in data if d["split"] == split]
        print(f"{split}: {len(rs)} parsed, {sum(d['target'] is not None for d in rs)} with clean mapping and size <= {MAX_PRECURSOR_ATOMS}/{MAX_PRODUCT_ATOMS}")


def load(path="data/schneider50k.pkl"):
    return pickle.loads(Path(path).read_bytes())


def collate(reactions, device):
    """Pad a list of reactions into dense tensors. edits holds 0 for unchanged and k + 1 for new bond type k."""
    B = len(reactions)
    na, nb = max(len(r["a"]["x"]) for r in reactions), max(len(r["b"]["x"]) for r in reactions)
    out = {
        "xa": np.zeros((B, na, N_ATOM_FEAT), np.float32), "xb": np.zeros((B, nb, N_ATOM_FEAT), np.float32),
        "ba": np.zeros((B, na, na), np.int64), "bb": np.zeros((B, nb, nb), np.int64),
        "mask_a": np.zeros((B, na), bool), "mask_b": np.zeros((B, nb), bool),
        "el_a": np.zeros((B, na), np.int64), "el_b": np.full((B, nb), -1, np.int64),
        "frag_a": np.full((B, na), -1, np.int64), "sym_a": np.full((B, na), -1, np.int64), "sym_b": np.full((B, nb), -2, np.int64),
        "h_a": np.zeros((B, na), np.float32), "cap_a": np.zeros((B, na), np.float32),
        "h_b": np.zeros((B, nb), np.float32), "q_a": np.zeros((B, na), np.float32), "q_b": np.zeros((B, nb), np.float32),
        "target": np.full((B, nb), -1, np.int64), "edits": np.zeros((B, na, na), np.int64),
        "label": np.array([r["label"] for r in reactions], np.int64),
        "labelled": np.array([r.get("labelled", True) for r in reactions], bool),
    }

    # histogram of fired transition types, an auxiliary target from the recorded mapping
    if "hist" in reactions[0]:
        out["hist"] = np.stack([r["hist"] for r in reactions]).astype(np.float32)

    for k, r in enumerate(reactions):
        a, b = r["a"], r["b"]
        n, m = len(a["x"]), len(b["x"])
        out["xa"][k, :n], out["xb"][k, :m] = a["x"], b["x"]
        out["ba"][k, :n, :n], out["bb"][k, :m, :m] = dense_bonds(a), dense_bonds(b)
        out["mask_a"][k, :n], out["mask_b"][k, :m] = True, True
        out["el_a"][k, :n], out["el_b"][k, :m] = a["element"], b["element"]
        out["frag_a"][k, :n], out["sym_a"][k, :n], out["sym_b"][k, :m] = a["fragment"], a["skeleton"], b["skeleton"]
        out["h_a"][k, :n], out["h_b"][k, :m], out["q_a"][k, :n], out["q_b"][k, :m] = a["h"], b["h"], a["q"], b["q"]
        out["cap_a"][k, :n] = [EXTRA_CAPACITY.get(int(e), 0) for e in a["element"]] + np.maximum(-a["q"], 0)

        if r["target"] is not None:
            out["target"][k, :m] = r["target"]

        if r["edits"] is not None and len(r["edits"]):
            i, j, t = r["edits"].T.astype(np.int64)
            out["edits"][k, i, j] = t + 1
            out["edits"][k, j, i] = t + 1

    return {k: torch.as_tensor(v, device=device) for k, v in out.items()}
