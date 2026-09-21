"""Schneider 50k reactions as transitions between two attributed graphs.

State A is the precursor graph (reactants and reagents, roles not given), state B the product graph.
In Petri terms atoms are places, bond electron pairs are tokens and the transitions that fire between A
and B are the bond edits (the Dugundji-Ugi reaction matrix R in E_B = E_A + R is a state equation).

Atoms are put in canonical order *after* the atom-map numbers have been removed, so neither the order nor
any feature can leak the ground-truth mapping to a model.

    uv run python -m npf.chem_data        # builds data/schneider50k.pkl
"""
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
BOND_ORDER = np.array([0.0, 1.0, 2.0, 3.0, 1.5])  # bond type index -> bond order
# Capacity of the hydrogen place: how many valence tokens an atom can take up beyond its hydrogens by changing
# charge or expanding its valence (onium / N-oxide formation, S and P oxidation, halide anions). Measured on the
# data: carbon needs none (its hydrogen place goes negative in 0.02 % of all reactions), N and O one, S two.
EXTRA_CAPACITY = {7: 1, 8: 1, 15: 2, 16: 2, 17: 1, 35: 1, 53: 1}
HYPERVALENT = {15: 2, 16: 4, 34: 4, 53: 4}  # valence expansion in steps of two tokens (S=O, P=O, ...)
N_ATOM_FEAT = len(ELEMENTS) + 1 + 6 + 4 + 5 + 2
MAX_PRECURSOR_ATOMS, MAX_PRODUCT_ATOMS = 100, 70


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
    """Symmetry classes of the bare skeleton (elements and connectivity only). Atoms that differ only by bond
    orders, hydrogens or charges - the two oxygens of a carboxylic acid or a nitro group, tautomers - are
    interchangeable in an atom mapping, and these classes treat them as such."""
    bare = Chem.RWMol(mol)
    for bond in bare.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE); bond.SetIsAromatic(False)
    for atom in bare.GetAtoms():
        atom.SetFormalCharge(0); atom.SetNumExplicitHs(0); atom.SetNoImplicit(True); atom.SetIsAromatic(False)
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
    new = np.empty_like(order); new[order] = np.arange(len(order))
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
        "symmetry": np.array(list(Chem.CanonicalRankAtoms(mol, breakTies=False)), np.int16)[order],  # equal = equivalent
        "skeleton": skeleton_classes(mol)[order],
        "maps": maps[order],
    }


def dense_bonds(g):
    n = len(g["x"])
    b = np.zeros((n, n), np.int8)
    if len(g["bonds"]):
        i, j, t = g["bonds"].T
        b[i, j] = t; b[j, i] = t
    return b


def featurise(row):
    """Both graphs for every parseable reaction; `target` and `edits` only if the recorded atom mapping is a
    clean injection of product atoms into precursor atoms (classification does not need it)."""
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
    target = np.array([where[m] for m in maps_b], np.int16)  # product atom -> precursor atom
    if (a["element"][target] != b["element"]).any():
        return out
    # the transitions that fire: new bond type for every precursor pair whose bond differs between A and B
    ba, bb = dense_bonds(a), dense_bonds(b)
    after = np.zeros_like(ba)
    after[np.ix_(target, target)] = bb
    kept = np.zeros(len(ba), bool); kept[target] = True
    changed = (after != ba) & (kept[:, None] | kept[None, :])
    i, j = np.nonzero(np.triu(changed, 1))
    out["edits"] = np.stack([i, j, after[i, j]], 1).astype(np.int16)  # (atom, atom, new bond type; 0 = bond broken)
    out["target"] = target
    # per-atom P-invariant of the valence net: bond tokens + hydrogens - charge is conserved by every firing
    valence = lambda g, bonds: BOND_ORDER[bonds].sum(1) + g["h"] - g["q"]
    out["conserved"] = np.isclose(valence(a, ba)[target], valence(b, bb))
    return out


def build(tsv="data/schneider50k.tsv", out="data/schneider50k.pkl"):
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


RD_BOND = {v: k for k, v in BOND_TYPE.items()}


def canonical_product(smiles):
    """Largest fragment, no stereochemistry, canonical - the form in which products are compared."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return max(Chem.MolToSmiles(mol).split("."), key=lambda f: (Chem.MolFromSmiles(f).GetNumHeavyAtoms(), f))


def marking_to_products(a, edits):
    return marking_fragments(a, edits)[0]


RECONSTRUCTION = 3  # 1: rules of the first Schneider 50k results; 2: adds R1-R5; 3: adds the neutral-product convention R6 (all chosen on training reactions only)
NORMAL_VALENCE = {15: (3, 5), 16: (2, 4, 6), 34: (2, 4, 6), 17: (1, 3, 5, 7), 35: (1, 3, 5, 7), 53: (1, 3, 5, 7)}
ANION_PRIORITY = {53: 0, 35: 0, 17: 0, 9: 0, 8: 1, 16: 1, 7: 2}
NEUTRALISE_MIN_ATOMS = 2  # single-atom ions (halides, hydroxide) stay as they are


def marking_fragments(a, edits):
    """Molecules encoded by the marking after the firings `edits` [(i, j, new bond type)], as sets of
    canonical SMILES: the fragments that contain a re-typed bond place, and all fragments.

    Hydrogens are not predicted. They follow from the per-atom P-invariant (bond tokens + H - charge is
    conserved): tokens an atom cannot cover with hydrogens go to its charge (onium formation), and fast
    acid-base transitions then relax the marking - a new cation with a hydrogen loses the proton, a new
    aromatic cation takes it from an [nH] of the same ring system (tautomer shift).

    A slack token can be a hydrogen, a lone pair or a charge; the marking does not say which. Version 2 decides with
    three more conservation arguments: (R1) S, P and halogens return tokens to lone pairs in steps of two, (R2/R3)
    inside an aromatic system a surplus token pairs with the hydrogen of an [nH] (both vanish into the pi system) and
    a deficit is covered by an aromatic n that takes up a hydrogen, (R5) total charge is conserved: for every new
    cation one of the atoms that would have gained a hydrogen becomes an anion instead (N-oxides, nitro groups,
    halide counter-ions of quaternary salts)."""
    before = dense_bonds(a)
    after = before.copy()
    for i, j, t in edits:
        after[i, j] = after[j, i] = t
    left = np.ceil(a["h"] - (BOND_ORDER[after] - BOND_ORDER[before]).sum(1) - 1e-6)  # aromatic bonds count 1.5
    expand = np.array([HYPERVALENT.get(int(e), 0) for e in a["element"]])  # S, P, ... take tokens from lone pairs, no charge
    deficit = np.maximum(-left, 0)
    h, q = np.maximum(left, 0).astype(int), (a["q"] + deficit - np.minimum(deficit, expand) // 2 * 2).astype(int)
    touched = np.zeros(len(h), bool)
    for i, j, _ in edits:
        touched[[i, j]] = True
    aromatic = (after == 4).any(1)

    def ring_system(i):
        seen, todo = {i}, [i]
        while todo:
            for k in np.nonzero(after[todo.pop()] == 4)[0]:
                if k not in seen:
                    seen.add(k); todo.append(k)
        return seen

    if RECONSTRUCTION >= 2:
        for i in np.nonzero(touched & (h > a["h"]))[0]:
            allowed = NORMAL_VALENCE.get(int(a["element"][i]))
            bonds = BOND_ORDER[after[i]].sum()
            if allowed:  # R1: hypervalent elements give tokens back to their lone pairs, two at a time
                target = min((v for v in allowed if v >= bonds - max(q[i], 0) - 1e-6), default=allowed[-1])
                while h[i] >= 2 and bonds + h[i] - q[i] - 2 >= target - 1e-6:
                    h[i] -= 2
            elif aromatic[i] and a["element"][i] == 6:  # R2: the surplus token of an aromatic carbon joins the pi system
                donors = [k for k in ring_system(i) if a["element"][k] == 7 and h[k] > 0 and q[k] == 0 and k != i]
                if donors and h[i] > 0:
                    h[i] -= 1; h[donors[0]] -= 1
    for i in np.nonzero(touched & (q > 0))[0]:
        if h[i] > 0:
            h[i] -= 1; q[i] -= 1
        elif aromatic[i]:  # aromatic: walk the ring system for an [nH]
            donors = [k for k in ring_system(i) if a["element"][k] == 7 and h[k] > 0 and q[k] == 0]
            if donors:
                h[donors[0]] -= 1; q[i] -= 1
            elif RECONSTRUCTION >= 2 and a["element"][i] == 6:  # R3: ... or for an aromatic n that takes up a hydrogen (lactam)
                takers = [k for k in ring_system(i) if a["element"][k] == 7 and h[k] == 0 and q[k] == 0 and (after[k] > 0).sum() == 2]
                if takers:
                    h[takers[0]] += 1; q[i] -= 1
    if RECONSTRUCTION >= 2:  # R5: charge is conserved
        surplus = int(q.sum() - a["q"].sum())
        if surplus > 0:
            new_cation = q > a["q"]
            gained = [k for k in np.nonzero(touched & (h > a["h"]) & (q == a["q"]))[0] if int(a["element"][k]) in ANION_PRIORITY]
            gained.sort(key=lambda k: (not new_cation[after[k] > 0].any(), ANION_PRIORITY[int(a["element"][k])]))
            for k in gained[:surplus]:
                h[k] -= 1; q[k] -= 1
    if RECONSTRUCTION >= 3:  # R6: products are recorded in their neutral form (convention of the data, chosen on training reactions)
        from scipy.sparse.csgraph import connected_components
        _, member = connected_components(after > 0, directed=False)
        for f in np.unique(member[touched]):
            atoms = np.nonzero(member == f)[0]
            if len(atoms) < NEUTRALISE_MIN_ATOMS:
                continue
            net = int(q[atoms].sum())
            for k in atoms:
                if net < 0 and q[k] < 0 and int(a["element"][k]) in (7, 8, 16) and not (q[after[k] > 0] > 0).any():
                    h[k] += 1; q[k] += 1; net += 1  # protonate an anion that is not part of a zwitterion pair
                elif net > 0 and q[k] > 0 and h[k] > 0 and int(a["element"][k]) in (7, 8, 15, 16) and not (q[after[k] > 0] < 0).any():
                    h[k] -= 1; q[k] -= 1; net -= 1  # deprotonate an onium ion
    mol = Chem.RWMol()
    for el, hh, qq in zip(a["element"], h, q):
        atom = Chem.Atom(int(el))
        atom.SetFormalCharge(int(qq)); atom.SetNumExplicitHs(int(hh)); atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for i, j in zip(*np.nonzero(np.triu(after, 1))):
        mol.AddBond(int(i), int(j), RD_BOND[int(after[i, j])])
        if after[i, j] == 4:
            for k in (i, j):
                mol.GetAtomWithIdx(int(k)).SetIsAromatic(True)
    out, spectators = set(), set()
    for atoms, frag in zip(Chem.GetMolFrags(mol), Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)):
        try:
            Chem.SanitizeMol(frag)
            (out if touched[list(atoms)].any() else spectators).add(canonical_product(Chem.MolToSmiles(frag)))
        except Exception:
            pass
    return out, out | spectators  # molecules a firing touched, and everything the final marking contains


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
    """USPTO-MIT (Jin et al. 2017): the standard forward-prediction benchmark, 409k / 30k / 40k reactions."""
    jobs = []
    for split, name in (("train", "train.txt"), ("val", "valid.txt"), ("test", "test.txt")):
        jobs += [(line, split, len(jobs) + k) for k, line in enumerate(open(Path(folder) / name)) if line.strip()]
    with Pool(12) as pool:
        data = [d for d in pool.map(_featurise_mit, jobs, chunksize=512) if d is not None]
    Path(out).write_bytes(pickle.dumps({"reactions": data, "classes": ["-"]}, protocol=4))
    for split in ("train", "val", "test"):
        rs = [d for d in data if d["split"] == split]
        print(f"{split}: {len(rs)} parsed, {sum(d['target'] is not None for d in rs)} with clean mapping and size <= {MAX_PRECURSOR_ATOMS}/{MAX_PRODUCT_ATOMS}")


def _mol(element, bonds, h, q):
    mol = Chem.RWMol()
    for el, hh, qq in zip(element, h, q):
        atom = Chem.Atom(int(el))
        atom.SetFormalCharge(int(qq)); atom.SetNumExplicitHs(int(max(hh, 0))); atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for i, j in zip(*np.nonzero(np.triu(bonds, 1))):
        mol.AddBond(int(i), int(j), RD_BOND[int(bonds[i, j])])
        if bonds[i, j] == 4:
            for k in (i, j):
                mol.GetAtomWithIdx(int(k)).SetIsAromatic(True)
    return mol


def mapping_from_marking(reaction, edits):
    """Atom mapping read off a firing vector: places keep their identity while tokens move, so if the marking
    after `edits` contains the recorded product, matching the two graphs gives product atom -> precursor atom.
    Returns None if the product is not reached. (Connectivity and elements are matched; bond orders follow.)"""
    a, b = reaction["a"], reaction["b"]
    after = dense_bonds(a).copy()
    for i, j, t in edits:
        after[i, j] = after[j, i] = t
    bare = lambda element, bonds: _mol(element, (bonds > 0).astype(int), np.zeros(len(element)), np.zeros(len(element)))
    whole, query = bare(a["element"], after), bare(b["element"], dense_bonds(b))
    for m in (whole, query):
        m.UpdatePropertyCache(strict=False); Chem.FastFindRings(m)
    pieces, query_pieces = [], []
    frags = Chem.GetMolFrags(whole, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=pieces)
    query_frags = Chem.GetMolFrags(query, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=query_pieces)
    mapping = np.full(len(b["element"]), -1, np.int64)
    for q, q_index in zip(query_frags, query_pieces):  # every product molecule has to be found among the fragments of the marking
        for frag, index in zip(frags, pieces):
            match = frag.GetSubstructMatch(q) if frag.GetNumAtoms() == q.GetNumAtoms() else ()
            if match:
                mapping[list(q_index)] = [index[k] for k in match]
                break
        else:
            return None
    return mapping if len(set(mapping)) == len(mapping) else None


def load(path="data/schneider50k.pkl"):
    return pickle.loads(Path(path).read_bytes())


def collate(reactions, device):
    """Pad a list of reactions into dense tensors."""
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
        "target": np.full((B, nb), -1, np.int64), "edits": np.zeros((B, na, na), np.int64),  # 0 = unchanged, k + 1 = new type k
        "label": np.array([r["label"] for r in reactions], np.int64),
        "labelled": np.array([r.get("labelled", True) for r in reactions], bool),
    }
    if "hist" in reactions[0]:  # histogram of fired transition types (auxiliary target, from the recorded mapping)
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
            out["edits"][k, i, j] = t + 1; out["edits"][k, j, i] = t + 1
    return {k: torch.as_tensor(v, device=device) for k, v in out.items()}


if __name__ == "__main__":
    import sys
    build_uspto_mit() if "uspto-mit" in sys.argv else build()
