"""From a marking of the valence net back to molecules."""

import numpy as np
from rdkit import Chem

from .featurisation import BOND_ORDER, HYPERVALENT, RD_BOND, dense_bonds

# version of the decoding rules, all chosen on training reactions only
# 1 is the rule set of the first Schneider 50k results, 2 adds R1 to R5, 3 adds the neutral product convention R6
RECONSTRUCTION = 3
NORMAL_VALENCE = {
    15: (3, 5),
    16: (2, 4, 6),
    34: (2, 4, 6),
    17: (1, 3, 5, 7),
    35: (1, 3, 5, 7),
    53: (1, 3, 5, 7),
}
ANION_PRIORITY = {53: 0, 35: 0, 17: 0, 9: 0, 8: 1, 16: 1, 7: 2}

# single atom ions such as halides and hydroxide stay as they are
NEUTRALISE_MIN_ATOMS = 2


def canonical_product(smiles):
    """Largest fragment, no stereochemistry, canonical. The form in which products are compared."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    Chem.RemoveStereochemistry(mol)

    return max(
        Chem.MolToSmiles(mol).split("."),
        key=lambda f: (Chem.MolFromSmiles(f).GetNumHeavyAtoms(), f),
    )


def marking_to_products(a, edits):
    return marking_fragments(a, edits)[0]


def marking_fragments(a, edits):
    """Molecules of the marking after the firings edits [(i, j, new bond type)], as sets of canonical SMILES, the
    fragments that hold a re-typed bond place and all fragments.

    A slack token is a hydrogen, a lone pair or a charge, and the marking does not say which, so the rules decide by
    conservation. The valence sum_j b_ij + h_i - q_i of every atom is a P-invariant. Tokens an atom cannot cover with
    hydrogens go to its charge, a new cation with a hydrogen loses the proton and a new aromatic cation takes it from
    an [nH] of its ring system. R1 S, P and halogens return tokens to lone pairs in steps of two. R2 and R3 in an
    aromatic system a surplus token pairs with the hydrogen of an [nH] and a deficit is covered by an aromatic n that
    takes one up, the bond order 1.5 hides these tautomer shifts. R5 charge is conserved, so for every new cation an
    atom that would have gained a hydrogen becomes an anion instead, as in N-oxides, nitro groups and the halides of
    quaternary salts. R6 products are recorded in their neutral form.
    """
    before = dense_bonds(a)
    after = before.copy()
    for i, j, t in edits:
        after[i, j] = after[j, i] = t

    # slack marking after the firings from the valence invariant. aromatic bonds count 1.5, hence the ceiling
    left = np.ceil(a["h"] - (BOND_ORDER[after] - BOND_ORDER[before]).sum(1) - 1e-6)

    # S, P and halogens cover a deficit from their lone pairs in steps of two, without a charge
    expand = np.array([HYPERVALENT.get(int(e), 0) for e in a["element"]])
    deficit = np.maximum(-left, 0)
    h, q = np.maximum(left, 0).astype(int), (
        a["q"] + deficit - np.minimum(deficit, expand) // 2 * 2
    ).astype(int)
    touched = np.zeros(len(h), bool)
    for i, j, _ in edits:
        touched[[i, j]] = True

    aromatic = (after == 4).any(1)

    def ring_system(i):
        seen, todo = {i}, [i]
        while todo:
            for k in np.nonzero(after[todo.pop()] == 4)[0]:
                if k not in seen:
                    seen.add(k)
                    todo.append(k)

        return seen

    if RECONSTRUCTION >= 2:
        for i in np.nonzero(touched & (h > a["h"]))[0]:
            allowed = NORMAL_VALENCE.get(int(a["element"][i]))
            bonds = BOND_ORDER[after[i]].sum()

            # R1 hypervalent elements give tokens back to their lone pairs, two at a time
            if allowed:
                target = min(
                    (v for v in allowed if v >= bonds - max(q[i], 0) - 1e-6),
                    default=allowed[-1],
                )
                while h[i] >= 2 and bonds + h[i] - q[i] - 2 >= target - 1e-6:
                    h[i] -= 2

            # R2 the surplus token of an aromatic carbon joins the pi system together with the hydrogen of an [nH]
            elif aromatic[i] and a["element"][i] == 6:
                donors = [
                    k
                    for k in ring_system(i)
                    if a["element"][k] == 7 and h[k] > 0 and q[k] == 0 and k != i
                ]
                if donors and h[i] > 0:
                    h[i] -= 1
                    h[donors[0]] -= 1

    # a new cation with a hydrogen loses the proton. an aromatic cation looks for an [nH] in its ring system
    for i in np.nonzero(touched & (q > 0))[0]:
        if h[i] > 0:
            h[i] -= 1
            q[i] -= 1
        elif aromatic[i]:
            donors = [
                k
                for k in ring_system(i)
                if a["element"][k] == 7 and h[k] > 0 and q[k] == 0
            ]
            if donors:
                h[donors[0]] -= 1
                q[i] -= 1

            # R3 or for an aromatic n that takes up a hydrogen, as in a lactam
            elif RECONSTRUCTION >= 2 and a["element"][i] == 6:
                takers = [
                    k
                    for k in ring_system(i)
                    if a["element"][k] == 7
                    and h[k] == 0
                    and q[k] == 0
                    and (after[k] > 0).sum() == 2
                ]
                if takers:
                    h[takers[0]] += 1
                    q[i] -= 1

    # R5 charge is conserved. atoms next to a new cation come first, then halogens, then O and S, then N
    if RECONSTRUCTION >= 2:
        surplus = int(q.sum() - a["q"].sum())
        if surplus > 0:
            new_cation = q > a["q"]
            gained = [
                k
                for k in np.nonzero(touched & (h > a["h"]) & (q == a["q"]))[0]
                if int(a["element"][k]) in ANION_PRIORITY
            ]
            gained.sort(
                key=lambda k: (
                    not new_cation[after[k] > 0].any(),
                    ANION_PRIORITY[int(a["element"][k])],
                )
            )

            for k in gained[:surplus]:
                h[k] -= 1
                q[k] -= 1

    # R6 touched fragments are written in their neutral form. anions outside a zwitterion pair are protonated and
    # onium ions with a hydrogen are deprotonated
    if RECONSTRUCTION >= 3:
        from scipy.sparse.csgraph import connected_components

        _, member = connected_components(after > 0, directed=False)
        for f in np.unique(member[touched]):
            atoms = np.nonzero(member == f)[0]
            if len(atoms) < NEUTRALISE_MIN_ATOMS:
                continue

            net = int(q[atoms].sum())
            for k in atoms:
                if (
                    net < 0
                    and q[k] < 0
                    and int(a["element"][k]) in (7, 8, 16)
                    and not (q[after[k] > 0] > 0).any()
                ):
                    h[k] += 1
                    q[k] += 1
                    net += 1
                elif (
                    net > 0
                    and q[k] > 0
                    and h[k] > 0
                    and int(a["element"][k]) in (7, 8, 15, 16)
                    and not (q[after[k] > 0] < 0).any()
                ):
                    h[k] -= 1
                    q[k] -= 1
                    net -= 1

    mol = Chem.RWMol()
    for el, hh, qq in zip(a["element"], h, q):
        atom = Chem.Atom(int(el))
        atom.SetFormalCharge(int(qq))
        atom.SetNumExplicitHs(int(hh))
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)

    for i, j in zip(*np.nonzero(np.triu(after, 1))):
        mol.AddBond(int(i), int(j), RD_BOND[int(after[i, j])])

        if after[i, j] == 4:
            for k in (i, j):
                mol.GetAtomWithIdx(int(k)).SetIsAromatic(True)

    out, spectators = set(), set()
    for atoms, frag in zip(
        Chem.GetMolFrags(mol), Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    ):
        try:
            Chem.SanitizeMol(frag)
            (out if touched[list(atoms)].any() else spectators).add(
                canonical_product(Chem.MolToSmiles(frag))
            )
        except Exception:
            pass

    # the molecules that a firing touched, and everything the final marking contains
    return out, out | spectators


def molecule(element, bonds, h, q):
    """RDKit molecule of a marking. Hydrogen counts and charges are explicit, so RDKit adds nothing on its own."""
    mol = Chem.RWMol()
    for el, hh, qq in zip(element, h, q):
        atom = Chem.Atom(int(el))
        atom.SetFormalCharge(int(qq))
        atom.SetNumExplicitHs(int(max(hh, 0)))
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)

    for i, j in zip(*np.nonzero(np.triu(bonds, 1))):
        mol.AddBond(int(i), int(j), RD_BOND[int(bonds[i, j])])

        if bonds[i, j] == 4:
            for k in (i, j):
                mol.GetAtomWithIdx(int(k)).SetIsAromatic(True)

    return mol
