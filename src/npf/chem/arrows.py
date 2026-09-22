"""The arrow net. The electron pairs of a set of molecules as a Petri net whose firings are the arrows of arrow pushing.

Every atom i has two places, its lone pairs L_i and its free octet slots V_i, and every atom pair has a bond place B_ij
holding the bond order. An arrow a(i, j) moves a lone pair of i into the bond ij, an arrow b(i, j) moves a pair of the
bond ij onto i as a lone pair. A bond pair that moves to another bond is b followed by a, so two kinds suffice.

    a(i, j)   L_i -1   B_ij +1   V_j -1
    b(i, j)   B_ij -1  L_i +1    V_j +1

The octet budget L_i + sum_j B_ij + V_i of every atom is a P-invariant, so enabling is the octet rule, an arrow may
only end at an atom with a free slot. The formal charge q_i = e_i - 2 L_i - sum_j B_ij is linear in the marking. A
window lo <= q <= hi per element holds whenever a step ends, and inside a step, whose arrows are concerted, a charge may
leave it by one. A bond place holds at most a triple bond. Arrows move pairs, so an unpaired electron is a constant of
the marking.
"""

import random
from collections import Counter
from functools import lru_cache

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
PARSE = Chem.SmilesParserParams()
PARSE.removeHs = False
PARSE.sanitize = True
TABLE = Chem.GetPeriodicTable()
BOND = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
A, B = 0, 1
MAX_ORDER = 3

# elements whose charge window always includes -1 and +1, the charges that arrows make inside a step
NONMETALS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 52, 53}


def molecule(smiles):
    """Marking of mapped molecules, as arrays indexed by map number minus one.

    Returns z (atomic numbers), q (formal charges), lone (lone pairs), odd (an unpaired electron, 0 or 1), bonds
    {(i, j): order} with i < j. Aromatic bonds are written as a Kekule structure, so every bond place holds a whole
    number of pairs.
    """
    mol = Chem.MolFromSmiles(smiles, PARSE)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    n = mol.GetNumAtoms()
    index = np.array([a.GetAtomMapNum() - 1 for a in mol.GetAtoms()])
    assert sorted(index) == list(range(n)), "map numbers must be 1..n"

    z, q, lone, odd = (np.zeros(n, np.int64) for _ in range(4))
    bonds = {}
    for bond in mol.GetBonds():
        i, j = sorted((index[bond.GetBeginAtomIdx()], index[bond.GetEndAtomIdx()]))
        bonds[int(i), int(j)] = int(bond.GetBondTypeAsDouble())

    used = Counter()
    for (i, j), order in bonds.items():
        used[i] += order
        used[j] += order

    for atom in mol.GetAtoms():
        k = index[atom.GetIdx()]
        z[k], q[k] = atom.GetAtomicNum(), atom.GetFormalCharge()

        # nonbonding electrons are what the valence shell keeps after the charge and the bonds
        free = TABLE.GetNOuterElecs(int(z[k])) - q[k] - used[k]
        assert free >= 0, "more bonds than valence electrons"
        lone[k], odd[k] = free // 2, free % 2

    return {"z": z, "q": q, "lone": lone, "odd": odd, "bonds": bonds}


def step(line):
    """One elementary step 'reactants>>products|pathway' as two markings of the same atoms."""
    reaction, pathway = line.rsplit("|", 1)
    left, right = reaction.split(">>")
    m_a, m_b = molecule(left), molecule(right)
    assert np.array_equal(m_a["z"], m_b["z"]), "atoms differ between the two sides"
    assert np.array_equal(m_a["odd"], m_b["odd"]), "a single electron moves"

    return m_a, m_b, pathway


def shell(m):
    """Orbitals in the valence shell of every atom that hold electrons, lone pairs, bonds and an unpaired electron."""
    s = m["lone"] + m["odd"]
    for (i, j), order in m["bonds"].items():
        s[i] += order
        s[j] += order

    return s


def apply(m, arrows):
    """The marking reached by firing the arrows, in any order (state equation)."""
    lone, q, bonds = m["lone"].copy(), m["q"].copy(), dict(m["bonds"])
    for kind, i, j in arrows:
        key = (min(i, j), max(i, j))
        sign = 1 if kind == A else -1
        lone[i] -= sign
        q[i] += sign
        q[j] -= sign
        bonds[key] = bonds.get(key, 0) + sign

        if not bonds[key]:
            del bonds[key]

    return {"z": m["z"], "q": q, "lone": lone, "odd": m["odd"], "bonds": bonds}


def decompose(m_a, m_b):
    """All firing vectors of arrows that take m_a to m_b, as lists of (kind, i, j), without order.

    Every pair a bond gains came from a lone pair of one of its two atoms, every pair it loses went to one of them,
    and the lone pairs of each atom must change as recorded. The two choices per pair are enumerated with pruning.
    """
    places = set(m_a["bonds"]) | set(m_b["bonds"])
    units = []

    for i, j in sorted(places):
        change = m_b["bonds"].get((i, j), 0) - m_a["bonds"].get((i, j), 0)
        units += [(A if change > 0 else B, i, j)] * abs(change)

    need = {k: v for k, v in enumerate((m_b["lone"] - m_a["lone"]).tolist()) if v}
    touches = Counter(x for _, i, j in units for x in (i, j))

    # a lone pair can only change through a bond place, anything else is an electron transfer the net does not have
    if any(k not in touches for k in need):
        return

    left = Counter(touches)

    def assign(k, balance):
        if k == len(units):
            if all(
                balance.get(x, 0) == need.get(x, 0) for x in set(balance) | set(need)
            ):
                yield []

            return

        kind, i, j = units[k]
        left[i] -= 1
        left[j] -= 1
        delta = -1 if kind == A else 1
        for end, other in ((i, j), (j, i)):
            balance[end] = balance.get(end, 0) + delta

            # every atom must still be able to reach its recorded change with the units that remain
            if all(abs(need.get(x, 0) - balance.get(x, 0)) <= left[x] for x in (i, j)):
                for rest in assign(k + 1, balance):
                    yield [(kind, end, other)] + rest

            balance[end] -= delta

        left[i] += 1
        left[j] += 1

    yield from assign(0, {})


class Orders:
    """The enabled orders of one firing vector of arrows from m_a. cap, lo and hi are per atom arrays.

    The octet capacity holds after every arrow, a charge stays within its window widened by slack inside the step, and
    STOP is enabled only when every charge is back inside its window. The marking reached depends only on the multiset
    fired so far (state equation), so completability is memoised on the multiset that remains.
    """

    def __init__(self, m_a, arrows, cap, lo, hi, slack=1):
        self.arrows, self.m = tuple(sorted(arrows)), m_a
        self.cap, self.lo, self.hi = cap, lo, hi
        self.lo_in, self.hi_in = lo - slack, hi + slack
        self.pairs = shell(m_a)
        self.completes = lru_cache(maxsize=None)(self._completes)

    def delta(self, rest):
        """Changes of lone pairs, shells, charges and bond orders after firing everything except rest."""
        dl, ds, dq, db = Counter(), Counter(), Counter(), Counter()
        for (kind, i, j), c in (Counter(self.arrows) - Counter(rest)).items():
            sign = c if kind == A else -c
            dl[i] -= sign
            ds[i] += sign
            ds[j] += sign
            dq[i] += sign
            dq[j] -= sign
            db[min(i, j), max(i, j)] += sign

        return dl, ds, dq, db

    def enabled(self, arrow, delta):
        kind, i, j = arrow
        dl, ds, dq, db = delta
        q = self.m["q"]
        order = (
            self.m["bonds"].get((min(i, j), max(i, j)), 0) + db[min(i, j), max(i, j)]
        )

        if kind == A:
            return (
                self.m["lone"][i] + dl[i] >= 1
                and self.cap[j] - (self.pairs[j] + ds[j]) >= 1
                and order < MAX_ORDER
                and q[i] + dq[i] + 1 <= self.hi_in[i]
                and q[j] + dq[j] - 1 >= self.lo_in[j]
            )

        return (
            order >= 1
            and q[i] + dq[i] - 1 >= self.lo_in[i]
            and q[j] + dq[j] + 1 <= self.hi_in[j]
        )

    def moves(self, rest):
        """Distinct arrows of rest that are enabled now and after which the rest can still be fired to the end."""
        delta, out = self.delta(rest), []
        for k, arrow in enumerate(rest):
            if k and rest[k - 1] == arrow:
                continue

            if self.enabled(arrow, delta) and self.completes(rest[:k] + rest[k + 1 :]):
                out.append(arrow)

        return out

    def _completes(self, rest):
        if not rest:
            dq = self.delta(rest)[2]
            return all(
                self.lo[x] <= self.m["q"][x] + d <= self.hi[x] for x, d in dq.items()
            )

        return bool(self.moves(rest))

    def walk(self, pick):
        """A full enabled sequence, choosing among the valid moves with pick, or None if there is none."""
        if not self.completes(self.arrows):
            return None

        rest, seq = self.arrows, []
        while rest:
            arrow = pick(self.moves(rest))
            seq.append(arrow)
            k = rest.index(arrow)
            rest = rest[:k] + rest[k + 1 :]

        return seq

    def first(self):
        return self.walk(lambda moves: moves[0])

    def sample(self, rng=random):
        return self.walk(rng.choice)


def limits(m, octet, window):
    """Per atom octet capacity and charge window from per element tables, never tighter than the atom's own start."""
    z = m["z"]
    cap = np.array([octet.get(int(e), 4) for e in z])
    lo = np.array([window.get(int(e), (0, 0))[0] for e in z])
    hi = np.array([window.get(int(e), (0, 0))[1] for e in z])

    return np.maximum(cap, shell(m)), np.minimum(lo, m["q"]), np.maximum(hi, m["q"])


def tables(shells, charges, least=10):
    """Per element octet capacities and charge windows from counts over training markings.

    shells and charges map an atomic number to a Counter of observed values. A value counts if seen at least least
    times, so that single annotation errors do not widen the net.
    """
    octet, window = {}, {}
    for z, seen in shells.items():
        kept = [s for s, c in seen.items() if c >= least] or [min(seen)]
        octet[z] = max(kept)

    for z, seen in charges.items():
        kept = [q for q, c in seen.items() if c >= least] or [0]

        if z in NONMETALS:
            kept += [-1, 1]

        window[z] = (min(kept), max(kept))

    return octet, window


def to_mol(m):
    """RDKit molecule of a marking, every hydrogen an explicit atom."""
    mol = Chem.RWMol()
    for z, q in zip(m["z"].tolist(), m["q"].tolist()):
        atom = Chem.Atom(z)
        atom.SetFormalCharge(q)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)

    for (i, j), order in m["bonds"].items():
        mol.AddBond(int(i), int(j), BOND[order])

    mol = mol.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)

    return mol


def _standard(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)


def scoring_form(mol):
    """The string FlowER compares, no atom maps, no stereochemistry, all hydrogens written, after a parse round trip
    with sanitisation. None if RDKit rejects the molecule."""
    try:
        again = Chem.MolFromSmiles(_standard(mol), PARSE)
        return _standard(again) if again is not None else None
    except Exception:
        return None


def target_form(smiles):
    """scoring_form of a recorded product side."""
    mol = Chem.MolFromSmiles(smiles, PARSE)

    return scoring_form(mol) if mol is not None else None
