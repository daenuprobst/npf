"""The electron net. The same net as the arrow net with one electron as the token instead of one pair, so that a
single electron can move and radical steps are expressible.

Every atom i has a nonbonding place N_i and a free octet place V_i, every atom pair a bond place B_ij, all counted in
electrons, so a lone pair is two tokens, an unpaired electron is one and a single bond is two. An arrow carries a
weight w of one or two, a fishhook or a full pair.

    a(i, j, w)   N_i -w   B_ij +w   V_j -w
    b(i, j, w)   B_ij -w  N_i +w    V_j +w

The octet budget N_i + sum_j B_ij + V_i = 2 cap_i of every atom is a P-invariant, so enabling is again the octet rule.
The formal charge doubles, 2 q_i = 2 e_i - 2 N_i - sum_j B_ij, and moves by w, so a fishhook shifts a charge by half a
unit inside a step and whole charges are back at its end. The arrow net of arrows.py is exactly the sub-net generated
by the transitions of weight two.
"""

import random
from collections import Counter
from functools import lru_cache

import numpy as np
from rdkit import Chem

from .arrows import (  # noqa: F401
    BOND,
    MAX_ORDER,
    PARSE,
    TABLE,
    A,
    B,
    scoring_form,
    tables,
    target_form,
)

MAX_WEIGHT = 2


def molecule(smiles):
    """Marking of mapped molecules as arrays indexed by map number minus one, z atomic numbers, q2 doubled formal
    charges, nb nonbonding electrons and be a dict from (i, j) with i < j to the electrons in the bond. Aromatic
    bonds are kekulised, so every bond place holds a whole number of electrons."""
    mol = Chem.MolFromSmiles(smiles, PARSE)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    n = mol.GetNumAtoms()
    index = np.array([a.GetAtomMapNum() - 1 for a in mol.GetAtoms()])
    assert sorted(index) == list(range(n)), "map numbers must be 1..n"

    z, q2, nb = (np.zeros(n, np.int64) for _ in range(3))
    be = {}
    for bond in mol.GetBonds():
        i, j = sorted((index[bond.GetBeginAtomIdx()], index[bond.GetEndAtomIdx()]))
        be[int(i), int(j)] = 2 * int(bond.GetBondTypeAsDouble())

    used = Counter()
    for (i, j), electrons in be.items():
        used[i] += electrons
        used[j] += electrons

    for atom in mol.GetAtoms():
        k = index[atom.GetIdx()]
        z[k], q2[k] = atom.GetAtomicNum(), 2 * atom.GetFormalCharge()

        # nonbonding electrons are what the valence shell keeps after the charge and the bonds
        free = TABLE.GetNOuterElecs(int(z[k])) - atom.GetFormalCharge() - used[k] // 2
        assert free >= 0, "more bonds than valence electrons"
        nb[k] = free

    return {"z": z, "q2": q2, "nb": nb, "be": be}


def step(line):
    """One elementary step 'reactants>>products|pathway' as two markings of the same atoms."""
    reaction, pathway = line.rsplit("|", 1)
    left, right = reaction.split(">>")
    m_a, m_b = molecule(left), molecule(right)
    assert np.array_equal(m_a["z"], m_b["z"]), "atoms differ between the two sides"

    return m_a, m_b, pathway


def shell(m):
    """Electrons in the valence shell of every atom, nonbonding and bonding alike."""
    s = m["nb"].copy()
    for (i, j), electrons in m["be"].items():
        s[i] += electrons
        s[j] += electrons

    return s


def apply(m, arrows):
    """The marking reached by firing the arrows, in any order (state equation)."""
    nb, q2, be = m["nb"].copy(), m["q2"].copy(), dict(m["be"])
    for kind, i, j, w in arrows:
        key = (min(i, j), max(i, j))
        sign = w if kind == A else -w
        nb[i] -= sign
        q2[i] += sign
        q2[j] -= sign
        be[key] = be.get(key, 0) + sign

        if not be[key]:
            del be[key]

    return {"z": m["z"], "q2": q2, "nb": nb, "be": be}


def _weigh(kind, i, j, electrons):
    """One end of a bond change as arrows, as few as possible, so a pair moves as one arrow and never as two."""
    out = [(kind, i, j, MAX_WEIGHT)] * (electrons // MAX_WEIGHT)

    return out + [(kind, i, j, 1)] * (electrons % MAX_WEIGHT)


def decompose(m_a, m_b):
    """All firing vectors of arrows that take m_a to m_b, as lists of (kind, i, j, w) without order. The electrons a
    bond place gains come from the nonbonding places of its two atoms and the ones it loses go to them, so the split
    between the two ends is enumerated per place, with pruning on the recorded nonbonding changes. Splits that keep
    pairs together come first, so the coarsest firing vector is found first."""
    places = sorted(set(m_a["be"]) | set(m_b["be"]))
    changes = [
        (i, j, m_b["be"].get((i, j), 0) - m_a["be"].get((i, j), 0)) for i, j in places
    ]
    changes = [(i, j, c) for i, j, c in changes if c]
    need = {k: v for k, v in enumerate((m_b["nb"] - m_a["nb"]).tolist()) if v}
    touches = Counter()
    for i, j, c in changes:
        touches[i] += abs(c)
        touches[j] += abs(c)

    # a nonbonding place can only change through a bond place, anything else is an electron transfer the net has not
    if any(k not in touches for k in need):
        return

    left = Counter(touches)

    def splits(c):
        """How many of the |c| electrons come from the first end, pairs kept together first."""
        even = [x for x in range(abs(c) + 1) if x % 2 == 0 and (abs(c) - x) % 2 == 0]

        return even + [x for x in range(abs(c) + 1) if x not in set(even)]

    def assign(k, balance):
        if k == len(changes):
            if all(
                balance.get(x, 0) == need.get(x, 0) for x in set(balance) | set(need)
            ):
                yield []

            return

        i, j, c = changes[k]
        kind, delta = (A, -1) if c > 0 else (B, 1)
        left[i] -= abs(c)
        left[j] -= abs(c)

        for take in splits(c):
            balance[i] = balance.get(i, 0) + delta * take
            balance[j] = balance.get(j, 0) + delta * (abs(c) - take)

            # every atom must still be able to reach its recorded change with the electrons that remain
            if all(abs(need.get(x, 0) - balance.get(x, 0)) <= left[x] for x in (i, j)):
                head = _weigh(kind, i, j, take) + _weigh(kind, j, i, abs(c) - take)
                for rest in assign(k + 1, balance):
                    yield head + rest

            balance[i] -= delta * take
            balance[j] -= delta * (abs(c) - take)

        left[i] += abs(c)
        left[j] += abs(c)

    yield from assign(0, {})


class Orders:
    """The enabled orders of one firing vector of arrows from m_a, with per atom arrays cap, lo and hi in doubled
    charge. The octet capacity holds after every arrow, a charge stays within its window widened by slack inside the
    step, and STOP needs every charge back inside its window and whole. The marking depends only on the multiset
    fired, so completability is memoised on the multiset that remains."""

    def __init__(self, m_a, arrows, cap, lo, hi, slack=2):
        self.arrows, self.m = tuple(sorted(arrows)), m_a
        self.cap, self.lo, self.hi = cap, lo, hi
        self.lo_in, self.hi_in = lo - slack, hi + slack
        self.filled = shell(m_a)
        self.completes = lru_cache(maxsize=None)(self._completes)

    def delta(self, rest):
        """Changes of nonbonding electrons, shells, doubled charges and bond electrons after firing all but rest."""
        dn, ds, dq, db = Counter(), Counter(), Counter(), Counter()
        for (kind, i, j, w), c in (Counter(self.arrows) - Counter(rest)).items():
            sign = c * w if kind == A else -c * w

            # the tail trades nonbonding electrons for bonding ones, so only the head of an arrow changes its shell
            dn[i] -= sign
            ds[j] += sign
            dq[i] += sign
            dq[j] -= sign
            db[min(i, j), max(i, j)] += sign

        return dn, ds, dq, db

    def enabled(self, arrow, delta):
        kind, i, j, w = arrow
        dn, ds, dq, db = delta
        q2 = self.m["q2"]
        electrons = (
            self.m["be"].get((min(i, j), max(i, j)), 0) + db[min(i, j), max(i, j)]
        )

        if kind == A:
            return (
                self.m["nb"][i] + dn[i] >= w
                and 2 * self.cap[j] - (self.filled[j] + ds[j]) >= w
                and electrons + w <= 2 * MAX_ORDER
                and q2[i] + dq[i] + w <= self.hi_in[i]
                and q2[j] + dq[j] - w >= self.lo_in[j]
            )

        return (
            electrons >= w
            and q2[i] + dq[i] - w >= self.lo_in[i]
            and q2[j] + dq[j] + w <= self.hi_in[j]
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
                self.lo[x] <= self.m["q2"][x] + d <= self.hi[x]
                and not (d + self.m["q2"][x]) % 2
                for x, d in dq.items()
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
    """Per atom octet capacity in pairs and doubled charge window from per element tables, never tighter than the
    atom's own start."""
    z = m["z"]
    cap = np.array([octet.get(int(e), 4) for e in z])
    lo = np.array([2 * window.get(int(e), (0, 0))[0] for e in z])
    hi = np.array([2 * window.get(int(e), (0, 0))[1] for e in z])
    pairs = np.ceil(shell(m) / 2).astype(np.int64)

    return np.maximum(cap, pairs), np.minimum(lo, m["q2"]), np.maximum(hi, m["q2"])


def to_mol(m):
    """RDKit molecule of a marking, every hydrogen an explicit atom and an odd nonbonding place a radical."""
    mol = Chem.RWMol()
    for z, q2, nb in zip(m["z"].tolist(), m["q2"].tolist(), m["nb"].tolist()):
        atom = Chem.Atom(z)
        atom.SetFormalCharge(q2 // 2)
        atom.SetNoImplicit(True)
        atom.SetNumRadicalElectrons(nb % 2)
        mol.AddAtom(atom)

    for (i, j), electrons in m["be"].items():
        mol.AddBond(int(i), int(j), BOND[electrons // 2])

    mol = mol.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)

    return mol
