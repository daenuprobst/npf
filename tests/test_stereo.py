"""Products compared with their stereochemistry, which the marking copies from the precursors away from the reaction
centre."""
import itertools

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from npflow import chem
from npflow.chem.decode import CHIRAL, _odd, stereo_source


def firing(smiles):
    """A reaction and the first firing vector of the mapper."""
    r = chem.reaction(smiles)
    vectors = chem.targets(r)["vectors"]
    assert vectors, smiles

    return r, vectors[0]


def mirror(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() in CHIRAL:
            atom.InvertChirality()

    return Chem.MolToSmiles(mol)


def test_parity_of_a_permutation():
    for perm in itertools.permutations(range(4)):
        inversions = sum(perm[i] > perm[j] for i in range(4) for j in range(i + 1, 4))
        assert _odd([0, 1, 2, 3], list(perm)) == inversions % 2


@pytest.mark.parametrize(
    "precursor", ["C[C@H](O)c1ccccc1", "C[C@@H](O)c1ccccc1", "OC[C@@H](N)Cc1ccccc1"]
)
def test_untouched_stereocentre_keeps_its_configuration(precursor):
    # the product of a template keeps the configuration of the carbon next to the acylated oxygen
    template = AllChem.ReactionFromSmarts("[C:1][OH:2].[C:3](=[O:4])Cl>>[C:3](=[O:4])[O:2][C:1]")
    product = template.RunReactants((Chem.MolFromSmiles(precursor), Chem.MolFromSmiles("CC(=O)Cl")))[0][0]
    Chem.SanitizeMol(product)
    product = Chem.MolToSmiles(product)

    for order in (f"{precursor}.CC(=O)Cl", f"CC(=O)Cl.{precursor}"):
        r, edits = firing(f"{order}>>{product}")
        assert chem.product_found(r, edits, stereo=True)
        assert chem.product_found(r, edits)

        # the mirror image is the same molecule without stereo only
        wrong = chem.reaction(f"{order}>>{mirror(product)}")
        assert not chem.product_found(wrong, edits, stereo=True)
        assert chem.product_found(wrong, edits)


@pytest.mark.parametrize("vinyl,product,right", [
    ("C/C=C/Br", "C/C=C/c1ccccc1", True),
    ("C/C=C/Br", "C/C=C\\c1ccccc1", False),
    ("C/C=C\\Br", "C/C=C\\c1ccccc1", True),
    ("CC/C(C)=C(/C)Br", "CC/C(C)=C(/C)c1ccccc1", True),
])
def test_cross_coupling_retains_the_double_bond(vinyl, product, right):
    # the aryl group takes the place of the bromide on the same side of the double bond
    r, edits = firing(f"{vinyl}.OB(O)c1ccccc1>>{product}")
    assert chem.product_found(r, edits, stereo=True) == right
    assert chem.product_found(r, edits)


def test_no_stereo_at_the_reaction_centre():
    # the configuration of a carbon that a firing touched is left open, so an inversion is never predicted
    r, edits = firing("C[C@H](Br)CC.[OH-]>>C[C@@H](O)CC")
    assert chem.product_found(r, edits)
    assert not chem.product_found(r, edits, stereo=True)
    assert "CCC(C)O" in chem.marking_fragments(r["a"], edits, stereo_source("C[C@H](Br)CC.[OH-]", r["a"]))[0]


def test_stereo_source_refuses_a_graph_it_does_not_give():
    r = chem.reaction("C[C@H](O)c1ccccc1.CC(=O)Cl>>CC(=O)O[C@@H](C)c1ccccc1")
    assert stereo_source("C[C@H](O)c1ccccc1.CC(=O)Cl", r["a"]) is not None
    assert stereo_source("C[C@H](N)c1ccccc1.CC(=O)Cl", r["a"]) is None


def test_stereo_changes_nothing_without_stereo(schneider):
    # reactions whose SMILES hold no stereo are scored alike with and without it, and the flag off is the old rule
    for r in schneider["reactions"][:120]:
        if r["edits"] is None or any(c in r["smiles"] for c in "@/\\"):
            continue

        edits = np.asarray(r["edits"], np.int64)
        assert chem.product_found(r, edits, stereo=True) == chem.product_found(r, edits)
        assert chem.marking_fragments(r["a"], edits) == chem.marking_fragments(r["a"], edits, None)
