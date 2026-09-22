"""The model registry of the general nets."""
from npf import models
from npf.models.learned_incidence import SheafNPF

NAMES = ("npf", "npf-mlp", "npf-prior", "pgnn", "pgnn-eq10", "pgnn+", "pgnn+se", "gnn", "se-only", "npf-kl")


def test_names_cover_build():
    for name in models.NAMES:
        models.build(name, "transitions", hidden=8)

    assert set(models.NAMES) == set(NAMES)


def test_sheaf_npf_signature_change():
    """The first positional argument is n_types. A positional call with an older meaning would be silent."""
    assert SheafNPF(n_types=3).raw_ratio.shape == (3, 3)
    assert SheafNPF().raw_ratio.shape == (4, 4) and SheafNPF().consistency is True
