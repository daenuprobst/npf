"""Training targets from the net alone, and the scoring of a marking against the recorded product.

A reaction with a product but no atom map still has firing vectors, the minimum ones that the mapper of
npf.chem.mapper finds. Every optimal mapping gives one. Those with an enabled order that move at most MAX_TOKENS tokens
and decode to the recorded product are the targets of the token game, and the mapping that the third level chooses
serves the classifier.
"""

import time

import numpy as np

from . import cost, mapper
from .decode import NO_STEREO, canonical_product, marking_fragments, stereo_source
from .featurisation import dense_bonds
from .minimise import cost_of
from .orders import count_orders

# targets that move more tokens are mostly incomplete records, not trained on
MAX_TOKENS = 10

# at most this many firing vectors per reaction enter the loss
MAX_VECTORS = 8


def firing_vector(reaction, mapping):
    """The firings (i, j, new bond type) on precursor bond places that a product -> precursor mapping implies."""
    before, after = dense_bonds(reaction["a"]), np.zeros(
        (len(reaction["a"]["x"]),) * 2, np.int8
    )
    after[np.ix_(mapping, mapping)] = dense_bonds(reaction["b"])
    kept = np.zeros(len(before), bool)
    kept[mapping] = True
    i, j = np.nonzero(np.triu((after != before) & (kept[:, None] | kept[None, :]), 1))

    return np.stack([i, j, after[i, j]], 1).astype(np.int16)


def recorded_products(reaction, stereo=False):
    """Every recorded product molecule, canonical and without stereochemistry unless stereo, has to be predicted."""
    return {
        canonical_product(smi, stereo)
        for smi in reaction["smiles"].split(">>")[1].split(".")
    }


def product_found(reaction, edits, stereo=False):
    """The firings make the recorded major product and every recorded molecule is in the final marking, so the
    counter-ions of salts are spectators. With stereo the molecules are compared with their stereochemistry, which the
    marking takes from the precursors where no firing touched them and leaves open at the reaction centre."""
    source = None

    # precursors whose SMILES does not give their graph lend no stereo
    if stereo:
        source = stereo_source(reaction["smiles"].split(">>")[0], reaction["a"]) or NO_STEREO

    touched, everything = marking_fragments(reaction["a"], edits, source)
    recorded = reaction["smiles"].split(">>")[1]

    return (
        canonical_product(recorded, stereo) in touched
        and recorded_products(reaction, stereo) <= everything
    )


def product_major(reaction, edits):
    """The largest molecule the firings make is the recorded major product, one molecule chosen without the record, the
    way the largest molecule of a SMILES prediction is scored."""
    touched = marking_fragments(reaction["a"], edits)[0]

    return bool(touched) and canonical_product(
        ".".join(sorted(touched))
    ) == canonical_product(reaction["smiles"].split(">>")[1])


def targets(reaction, seconds=3.0, limit=64):
    """The mapping the third level chooses among those of minimum cost and the firing vectors of all of them that the
    token game can train on. seconds limits the wall time of the search, the mapping is None when no seating exists,
    and proved says whether the minimum and the listing of its ties were finished in time.
    """
    start = time.time()
    row = {
        "id": reaction["id"],
        "mapping": None,
        "vectors": [],
        "proved": False,
        "mappings": 0,
    }

    # a product with more atoms than its precursors cannot be seated, the record is incomplete
    if len(reaction["b"]["x"]) > len(reaction["a"]["x"]):
        return row | {"seconds": time.time() - start}

    try:
        maps, row["proved"] = mapper.cheapest_mappings(
            reaction, limit=limit, seconds=seconds, **cost.CHOSEN
        )
        row["mappings"], seen = len(maps), set()

        if maps:
            row["mapping"] = mapper.choose(reaction, maps).astype(np.int16)

        for mapping in maps:
            mapping = np.asarray(mapping, np.int64)

            sigma = firing_vector(reaction, mapping)
            key = sigma.tobytes()
            if key in seen:
                continue

            seen.add(key)

            if (
                cost_of(reaction, mapping) <= MAX_TOKENS
                and count_orders(reaction, sigma.astype(np.int64))[1]
                and product_found(reaction, sigma.astype(np.int64))
            ):
                row["vectors"].append(sigma)

            if len(row["vectors"]) >= MAX_VECTORS:
                break

        row["distinct"] = len(seen)
    except Exception as error:
        row["error"] = repr(error)[:200]

    return row | {"seconds": time.time() - start}


def attach(reaction, row, single=False):
    """The targets of a row on its reaction. edits_set holds the firing vectors of the token game, the first alone with
    single, target and edits the first mapping for the classifier, and a recorded map is dropped.
    """
    vectors = row["vectors"][:1] if single else row["vectors"]
    reaction["edits_set"] = vectors or None
    mapping = row["mapping"]

    if mapping is not None and len(mapping) == len(reaction["b"]["x"]):
        reaction["target"], reaction["edits"] = mapping.astype(np.int16), firing_vector(
            reaction, mapping.astype(np.int64)
        )
    else:
        reaction["target"], reaction["edits"] = None, vectors[0] if vectors else None

    return reaction
