"""Training targets from the net alone, and the scoring of a marking against the recorded product.

A reaction with a product but no atom map still has firing vectors, the minimum ones that the exact mapper finds.
Every optimal mapping gives one. Those with an enabled order that move at most MAX_TOKENS tokens and decode to the
recorded product are the targets of the token game, and the first mapping serves the classifier.
"""

import time

import numpy as np

from . import exact
from .decode import canonical_product, marking_fragments
from .featurisation import dense_bonds
from .minimise import cost_of, feasible_start
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


def recorded_products(reaction):
    """Every recorded product molecule (canonical, no stereochemistry) has to be predicted."""
    return {
        canonical_product(smi) for smi in reaction["smiles"].split(">>")[1].split(".")
    }


def product_found(reaction, edits):
    """The recorded major product is made by the firings, and every recorded product molecule (counter-ions of
    salts are spectators) is part of the final marking."""
    touched, everything = marking_fragments(reaction["a"], edits)
    recorded = reaction["smiles"].split(">>")[1]

    return (
        canonical_product(recorded) in touched
        and recorded_products(reaction) <= everything
    )


def product_major(reaction, edits):
    """The largest molecule that the firings make is the recorded major product. This commits to one molecule without
    looking at the record, the way the largest molecule of a SMILES prediction is scored, so both sides compare alike.
    """
    touched = marking_fragments(reaction["a"], edits)[0]

    return bool(touched) and canonical_product(
        ".".join(sorted(touched))
    ) == canonical_product(reaction["smiles"].split(">>")[1])


def targets(reaction, seconds=3.0, limit=64):
    """The first mapping of minimum cost and the firing vectors of all of them that the token game can train on.

    seconds is the budget in deterministic solver time, for the proof and again for listing the ties. The mapping is
    None when no seating exists, proved says whether the minimum was proved before the budget ran out.
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
        # a feasible seating by element and environment, the solver starts from it
        hint = feasible_start(reaction)
        maps, row["proved"] = exact.cheapest_mappings(
            reaction,
            limit=limit,
            seconds=4 * seconds,
            hint=hint,
            deterministic=seconds,
            **exact.CHOSEN,
        )
        row["mappings"], seen = len(maps), set()

        for mapping in maps:
            mapping = np.asarray(mapping, np.int64)

            if row["mapping"] is None:
                row["mapping"] = mapping.astype(np.int16)

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
    """The targets of a row on its reaction. edits_set holds the firing vectors of the token game (the first alone
    with single), target and edits the first mapping for the classifier. A recorded map is dropped.
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
