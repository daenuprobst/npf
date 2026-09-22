"""The valence Petri net of a chemical reaction.

Places are bond places B_ij marked with the bond order and slack places S_i marked with the free valence of atom i.
Transitions move tokens between S_i, S_j and B_ij. The state equation is the Dugundji Ugi equation B + R = E,
the valence of every atom is a P-invariant, and enabling is the valence rule.
"""
from .classifier import Classifier
from .decode import canonical_product, marking_fragments, marking_to_products, molecule
from .encoder import N_BOND, Encoder, PetriLayer
from .featurisation import (BOND_ORDER, BOND_TYPE, ELEMENTS, EXTRA_CAPACITY, HYPERVALENT, MAX_PRECURSOR_ATOMS, MAX_PRODUCT_ATOMS, N_ATOM_FEAT,
                            RD_BOND, batch_indices, batches, build, build_uspto_mit, collate, dense_bonds, featurise, graph, load, reaction,
                            skeleton_classes)
from .mapping import map_reaction, mapped_smiles
from .minimise import branch_and_bound, cost_of, domains, feasible_start, token_descent, weighted_cost
from .orders import count_orders
from .one_shot import Forward
from .targets import MAX_TOKENS, MAX_VECTORS, attach, firing_vector, product_found, product_major, recorded_products, targets
from .token_game import TokenGame
from .verifier import Verifier

__all__ = ["BOND_ORDER", "BOND_TYPE", "build", "build_uspto_mit", "canonical_product", "Classifier", "collate",
           "dense_bonds", "ELEMENTS", "Encoder", "EXTRA_CAPACITY", "featurise", "Forward", "graph", "HYPERVALENT",
           "load", "marking_fragments", "marking_to_products",
           "MAX_PRECURSOR_ATOMS", "MAX_PRODUCT_ATOMS", "molecule", "N_ATOM_FEAT", "N_BOND", "PetriLayer", "RD_BOND",
           "skeleton_classes", "TokenGame", "Verifier"]
__all__ += ["branch_and_bound", "cost_of", "count_orders", "domains", "feasible_start", "token_descent", "weighted_cost"]
__all__ += ["attach", "batch_indices", "batches", "firing_vector", "map_reaction", "mapped_smiles", "MAX_TOKENS", "MAX_VECTORS",
            "product_found", "product_major", "reaction", "recorded_products", "targets"]
