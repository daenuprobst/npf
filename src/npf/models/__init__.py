"""Models for two tasks on attributed Petri nets.

transitions  given two markings A and B, predict the firing counts of every transition
next         given a marking, predict the marking one step ahead

Every model sees the same inputs and uses the same number of rounds.
"""

from .gnn import GNN
from .linear_pgnn import LinearPGNN
from .npf import NPF
from .pgnn import PGNN
from .state_equation_only import StateEquationOnly

__all__ = ["build", "GNN", "LinearPGNN", "NAMES", "NPF", "PGNN", "StateEquationOnly"]

NAMES = {
    "npf": "Neural Petri Flow",
    "npf-mlp": "NPF with a generic set function as rate law instead of the product form",
    "npf-prior": "NPF without the posterior correction",
    "npf-kl": "NPF whose projection is the I projection instead of its first linearised Newton step",
    "pgnn": "PGNN as published, Eqs. 9 to 12 with the signed aggregation of Eq. 13",
    "pgnn-eq10": "PGNN with Eq. 10 read literally, a place only hears its incoming transitions",
    "pgnn+": "PGNN with learned messages in both directions",
    "pgnn+se": "PGNN+ with the state equation projection on top",
    "gnn": "message passing on the place graph",
    "se-only": "state equation only, no learning",
}


def build(name, task, hidden=64, rounds=4):
    """name@k overrides the number of rounds, npf@16 plays the token game with 16 finer time steps."""
    if "@" in name:
        name, rounds = name.split("@")[0], int(name.split("@")[1])

    if name in ("npf", "npf-mlp", "npf-prior", "npf-kl"):
        return NPF(
            task,
            hidden,
            rounds,
            product_form=name != "npf-mlp",
            posterior=name != "npf-prior",
            divergence="kl" if name == "npf-kl" else "gauss",
        )

    if name == "gnn":
        return GNN(task, hidden, rounds)

    if name == "se-only":
        return StateEquationOnly()

    aggregate = {
        "pgnn": "incidence",
        "pgnn-eq10": "incoming",
        "pgnn+": "learned",
        "pgnn+se": "learned",
    }[name]

    return PGNN(
        task, hidden, rounds, aggregate=aggregate, state_equation=name == "pgnn+se"
    )
