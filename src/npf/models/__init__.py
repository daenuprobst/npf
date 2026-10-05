"""Models of the general net.

transitions  given two markings A and B, predict the firing counts of every transition, every model
next         given a marking, predict the marking one time unit ahead, NPF only

Every model sees the same inputs and uses the same number of rounds.
"""

from .npf import NPF
from .pgnn import PGNN
from .state_equation_only import StateEquationOnly

__all__ = ["build", "NAMES", "NPF", "PGNN", "StateEquationOnly"]

NAMES = {
    "npf": "Neural Petri Flow",
    "pgnn": "PGNN as published, Eqs. 9 to 12 with the signed aggregation of Eq. 13",
    "pgnn+": "PGNN with learned messages in both directions",
    "pgnn+se": "PGNN+ with the state equation projection on top",
    "se-only": "state equation only, no learning",
    "npf-k": "NPF with a rate constant per transition of a fixed net, the other models take -k as well",
    "npf-gma": "NPF with rate constants and kinetic orders on read arcs from the boundary, generalised mass action",
    "npf-gma-row": "npf-gma with its per-transition terms restricted to im C^T, the gauge ablation",
    "pgnn-gma": "PGNN with the same rate constants and boundary environment as npf-gma",
    "pgnn+se-gma": "PGNN+ with the state equation projection and the same information as npf-gma",
}


def build(name, task, hidden=64, rounds=4, n_ids=0, n_env=0, kl_options=None):
    """n_ids and n_env size the rate constants and read arcs of the -k and -gma variants. kl_options make every
    projection the I projection with these settings, whose output is a positive firing count vector on the state
    equation.
    """
    gauge = "row" if name.endswith("-row") else None
    name = name.removesuffix("-row")
    ids = n_ids if name.endswith(("-k", "-gma")) else 0
    env = n_env if name.endswith("-gma") else 0
    name = name.removesuffix("-k").removesuffix("-gma")

    if name == "npf":
        return NPF(task, hidden, rounds, n_ids=ids, n_env=env, gauge=gauge, kl_options=kl_options)

    if task != "transitions":
        raise ValueError(f"{name} predicts firing counts only, task {task} needs npf")

    if name == "se-only":
        return StateEquationOnly(kl_options)

    aggregate = {"pgnn": "incidence", "pgnn+": "learned", "pgnn+se": "learned"}[name]

    return PGNN(
        hidden, rounds, aggregate=aggregate, state_equation=name == "pgnn+se", n_ids=ids, n_env=env,
        kl_options=kl_options,
    )
