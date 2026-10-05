"""Minimisation of metabolic adjustment (Segre, Vitkup, Church 2002 PNAS 99 15112), quadratic form.

The wild type reference is pFBA of flux.py on the same culture with the knockouts undone, so reference and mutant
share the measured exchanges and growth. The mutant flux is the one closest to it in the Euclidean norm over the net
fluxes of all reactions left in the net. A culture without a blocked reaction in the core is its own wild type and
gets the pFBA flux. Needs cvxpy.

    uv run --with cvxpy python -m benchmarks.fluxes.baselines.moma --split all
"""

import dataclasses

from .. import data, flux
from . import common


def wild_type(inst):
    """Net fluxes of pFBA on the culture without its knockouts."""
    culture = dataclasses.replace(inst.culture, knockouts=set())
    wt = data.instance(common.model(), culture, inst.growth, inst.atpm)

    return common.net_flux(wt, flux.pfba(wt))


def reference(inst):
    """N and the wild type flux of every reaction left in the net of the culture."""
    ids, N = common.reactions(inst)
    w = wild_type(inst)

    return N, [w.get(r, 0.0) for r in ids]


def moma(inst):
    if not inst.culture.knockouts:
        return common.cancel(inst, flux.pfba(inst))

    N, w = reference(inst)

    # the Euclidean distance is strictly convex in the net fluxes, so the optimum is unique without a tie rule
    sigma = common.maintained(inst, lambda lo: common.project(inst, N, w, lo, alpha=0.0))

    return None if sigma is None else common.cancel(inst, sigma)


def main():
    args = common.parser(__doc__).parse_args()
    note = "quadratic MOMA to the pFBA wild type at the same measured exchanges and growth"
    common.evaluate("moma", args, lambda train: moma, note=note, training_free=True)


if __name__ == "__main__":
    main()
