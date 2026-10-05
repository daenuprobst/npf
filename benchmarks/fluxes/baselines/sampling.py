"""Mean of uniform flux samples over the culture polytope, the prior mean of constraint based flux inference.

The core model gets the constraints of the culture net, measured exchanges and growth fixed, knocked out reactions
closed, other exchanges secretion only except the inorganic ones that the core leaves open, ATP maintenance at 8.39
or at 0 if infeasible, and the flux bounds of the core model, 1000 in either direction, which the net lacks. It is
sampled with OptGP of cobrapy (Megchelenbrink et al. 2014 PLoS One 9 e86587). With --cyclefree the mean is replaced
by the closest loopless flux of CycleFreeFlux (Desouki et al. 2015 Bioinformatics 31 2159). Needs cobra.

    uv run --with cobra python -m benchmarks.fluxes.baselines.sampling --split all
"""

import math
from functools import lru_cache

from .. import data, flux
from . import common

SAMPLES, THINNING = 2000, 100


@lru_cache(maxsize=1)
def base():
    import cobra

    return cobra.io.model_from_dict(data.core())


def constrained(inst):
    """The core model with the constraints of the culture net."""
    model = base().copy()
    pinned, blocked = inst.culture.pinned, inst.culture.knockouts
    for r in model.reactions:
        if r.id in blocked:
            r.bounds = (0.0, 0.0)
        elif r.id == data.BIOMASS:
            r.bounds = (inst.growth, inst.growth)
        elif r.id in pinned:
            r.bounds = (pinned[r.id], pinned[r.id])
        elif r.id in data.UPTAKE_ONLY:
            r.bounds = (0.0, 0.0)
        elif r.id.startswith("EX_"):
            r.bounds = (-1000.0 if r.lower_bound <= -1000 else 0.0, max(r.upper_bound, 0.0))

    # the same maintenance bound as the maintenance place of the culture net
    model.reactions.ATPM.lower_bound = inst.atpm

    return None if math.isnan(model.slim_optimize(error_value=math.nan)) else model


def mean_flux(inst, seed, cyclefree):
    from cobra.flux_analysis.loopless import loopless_solution
    from cobra.sampling import sample

    model = constrained(inst)
    if model is None:
        return None

    # a culture whose fibre leaves the sampler fewer than three free directions is nearly a point, the least total
    # flux on the net stands in for its mean
    try:
        v = sample(model, SAMPLES, method="optgp", thinning=THINNING, processes=1, seed=seed).mean().to_dict()
    except ValueError:
        return common.cancel(inst, flux.pfba(inst))

    if cyclefree:
        # the sampler meets fixed bounds only to solver precision, CycleFreeFlux needs them exactly
        bounds = {r.id: r.bounds for r in model.reactions}
        v = {r: min(max(x, bounds[r][0]), bounds[r][1]) for r, x in v.items()}
        v = loopless_solution(model, fluxes=v).fluxes.to_dict()

    return common.to_sigma(inst, v)


def main():
    ap = common.parser(__doc__)
    ap.add_argument("--cyclefree", action="store_true")
    args = ap.parse_args()
    name = "sampling-cyclefree" if args.cyclefree else "sampling"
    note = f"OptGP, {SAMPLES} samples, thinning {THINNING}, core bounds 1000" + (", CycleFreeFlux" * args.cyclefree)
    common.evaluate(name, args, lambda train: lambda inst: mean_flux(inst, args.seed, args.cyclefree), note=note,
                    training_free=True)


if __name__ == "__main__":
    main()
