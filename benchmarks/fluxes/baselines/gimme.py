"""GIMME (Becker and Palsson 2008 PLoS Comput Biol 4 e1000082) on the culture net, Ishii 2007 cultures only.

As in the sim_intra test of Machado and Herrgard 2014, growth and measured exchanges are fixed (their objective
fraction 1), reaction levels take the minimum over and and the maximum over or, the threshold is the lower quartile
of the gene levels of the culture, and GIMME minimises the flux through reactions below it weighted by the distance
to it. Their solver returned an arbitrary optimum, here the least total flux among the optima is taken.

    uv run python -m benchmarks.fluxes.baselines.gimme --split all --omics transcripts
    uv run python -m benchmarks.fluxes.baselines.gimme --split all --omics proteins
"""

import math

import numpy as np

from . import common, omics

QUANTILE = 0.25


def gimme(inst, levels):
    theta = omics.quantile(levels, QUANTILE)
    level = omics.reaction_levels(levels, min, max)

    # both directions of a reaction below the threshold pay per unit of flux
    level = {r: level.get(r, math.nan) for r in inst.reaction}
    c = np.array([0.0 if math.isnan(level[r]) else max(theta - level[r], 0.0) for r in inst.reaction])
    C, b = inst.net.C, common.rhs(inst)

    def solve(lo):
        x = common.lp(c, lo, C, b)
        if x is None:
            return None

        x2 = common.least_total(lo, C, b, c[None], [common.tie(c @ x)])

        return x if x2 is None else x2

    sigma = common.maintained(inst, solve)

    return None if sigma is None else common.cancel(inst, sigma)


def main():
    args = omics.parser(__doc__).parse_args()
    omics.evaluate("gimme", args, gimme, note="GIMME with growth and exchanges fixed, lower quartile threshold")


if __name__ == "__main__":
    main()
