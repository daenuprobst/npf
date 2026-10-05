"""Flux balance analysis, growth left free and maximised, ties broken by the least total flux (pFBA of Lewis et al.
2010 Mol Syst Biol 6 390).

The measured exchanges stay fixed as in every other method, only the biomass place is released, so the balance
residual of a result is the gap between predicted and measured growth. With --uptake-only the measured secretions
are released as well and only the substrate uptake is fixed, the classical FBA prediction from the medium alone.

    uv run python -m benchmarks.fluxes.baselines.fba --split all
    uv run python -m benchmarks.fluxes.baselines.fba --split all --uptake-only
"""

import numpy as np

from .. import data
from . import common


def secreted(inst):
    """Places of the measured by-products, those with a non-negative measured exchange."""
    by_id = {r["id"]: r for r in common.model()["reactions"]}
    out = []
    for rid, rate in inst.culture.pinned.items():
        if rid.startswith("EX_") and rate >= 0:
            (met,) = by_id[rid]["metabolites"]
            out.append(inst.places.index(met))

    return out


def fba(inst, uptake_only=False):
    C, b = inst.net.C, common.rhs(inst)
    biomass = inst.places.index("biomass")
    free = secreted(inst) if uptake_only else []
    fixed = [p for p in range(len(b)) if p not in free and p != biomass]
    growth = inst.reaction.index(data.BIOMASS)
    A_eq, b_eq = C[fixed], b[fixed]

    # a released by-product place may only gain tokens, it is secreted and never taken up
    A_ub, b_ub = -C[free], np.zeros(len(free))

    def solve(lo):
        c = np.zeros(inst.net.n_trans)
        c[growth] = -1.0
        x = common.lp(c, lo, A_eq, b_eq, A_ub if free else None, b_ub if free else None)
        if x is None:
            return None

        # pFBA second stage with growth held at its maximum
        row = np.zeros((1, inst.net.n_trans))
        row[0, growth] = -1.0
        x2 = common.least_total(lo, A_eq, b_eq, np.vstack([A_ub, row]), np.append(b_ub, -common.tie(x[growth], -1e-7)))

        return common.cancel(inst, x if x2 is None else x2)

    return common.maintained(inst, solve)


def main():
    ap = common.parser(__doc__)
    ap.add_argument("--uptake-only", action="store_true")
    args = ap.parse_args()
    name = "fba-uptake" if args.uptake_only else "fba"
    note = ("max growth then least total flux, growth free, "
            + ("only substrate uptake fixed, secretions free" if args.uptake_only else "all measured exchanges fixed"))
    common.evaluate(name, args, lambda train: lambda inst: fba(inst, args.uptake_only), note=note, training_free=True)


if __name__ == "__main__":
    main()
