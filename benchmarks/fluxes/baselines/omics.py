"""Transcript and protein levels of the Ishii et al. 2007 cultures on the reactions of the core model.

Only Ishii 2007 measured omics, so the omics methods predict only its cultures and skip every other one. Transcripts
are qRT-PCR copies per ug total RNA of 85 genes, proteins LC-MS/MS mg per gDW of about 60 enzymes. A culture measured
in several series gets the mean over the series. RF06 has no transcripts, RF05 and RF06 have no proteins. Gene
levels become reaction levels through the gene rules of the core model with a function for and and one for or, genes
without a value are ignored and a rule without any measured gene gives NaN, as gene_to_reaction_levels of Machado
and Herrgard 2014 (github.com/cdanielmachado/transcript2flux).
"""

import ast
import csv
import math
from functools import lru_cache

import numpy as np

from .. import data
from . import common

KINDS = {"transcripts": "rna", "proteins": "protein"}


@lru_cache(maxsize=2)
def table(kind):
    """Levels by condition and b-number, averaged over series."""
    sums = {}
    for r in csv.DictReader(open(data.ROOT / f"ishii2007_{kind}.csv")):
        if r["value"] in ("", "nan") or not r["bnumber"]:
            continue

        s, n = sums.setdefault(r["condition"], {}).get(r["bnumber"], (0.0, 0))
        sums[r["condition"]][r["bnumber"]] = (s + float(r["value"]), n + 1)

    return {c: {g: s / n for g, (s, n) in genes.items()} for c, genes in sums.items()}


def genes(inst, kind):
    """Gene levels of a culture, None for a culture without them."""
    if inst.culture.dataset != "ishii2007":
        return None

    return table(kind).get(inst.culture.name)


def rule_level(rule, levels, f_and, f_or):
    """Level of a gene rule, NaN operands are skipped."""
    if not rule.strip():
        return math.nan

    def walk(node):
        if isinstance(node, ast.Name):
            return levels.get(node.id, math.nan)

        values = [x for x in (walk(v) for v in node.values) if not math.isnan(x)]
        if not values:
            return math.nan

        return f_and(values) if isinstance(node.op, ast.And) else f_or(values)

    return walk(ast.parse(rule, mode="eval").body)


def reaction_levels(levels, f_and=min, f_or=max):
    """Level of every core reaction by id, NaN where no gene of its rule was measured."""
    return {r["id"]: rule_level(r["gene_reaction_rule"], levels, f_and, f_or) for r in common.model()["reactions"]}


def parser(doc):
    ap = common.parser(doc)
    ap.add_argument("--omics", choices=tuple(KINDS), default="transcripts")

    return ap


def evaluate(method, args, predict, note):
    """Scores predict(inst, gene levels) on the Ishii cultures of every fold, all other cultures are skipped."""
    def one(inst):
        levels = genes(inst, args.omics)

        return None if levels is None else predict(inst, levels)

    common.evaluate(f"{method}-{KINDS[args.omics]}", args, lambda train: one, training_free=True,
                    note=f"{note}, {args.omics} of Ishii 2007, other cultures skipped")


def quantile(levels, q):
    return float(np.quantile(list(levels.values()), q))


def template(inst):
    """The core model as E-Flux2 and SPOT use it, reaction ids, S and the bounds.

    The DC template of Kim et al. 2016, glucose uptake unbounded and other carbon uptakes closed as in the core. The
    measured by-products are open, the unmeasured ones as the culture net has them, open by default and closed with
    METABOLISM_CLOSE_CARBON=1. The
    knockouts of the culture are closed. The ATP maintenance bound is dropped because expression levels carry no
    unit that 8.39 mmol/gDW/h could be compared with, as in the E-Flux wrapper of Machado and Herrgard 2014.
    """
    model = common.model()
    ids = [r["id"] for r in model["reactions"]]
    mets = {m["id"]: k for k, m in enumerate(model["metabolites"])}
    S = np.zeros((len(mets), len(ids)))
    for j, r in enumerate(model["reactions"]):
        for m, c in r["metabolites"].items():
            S[mets[m], j] = c

    lb = np.array([r["lower_bound"] for r in model["reactions"]], float)
    ub = np.array([r["upper_bound"] for r in model["reactions"]], float)
    lb[ids.index("EX_glc__D_e")] = -np.inf
    lb[ids.index("ATPM")] = 0.0

    # the added substrates are offered only to the cultures that grew on them
    for r in data.UPTAKE_ONLY:
        if r not in inst.culture.pinned:
            lb[ids.index(r)] = 0.0

    # the carbon by-products that the study measured may be secreted, the core closes the others
    for r, rate in inst.culture.pinned.items():
        if r.startswith("EX_") and rate > 0:
            ub[ids.index(r)] = 1000.0

    for r in inst.culture.knockouts:
        lb[ids.index(r)] = ub[ids.index(r)] = 0.0

    return ids, S, lb, ub


def scaled(inst, ids, v):
    """Fluxes in arbitrary units scaled to the measured glucose uptake, as firing counts of the culture net."""
    uptake = -v[ids.index("EX_glc__D_e")]
    if not uptake > 1e-9:
        return None

    factor = -inst.culture.pinned["EX_glc__D_e"] / uptake

    return common.to_sigma(inst, dict(zip(ids, factor * v)))
