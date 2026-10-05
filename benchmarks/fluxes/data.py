"""The E. coli core model as a Petri net per culture, and the 13C flux measurements it is scored on.

Places are the metabolites, one biomass place and one maintenance place, transitions the reactions, a reversible
reaction as a forward and a backward transition so that every firing count is non-negative. At steady state the
internal metabolites do not change between the two markings, an external metabolite whose exchange was measured changes
by the measured rate and the biomass place by the growth rate. Its exchange transition is then removed, so
C sigma = m_B - m_A is the flux balance of the culture. Exchanges that were not measured stay as sink transitions, and
for water, protons, oxygen, carbon dioxide, ammonium and phosphate also as source transitions, whose firing is latent.
Knocked out reactions lose their transitions. With METABOLISM_CLOSE_CARBON=1 in the environment no carbon by-product
that a study did not measure is secreted, the rule of the check of the sources in data/metabolism/raw/README.md, a
variant the paper reports beside the default.

The net is the core model with the reactions of iJO1366 that the 13C studies measure or need, the Entner-Doudoroff
pathway, cytochrome bo3 and the uptake of galactose, gluconate and glycerol, with the periplasm merged into the medium.
Acetaldehyde, which no source model exchanges, is not secreted. ATP maintenance feeds a maintenance place whose marking
rises by the bound of the core model, 8.39 mmol/gDW/h, and a free drain empties it, so the state equation alone forces
ATPM >= 8.39. Where a culture cannot meet the bound at its measured growth the bound is 0, and where it cannot meet its
measured growth at all the growth is lowered to the largest one the net allows. Transitions that fire in no run from
marking A to marking B, dead for this pair of markings, are removed, so that every transition left can fire and the
fibre {sigma >= 0, C sigma = m_B - m_A} has a point that is positive everywhere. The same nets serve every method.

The tables in data/metabolism come from the five studies listed in data/metabolism/raw/README.md.
"""

import csv
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from npf.nets import Net

ROOT = Path("data/metabolism")
CORE = ROOT / "raw/bigg/e_coli_core.json"
GENOME = ROOT / "raw/bigg/iJO1366.json"
BIOMASS = "BIOMASS_Ecoli_core_w_GAM"
DATASETS = ("ishii2007", "long2019", "haverkorn2011", "holm2010", "gerosa2015")
ATPM = 8.39

# reactions of iJO1366 that the core lacks, Entner-Doudoroff, cytochrome bo3, and the uptake routes of Gerosa et al.
ADDED = ("EDD", "EDA", "CYTBO3_4pp", "GALabcpp", "GALKr", "UGLT", "UDPG4E", "PGMT", "GLCNt2rpp", "GNK", "GLYCtpp",
         "GLYK", "G3PD5")

# substrates of the added routes, taken up only when the culture grew on them
UPTAKE_ONLY = ("EX_gal_e", "EX_glcn_e", "EX_glyc_e")

# no source model secretes acetaldehyde, open it would take the flux that carbon dioxide needs
CLOSED = ("EX_acald_e",)

# the variant with every unmeasured carbon by-product closed, carbon dioxide the one carbon exchange left free
CLOSE_CARBON = os.environ.get("METABOLISM_CLOSE_CARBON") == "1"
FREE_CARBON = ("EX_co2_e",)

# the Entner-Doudoroff measurement, EDD and EDA carry the same flux at steady state
MAPPED = {"EDD_EDA": {"EDD": 1.0}}

# off-gas rates are unreliable in the sources or come from their flux fits, so oxygen and CO2 stay latent
NOT_PINNED = {"EX_o2_e", "EX_co2_e"}

# the markings are uniform, only their difference carries the culture
INTERNAL, BOUNDARY = 10.0, 100.0


@dataclass
class Culture:
    dataset: str
    name: str
    carbon: str
    knockouts: set
    pinned: dict
    measurements: list = field(default_factory=list)


@dataclass
class Instance:
    """One culture on its net, with the markings, the measurement matrix and the reaction of every transition."""
    culture: Culture
    net: Net
    m_a: np.ndarray
    m_b: np.ndarray
    W: np.ndarray
    y: np.ndarray
    ids: list
    reaction: list
    sign: np.ndarray
    places: list
    growth: float
    atpm: float
    t_id: np.ndarray = None


@lru_cache(maxsize=2)
def _core(full):
    model = json.loads(CORE.read_text())

    if not full:
        return model

    genome = json.loads(GENOME.read_text())
    reactions = {r["id"]: r for r in genome["reactions"]}
    metabolites = {m["id"]: m for m in genome["metabolites"]}
    names = {m["id"] for m in model["metabolites"]}
    merge = lambda m: m[:-2] + "_e" if m.endswith("_p") else m
    for rid in ADDED + UPTAKE_ONLY:
        r = dict(reactions[rid])
        r["metabolites"] = {merge(m): c for m, c in r["metabolites"].items()}

        if rid in UPTAKE_ONLY:
            r["lower_bound"], r["upper_bound"] = -1000.0, 0.0

        for m in r["metabolites"]:
            if m not in names:
                source = metabolites[m]
                model["metabolites"].append({"id": m, "name": source.get("name", m), "compartment": m[-1],
                                             "formula": source.get("formula", ""), "charge": source.get("charge", 0)})
                names.add(m)

        model["reactions"].append(r)

    # in the variant no exchange of a metabolite with carbon secretes, a measured one is pinned by its culture anyway
    formula = {m["id"]: m.get("formula", "") for m in model["metabolites"]}
    for r in model["reactions"]:
        if r["id"] in CLOSED:
            r["upper_bound"] = 0.0
        elif CLOSE_CARBON and r["id"].startswith("EX_") and r["id"] not in FREE_CARBON:
            (met,) = r["metabolites"]
            if re.search(r"C(?![a-z])", formula[met]):
                r["upper_bound"] = 0.0

    return model


def core(full=True):
    return json.loads(json.dumps(_core(full)))


def expression(text):
    """PFK - FBP as {PFK: 1, FBP: -1}."""
    terms = re.findall(r"([+-]?)\s*([A-Za-z0-9_]+)", text)

    return {name: (-1.0 if s == "-" else 1.0) for s, name in terms}


def blocked(rule, off):
    """Whether a gene rule of b-numbers fails with the genes in off deleted."""
    if not rule.strip():
        return False

    text = re.sub(r"\b(b\d{4}|s\d{4})\b", lambda m: str(m.group(1) not in off), rule)

    return not eval(text.replace("and", " and ").replace("or", " or "))


def cultures(datasets=DATASETS):
    genome = {r["id"]: r.get("gene_reaction_rule", "") for r in json.loads(GENOME.read_text())["reactions"]}
    out = []
    for ds in datasets:
        rows = list(csv.DictReader(open(ROOT / f"{ds}_conditions.csv")))
        pinned, measured = {}, {}
        for r in csv.DictReader(open(ROOT / f"{ds}_exchange.csv")):
            if r["kind"] == "measured" and r["reaction_id"] not in NOT_PINNED and r["rate"] not in ("", "nan"):
                pinned.setdefault(r["condition"], {})[r["reaction_id"]] = float(r["rate"])

        for r in csv.DictReader(open(ROOT / f"{ds}_fluxes.csv")):
            # a flux that the source model fixed by assumption was not measured
            if r["flux"] in ("", "nan") or r.get("excluded_in_source") == "yes":
                continue

            coeff = expression(r["bigg_expression"]) if r["bigg_expression"] else MAPPED.get(r["measurement_id"])
            if coeff:
                measured.setdefault(r["condition"], []).append((r["measurement_id"], coeff, float(r["flux"])))

        for r in rows:
            name = r["condition"]
            if BIOMASS not in pinned.get(name, {}) or not measured.get(name):
                continue

            knocked = {x for x in r["core_reactions_blocked_by_gpr"].split(";") if x}
            genes = {g for g in r.get("knockout_bnumber", "").split(";") if g}
            knocked |= {rid for rid in ADDED if blocked(genome[rid], genes)}
            out.append(Culture(ds, name, r.get("carbon_source", "glucose"), knocked, pinned[name], measured[name]))

    return out


@lru_cache(maxsize=1)
def vocabulary():
    """Every transition any culture net can hold, by reaction and direction, and every boundary place."""
    model = core()
    transitions = []
    for r in model["reactions"]:
        if r["upper_bound"] > 0:
            transitions.append((r["id"], 1.0))

        if r["lower_bound"] < 0:
            transitions.append((r["id"], -1.0))

    transitions.append(("ATPM_drain", 1.0))
    places = [m["id"] for m in model["metabolites"]] + ["biomass", "maintenance"]
    boundary = [p for p in places if p.endswith("_e") or p in ("biomass", "maintenance")]

    return {t: k for k, t in enumerate(transitions)}, places, boundary


def instance(model, culture, growth, atpm, skip=frozenset()):
    """The net of one culture at a growth rate and a maintenance bound, without the transitions in skip, each given by
    its reaction and direction."""
    ids_t, places, _ = vocabulary()
    index = {p: i for i, p in enumerate(places)}
    external = np.array([p.endswith("_e") or p in ("biomass", "maintenance") for p in places])
    pre, pos, reaction, direction, kind = [], [], [], [], []

    def add(stoich, rid, s, k):
        if (rid, s) in skip:
            return

        pre.append({index[m]: -c for m, c in stoich.items() if c < 0})
        pos.append({index[m]: c for m, c in stoich.items() if c > 0})
        reaction.append(rid)
        direction.append(s)
        kind.append(k)

    for r in model["reactions"]:
        rid, stoich = r["id"], dict(r["metabolites"])
        if rid in culture.knockouts or rid in culture.pinned and rid != BIOMASS:
            continue

        if rid == BIOMASS:
            stoich["biomass"] = 1.0

        # a monitor place, ATPM puts a token on it and the pinned marking forces the bound through C sigma = dM
        if rid == "ATPM":
            stoich["maintenance"] = 1.0

        if rid.startswith("EX_"):
            # the medium offers only what the source measured, inorganic species are free in both directions
            if r["upper_bound"] > 0:
                add(stoich, rid, 1.0, 2.0)

            if r["lower_bound"] <= -1000 and rid not in UPTAKE_ONLY:
                add({m: -c for m, c in stoich.items()}, rid, -1.0, 2.0)

            continue

        if r["upper_bound"] > 0:
            add(stoich, rid, 1.0, 1.0 if r["lower_bound"] < 0 else 0.0)

        if r["lower_bound"] < 0:
            add({m: -c for m, c in stoich.items()}, rid, -1.0, -1.0)

    # the drain lets ATPM fire above the bound
    add({"maintenance": -1.0}, "ATPM_drain", 1.0, 0.0)

    pre_p, pre_t, pre_w = zip(*[(p, t, w) for t, d in enumerate(pre) for p, w in d.items()])
    pos_p, pos_t, pos_w = zip(*[(p, t, w) for t, d in enumerate(pos) for p, w in d.items()])
    net = Net(
        len(places), len(reaction),
        np.array(pre_p), np.array(pre_t), np.array(pre_w, float),
        np.array(pos_p), np.array(pos_t), np.array(pos_w, float),
        e=external[:, None].astype(float), a=np.array(kind)[:, None],
    )

    # steady state inside, the measured exchanges, growth and maintenance on the boundary
    m_a = np.where(external, BOUNDARY, INTERNAL)
    m_b = m_a.copy()
    m_b[index["biomass"]] += growth
    m_b[index["maintenance"]] += atpm
    reactions = {x["id"]: x for x in model["reactions"]}
    for rid, rate in culture.pinned.items():
        if rid != BIOMASS:
            (met,) = reactions[rid]["metabolites"]
            m_b[index[met]] += rate

    # a measurement is a signed sum of net fluxes, the net flux of a reaction is forward minus backward
    sign = np.array(direction)
    W = np.array([[coeff.get(rid, 0.0) * s for rid, s in zip(reaction, sign)] for _, coeff, _ in culture.measurements])
    y = np.array([v for _, _, v in culture.measurements])
    ids = [i for i, _, _ in culture.measurements]
    t_id = np.array([ids_t[(rid, s)] for rid, s in zip(reaction, sign)])

    return Instance(culture, net, m_a, m_b, W, y, ids, reaction, sign, places, growth, atpm, t_id)


def feasible(inst):
    """Whether marking B is reachable in the continuous sense, some sigma >= 0 with C sigma = m_B - m_A."""
    res = linprog(np.zeros(inst.net.n_trans), A_eq=inst.net.C, b_eq=inst.m_b - inst.m_a, bounds=(0, None),
                  method="highs")

    return res.status == 0


def dead(inst):
    """The transitions that fire in no run from A to B. Each linear program pushes every transition not yet seen
    firing to fire, z_t <= min(sigma_t, 1), and the ones that stay at zero when none moves are dead."""
    C, dm, n = inst.net.C, inst.m_b - inst.m_a, inst.net.n_trans
    unknown = set(range(n))
    while unknown:
        idx = sorted(unknown)
        m = len(idx)
        c = np.concatenate([np.zeros(n), -np.ones(m)])
        A_ub = np.zeros((m, n + m))
        A_ub[np.arange(m), idx] = -1.0
        A_ub[np.arange(m), n + np.arange(m)] = 1.0
        A_eq = np.hstack([C, np.zeros((C.shape[0], m))])
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(m), A_eq=A_eq, b_eq=dm, bounds=[(0, None)] * n + [(0, 1)] * m,
                      method="highs")
        fired = {t for t, z in zip(idx, res.x[n:]) if z > 1e-9}
        if not fired:
            break

        unknown -= fired

    return {(inst.reaction[t], float(inst.sign[t])) for t in unknown}


def max_growth(inst):
    """The largest growth the net allows with the other boundary places pinned, None if even 0 is out of reach."""
    b = inst.places.index("biomass")
    rows = [p for p in range(inst.net.n_places) if p != b]
    growth = inst.net.C[b]
    res = linprog(-growth, A_eq=inst.net.C[rows], b_eq=(inst.m_b - inst.m_a)[rows], bounds=(0, None), method="highs")

    return float(growth @ res.x) if res.status == 0 else None


def identified(inst, tol=1e-6):
    """Measurements that the two markings fix, those whose row lies in the row space of C."""
    projector = inst.net.C_pinv @ inst.net.C

    return np.abs(inst.W - inst.W @ projector).max(1) < tol * (1 + np.abs(inst.W).max(1))


def instances(datasets=DATASETS):
    """The culture nets, the measured growth kept wherever the net can reach it, and the cultures it cannot hold."""
    model = core()
    out, dropped = [], []
    for c in cultures(datasets):
        growth = c.pinned[BIOMASS]
        inst = None

        for atpm in (ATPM, 0.0):
            trial = instance(model, c, growth, atpm)
            if feasible(trial):
                inst = trial
                break

        # the growth the net cannot reach is lowered to the most it allows, as a reconciliation of the data
        if inst is None:
            best = max_growth(instance(model, c, 0.0, 0.0))
            if best is not None and best > 1e-6:
                inst = instance(model, c, min(growth, best * (1 - 1e-9)), 0.0)

        if inst is None or not feasible(inst):
            dropped.append(inst or instance(model, c, growth, 0.0))
            continue

        out.append(instance(model, c, inst.growth, inst.atpm, frozenset(dead(inst))))

    return out, dropped


def facts(out="results/metabolism/facts.json"):
    """The counts about the data that the paper quotes, the cultures kept and dropped, those whose maintenance bound is
    0 and those whose growth is lowered."""
    insts, dropped = instances()
    kept = {
        "cultures": len(insts),
        "dropped": [f"{i.culture.dataset}/{i.culture.name}" for i in dropped],
        "atpm_zero": sum(i.atpm == 0.0 for i in insts),
        "growth_lowered": sum(i.growth < i.culture.pinned[BIOMASS] * (1 - 1e-6) for i in insts),
        "measured_fluxes_median": float(np.median([len(i.y) for i in insts])),
    }
    Path(out).write_text(json.dumps(kept, indent=1))

    return kept


if __name__ == "__main__":
    print(json.dumps(facts(), indent=1))
