"""Aggregate results/<task>/<regime>/<model>-<seed>.json into markdown tables (mean ± std over seeds).

    uv run python -m npf.report > results/REPORT.md
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDER = ["se-only", "gnn", "pgnn-eq10", "pgnn", "pgnn@8", "pgnn+", "pgnn+se", "npf-prior", "npf-mlp", "npf", "npf@16"]
LABEL = {
    "se-only": "state equation only (no learning)", "gnn": "GNN (Eqs. 3-6)", "pgnn-eq10": "PGNN, Eq. 10 literal",
    "pgnn": "**PGNN (paper, Eqs. 9-12)**", "pgnn@8": "PGNN, 8 rounds (1 seed)", "pgnn+": "PGNN+ (strengthened)", "pgnn+se": "PGNN+ with state equation (ablation)",
    "npf-prior": "NPF, rate law + Petri semantics only (ablation)", "npf-mlp": "NPF, generic rate law (ablation)", "npf": "**NPF (ours)**", "npf@16": "NPF, 16 time steps",
}
PROTOCOL_LABEL = {
    "se-only": "state equation only (no learning)", "gnn": "GNN (Eqs. 3-6)", "pgnn-linear": "**PGNN, linear special case (paper, Eq. 13)**",
    "pgnn": "**PGNN (paper, Eqs. 9-12)**", "pgnn+": "PGNN+ (strengthened)", "npf-hard": "NPF, state equation forced (ablation)", "npf": "**NPF (ours)**",
}
# (split, metric, header, decimals); nrmse = RMSE / std of the target = sqrt(1 - R^2)
TABLES = {
    "transitions": [
        ("test", "rmse", "RMSE", 3), ("test", "mae", "MAE", 3), ("test", "rmse_row", "RMSE in im Cᵀ", 3),
        ("test", "rmse_ker", "RMSE in ker C", 3), ("test", "acc_sample", "all counts exact", 3),
        ("test", "consistent", "takes A to B", 3), ("test", "unexplained_tokens", "tokens unexplained", 2),
        ("test-large", "nrmse", "larger nets: nRMSE", 3), ("test-tokens", "nrmse", "3x tokens: nRMSE", 3),
        ("test-gap", "nrmse", "longer gap: nRMSE", 3),
    ],
    "next": [
        ("test", "nrmse@1", "1 step", 3), ("test", "nrmse@8", "8 steps", 3), ("test", "nrmse@40", "40 steps", 3),
        ("test", "drift@40", "conservation drift @40", 4), ("test", "negative_frac", "negative markings", 4),
        ("test-large", "nrmse@40", "larger nets @40", 3), ("test-tokens", "nrmse@40", "3x tokens @40", 3),
        ("test", "firing_corr", "hidden firing vs truth (r)", 3),
    ],
}


def cell(values, decimals):
    values = [v for v in values if v is not None]
    if not values:
        return "–"
    mean, std = np.mean(values), np.std(values)
    if abs(mean) < 10 ** -decimals and mean != 0:
        return f"{mean:.0e}"
    return f"{mean:.{decimals}f}" + (f" ± {std:.{decimals}f}" if len(values) > 1 else "")


def main(root="results"):
    runs = defaultdict(list)
    for path in sorted(q for task in TABLES for q in (Path(root) / task).glob("*/*.json")):
        r = json.loads(path.read_text())
        for m in r["metrics"].values():
            if "r2" in m:
                m["nrmse"] = float(np.sqrt(max(0.0, 1 - m["r2"])))
        runs[r["task"], r["regime"], r["model"]].append(r)
    for task, columns in TABLES.items():
        for regime in sorted({k[1] for k in runs if k[0] == task}):
            print(f"\n### {task} / {regime}\n")
            print("| model | params | " + " | ".join(c[2] for c in columns) + " |")
            print("|---|---:|" + "---:|" * len(columns))
            for model in ORDER:
                rs = runs.get((task, regime, model))
                if rs:
                    cells = [cell([r["metrics"].get(s, {}).get(k) for r in rs], d) for s, k, _, d in columns]
                    print(f"| {LABEL[model]} | {rs[0]['params']:,} | " + " | ".join(cells) + " |")
    protocol = Path(root) / "paper_protocol.json"
    if protocol.exists():
        results = json.loads(protocol.read_text())
        keys = [("rmse", "RMSE"), ("mae", "MAE"), ("rmse_row", "RMSE in im Cᵀ"), ("rmse_ker", "RMSE in ker C"), ("consistent", "takes A to B"),
                ("npf_wins", "NPF better in (paired runs)")]
        for noise, title in (("0.0", "exact states"), ("1.0", "states observed with noise U(-1, 1)")):
            print(f"\n### paper protocol (one fixed net, 70 training samples, 300 epochs) / {title}\n")
            print("| model | params | " + " | ".join(h for _, h in keys) + " |")
            print("|---|---:|" + "---:|" * len(keys))
            for name, label in PROTOCOL_LABEL.items():
                m = results.get(f"{name}|noise={noise}")
                if m:
                    print(f"| {label} | {m['params'][0]:,.0f} | " + " | ".join((f"{m[k][0]:.0%}" if k == "npf_wins" else f"{m[k][0]:.3f} ± {m[k][1]:.3f}") if k in m else "–" for k, _ in keys) + " |")


CHEM_LABEL = {"npf": "**NPF (Petri semantics)**", "npf-large": "**NPF (Petri semantics), width 256**", "npf-nogate": "NPF without participation gate (ablation)",
              "pgnn": "PGNN-style: same message passing, generic readout", "drfp": "DRFP + MLP (reproduced)",
              "npf-sigma": "**NPF, explicit firing vector from our own mapper**"}
CHEM_COLUMNS = {
    "classify": [("accuracy", "accuracy"), ("macro_f1", "macro F1")],
    "map": [("reactions_same_firing_vector", "same firing vector as recorded"), ("atoms_correct", "atoms"),
            ("disagree_ours_moves_fewer_tokens", "disagree, ours moves fewer tokens"), ("disagree_equal", "disagree, tie"),
            ("disagree_ours_moves_more_tokens", "disagree, ours moves more")],
    "forward": [("product_top1_beam", "product top-1"), ("product_top2_beam", "top-2"), ("product_top3_beam", "top-3"),
                ("product_top5_beam", "top-5"), ("product_top1", "top-1 (greedy / one-shot)"), ("valence_valid", "valence-valid")],
}


def chemistry(root="results"):
    for folder, title in (("chem", "Schneider 50k"), ("uspto_mit", "USPTO-MIT")):
        for task, columns in CHEM_COLUMNS.items():
            runs = defaultdict(list)
            for path in sorted((Path(root) / folder / task).glob("*-[0-9].json")):
                runs[path.stem.rsplit("-", 1)[0]].append(json.loads(path.read_text()))  # exact name: no label / size / ablation variants
            if not runs:
                continue
            print(f"\n### {title} / {task}\n")
            print("| model | params | seeds | " + " | ".join(h for _, h in columns) + " |")
            print("|---|---:|---:|" + "---:|" * len(columns))
            for model in ("drfp", "pgnn", "npf-nogate", "npf", "npf-large", "npf-sigma"):
                if model in runs:
                    rs = runs[model]
                    print(f"| {CHEM_LABEL[model]} | {rs[0]['params']:,} | {len(rs)} | " + " | ".join(cell([r["metrics"].get(k) for r in rs], 4) for k, _ in columns) + " |")
            extra = Path(root) / folder / task / ("rxnmapper.json" if task == "map" else "npf-ensemble.json")
            if extra.exists():
                m = json.loads(extra.read_text())
                if task == "map":
                    print("| RXNMapper (pretrained, unsupervised), same reactions and metric | – | – | " + " | ".join(cell([m.get(k)], 4) for k, _ in columns) + " |")
                else:
                    print(f"| **NPF, ensemble of {m['members']}** | – | – | {m['accuracy']:.4f} | – |")
    data_efficiency(root)
    insights(root)


def data_efficiency(root="results"):
    folder = Path(root) / "chem" / "forward"
    rows = {}
    for path in sorted(folder.glob("*-0*.json")):
        r = json.loads(path.read_text())
        if r["model"] in ("npf", "pgnn") and "product_top1" in r["metrics"] and not path.stem.endswith(("keep", "small", "backup")):
            rows[r["model"], r["n_train"]] = r["metrics"]["product_top1"]
    sizes = sorted({n for _, n in rows})
    if len(sizes) > 1:
        print("\n### Schneider 50k / forward: product top-1 (greedy) against the number of training reactions\n")
        print("| model | " + " | ".join(f"{n:,}" for n in sizes) + " |")
        print("|---|" + "---:|" * len(sizes))
        for model in ("pgnn", "npf"):
            print(f"| {CHEM_LABEL[model]} | " + " | ".join(f"{rows[model, n]:.4f}" if (model, n) in rows else "–" for n in sizes) + " |")


def insights(root="results"):
    folder = Path(root) / "chem"
    read = lambda name: json.loads((folder / name).read_text()) if (folder / name).exists() else None
    print("\n### Schneider 50k / what the Petri formulation gives beyond accuracy\n")
    fwd, reach, attr, audit, comb = (read(f"insights_{n}.json") for n in ("forward", "reachability", "attribution", "mapping_audit", "combined_mapping"))
    if fwd:
        one_shot = folder / "forward" / "pgnn-0.json"
        violations = 1 - json.loads(one_shot.read_text())["metrics"]["valence_valid"] if one_shot.exists() else float("nan")
        print(f"* **validity**: {fwd['beam_candidates_that_are_valid_molecules']:.2%} of all top-5 beam candidates are valid molecules; valence violations of the "
              f"one-shot counterpart: {violations:.2%} (token game: 0 by construction)")
        print(f"* **calibration**: expected calibration error {fwd['expected_calibration_error']:.3f} once the probabilities of all traces reaching the same product are added")
        print(f"* **firing order**: in sequences that both break and form bonds, a bond is broken first in {fwd['firing_order'].get('break first', 0)} "
              f"and formed first in {fwd['firing_order'].get('form first', 0)} cases (enabling: a saturated atom has no free valence token)")
        shown = [f"{c}: {', '.join(f'{smi} ({n})' for smi, n in v[:2])}" for c, v in list(fwd["byproducts_by_class"].items())[:12] if v]
        print("* **by-products** (never in the training targets; most frequent leaving fragments per class): " + "; ".join(shown))
    if reach:
        print(f"* **the forward model is an atom mapper**: the most probable firing sequence reaches the recorded product within 5 beams for "
              f"{reach['product_reached_within_beam']:.2%} of the test reactions; the mapping read off it has the recorded firing vector for "
              f"{reach['reactions_same_firing_vector']:.2%} of all reactions and moves more tokens than the record for only {reach['disagree_ours_moves_more_tokens']:.2%}")
    if comb:
        c = comb["summary"]
        print(f"* **two mappers, one net**: token-game mapping with the Sinkhorn mapper as fallback: {c['combined_same_firing_vector']:.2%}; "
              f"both agree with each other on {c['both_mappers_agree']:.2%} of the reactions and are then right in {c['same_given_both_agree']:.2%}")
    if attr:
        rows = attr["attribution_norm_by_distance_to_reaction_centre"]
        print("* **exact attribution**: mean contribution of an atom to the state-equation readout by distance (bonds) from the reaction centre: "
              + ", ".join(f"{d}: {v['mean']:.3f}" for d, v in list(rows.items())[:6]) + f"; exactly zero beyond 3 bonds for {rows['4']['share_exactly_zero']:.1%} of the atoms")
    if audit:
        print(f"* **data audit**: for {audit['recorded_mapping_moves_more_tokens_than_ours']:.2%} of the test reactions the recorded mapping moves more tokens than ours "
              f"(e.g. {audit['examples'][0]['tokens_recorded']:.0f} vs {audit['examples'][0]['tokens_ours']:.0f} for `{audit['examples'][0]['reaction'][:60]}`)")


def mean_std(values, decimals=3):
    return cell(list(values), decimals)


def load_bearing(root="results"):
    """Ablations and experiments that test whether each Petri component carries weight (see THEORY.md)."""
    root = Path(root)
    read = lambda path: json.loads((root / path).read_text()) if (root / path).exists() else None
    runs = lambda folder, pattern: [json.loads(q.read_text()) for q in sorted((root / folder).glob(pattern))]
    print("\n## Is every Petri component load-bearing?\n")

    rows = [("token game (full)", "npf-[0-9].json"), ("token game without enabling (no valence capacities)", "npf-noenabling-[0-9].json"),
            ("one-shot labelling with enabling masks and valence repair", "npf-oneshot-[0-9].json"), ("one-shot labelling, generic", "pgnn-[0-9].json")]
    if any(runs("chem/forward", pat) for _, pat in rows[1:3]):
        print("### forward prediction (Schneider 50k): which part of the token game matters\n\n| model | product top-1 (greedy) | valence-valid |\n|---|---:|---:|")
        for label, pat in rows:
            rs = runs("chem/forward", pat)
            if rs:
                print(f"| {label} | {mean_std([r['metrics']['product_top1'] for r in rs], 4)} | {mean_std([r['metrics']['valence_valid'] for r in rs], 4)} |")

    rows = [("mapper (full)", "npf-[0-9].json"), ("without the Sinkhorn equilibrium (row-softmax)", "npf-noequilibrium-[0-9].json"),
            ("without the kept-bond term (firing cost of bonds)", "npf-nokept-[0-9].json"), ("without the net-morphism check", "npf-nomorphism-[0-9].json"),
            ("without hydrogen / charge token costs", "npf-nocost-[0-9].json"), ("generic readout (PGNN-style)", "pgnn-[0-9].json")]
    if any(runs("chem/map", pat) for _, pat in rows[1:5]):
        print("\n### atom mapping (Schneider 50k): one component removed at a time\n\n| model | same firing vector as recorded | ours moves more tokens than the record |\n|---|---:|---:|")
        for label, pat in rows:
            rs = runs("chem/map", pat)
            if rs:
                print(f"| {label} | {mean_std([r['metrics']['reactions_same_firing_vector'] for r in rs], 4)} | "
                      f"{mean_std([r['metrics']['disagree_ours_moves_more_tokens'] for r in rs], 4)} |")

    loc = read("locality.json")
    if loc:
        lengths = [k for k in loc["npf"][0] if k.isdigit()]
        print("\n### locality lower bound (P3): RMSE on chain nets of length L after training on random graphs\n")
        print("| model | " + " | ".join(f"L = {k}" for k in lengths) + " | random graphs |\n|---|" + "---:|" * (len(lengths) + 1))
        for name, label in (("gnn", "GNN"), ("pgnn", "PGNN (paper)"), ("pgnn+", "PGNN+"), ("pgnn+se", "PGNN+ with state equation"), ("npf", "NPF")):
            if name in loc:
                print(f"| {label} | " + " | ".join(mean_std([r[k]["rmse"] for r in loc[name]]) for k in lengths + ["random graphs (in distribution)"]) + " |")

    eq = read("equilibrium.json")
    if eq:
        print("\n### thermodynamic equilibrium layer (P8): predict the equilibrium marking of a reversible net (nRMSE, 1 = no change; drift of conserved totals)\n")
        print("| model | params | test | larger nets | 3x tokens | conservation drift | negative markings |\n|---|---:|---:|---:|---:|---:|---:|")
        for name, label in (("gnn", "GNN"), ("pgnn", "PGNN (paper)"), ("pgnn+", "PGNN+"), ("pgnn+se", "PGNN+ with state equation"),
                            ("npf@16", "NPF token game, 16 rounds"), ("npf-thermo", "**NPF equilibrium layer**")):
            if name in eq:
                rs = eq[name]
                print(f"| {label} | {rs[0]['params']:,} | " + " | ".join(mean_std([r[k]["nrmse"] for r in rs]) for k in ("test", "test-large", "test-tokens"))
                      + f" | {np.mean([r['test']['conservation_drift'] for r in rs]):.0e} | {np.mean([r['test']['negative'] for r in rs]):.4f} |")

    sh = read("sheaf.json")
    if sh:
        print("\n### learned incidence (conversion ratios hidden; scored against the true net)\n")
        print("| model | RMSE | MAE | RMSE in im Cᵀ | takes A to B | larger nets: RMSE |\n|---|---:|---:|---:|---:|---:|")
        for name, label in (("pgnn", "PGNN (paper)"), ("pgnn+", "PGNN+"), ("npf-unit", "NPF assuming unit ratios"),
                            ("npf-learned-projection-only", "NPF, incidence learned through the projection only (ablation)"),
                            ("npf-learned", "**NPF, learned incidence (state-equation consistency loss)**"), ("npf-oracle", "NPF with the true incidence (oracle)")):
            if name in sh:
                rs = sh[name]
                print(f"| {label} | " + " | ".join(mean_std([r["test"][k] for r in rs]) for k in ("rmse", "mae", "rmse_row", "consistent"))
                      + f" | {mean_std([r['test-large']['rmse'] for r in rs])} |")
        for name, label in (("npf-learned", "with the consistency loss"), ("npf-learned-projection-only", "through the projection only")):
            if name in sh:
                print(f"\nLargest relative error of a learned conversion ratio, {label}: {np.mean([r['ratio_relative_error'] for r in sh[name]]):.2%}; "
                      f"largest violation of R[a,b]·R[b,c] = R[a,c] (no arbitrage): {np.mean([r['arbitrage_gap'] for r in sh[name]]):.2%}.")

    col = read("coloured.json")
    if col:
        print("\n### coloured tokens (masses and colours one step and eight steps ahead)\n")
        keys = ("mass_nrmse@1", "colour_rmse@1", "mass_nrmse@8", "colour_rmse@8", "colour_change_without_inflow")
        print("| model | params | " + " | ".join(keys) + " | larger nets: colour_rmse@8 |\n|---|---:|" + "---:|" * (len(keys) + 1))
        for name, label in (("pgnn+", "PGNN+ on [mass, colour]"), ("npf-free", "NPF masses, free colour update (ablation)"), ("npf-coloured", "**NPF, coloured tokens**")):
            if name in col:
                rs = col[name]
                print(f"| {label} | {rs[0]['params']:,} | " + " | ".join(mean_std([r["test"].get(k, float("nan")) for r in rs], 4) for k in keys)
                      + f" | {mean_std([r['test-large']['colour_rmse@8'] for r in rs], 4)} |")

    inv = read("chem/insights_invariance.json")
    if inv:
        print(f"\n### classifier: counterfactual invariance (P5), {inv['reactions_with_a_remote_site']} reactions, methyl added >= {inv['min_distance_bonds']} bonds from every change\n")
        print("| readout | max logit change | mean logit change | predictions changed |\n|---|---:|---:|---:|")
        for name, label in (("npf-nogate", "state-equation readout (pure)"), ("npf", "state-equation readout + participation gate"), ("pgnn", "generic readout")):
            if name in inv:
                print(f"| {label} | {np.mean(inv[name]['max_logit_change']):.1e} | {np.mean(inv[name]['mean_logit_change']):.1e} | {np.mean(inv[name]['predictions_changed']):.4%} |")

    variants = [("DRFP + MLP", "drfp"), ("generic readout", "pgnn"), ("state-equation readout, no gate", "npf-nogate"), ("state-equation readout + gate", "npf"),
                ("explicit firing vector from our clean mapper", "npf-sigma"), ("explicit firing vector from the recorded mapping (reference)", "npf-sigma-recorded")]
    columns = [("standard split", ""), ("size extrapolation", "-sizesplit"), ("250 labels", "-labels250"), ("250 labels + firing task", "-firing-labels250"),
               ("1000 labels", "-labels1000"), ("1000 labels + firing task", "-firing-labels1000"), ("all labels + firing task", "-firing")]
    table = {(m, tag): runs("chem/classify", f"{m}{tag}-[0-9].json") for _, m in variants for _, tag in columns}
    if any(v for (m, tag), v in table.items() if tag or m.startswith("npf-sigma")):
        print("\n### classifier variants (accuracy)\n\n| model | " + " | ".join(c for c, _ in columns) + " |\n|---|" + "---:|" * len(columns))
        for label, m in variants:
            print(f"| {label} | " + " | ".join(mean_std([r["metrics"]["accuracy"] for r in table[m, tag]], 4) if table[m, tag] else "–" for _, tag in columns) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:])
    chemistry(*sys.argv[1:])
    load_bearing(*sys.argv[1:])
