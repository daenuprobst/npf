"""Aggregate results/<task>/<regime>/<model>-<seed>.json into markdown tables (mean +- std over seeds).

uv run python -m benchmarks.report > results/REPORT.md
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from npf.chem.cost import CHOSEN

ORDER = [
    "se-only",
    "gnn",
    "pgnn-eq10",
    "pgnn",
    "pgnn+",
    "pgnn+se",
    "npf-prior",
    "npf-mlp",
    "npf",
    "npf-kl",
    "npf@16",
]
LABEL = {
    "se-only": "state equation only (no learning)",
    "gnn": "GNN (PGNN paper, Eqs. 3-6)",
    "pgnn-eq10": "PGNN, Eq. 10 of the PGNN paper, literal",
    "pgnn": "**PGNN (PGNN paper, Eqs. 9-12)**",
    "pgnn+": "PGNN+ (strengthened)",
    "pgnn+se": "PGNN+ with state equation (ablation)",
    "npf-prior": "NPF, rate law + Petri semantics only (ablation)",
    "npf-mlp": "NPF, generic rate law (ablation)",
    "npf": "**NPF (ours)**",
    "npf-kl": "NPF with the I-projection in place of the weighted step",
    "npf@16": "NPF, 16 time steps",
}
PROTOCOL_LABEL = {
    "se-only": "state equation only (no learning)",
    "gnn": "GNN (PGNN paper, Eqs. 3-6)",
    "pgnn-linear": "**PGNN, linear special case (PGNN paper, Eq. 13)**",
    "pgnn": "**PGNN (PGNN paper, Eqs. 9-12)**",
    "pgnn+": "PGNN+ (strengthened)",
    "npf-hard": "NPF, state equation forced (ablation)",
    "npf": "**NPF (ours)**",
}

# captions of the synthetic tables, the task and the kind of net and kinetics
TASK_TITLE = {
    "transitions": "Firing counts between two markings",
    "next": "Forward simulation",
}
REGIME_TITLE = {
    "graph": "directed graphs, stochastic token game",
    "petri": "Petri nets, stochastic token game",
    "petri-ode": "Petri nets, deterministic flow",
    "graph-sat": "directed graphs, saturating kinetics",
    "petri-sat": "Petri nets, saturating kinetics",
    "petri-min": "Petri nets, infinite-server kinetics set by the scarcest input",
}

# (split, metric, header, decimals), nrmse = RMSE / std of the target = sqrt(1 - R^2)
TABLES = {
    "transitions": [
        ("test", "rmse", "RMSE", 3),
        ("test", "mae", "MAE", 3),
        ("test", "rmse_row", "RMSE in im C^T", 3),
        ("test", "rmse_ker", "RMSE in ker C", 3),
        ("test", "acc_sample", "all counts exact", 3),
        ("test", "consistent", "takes A to B", 3),
        ("test", "unexplained_tokens", "tokens unexplained", 2),
        ("test-large", "nrmse", "larger nets: nRMSE", 3),
        ("test-tokens", "nrmse", "3x tokens: nRMSE", 3),
        ("test-gap", "nrmse", "longer gap: nRMSE", 3),
    ],
    "next": [
        ("test", "nrmse@1", "1 step", 3),
        ("test", "nrmse@8", "8 steps", 3),
        ("test", "nrmse@40", "40 steps", 3),
        ("test", "drift@40", "conservation drift @40", 4),
        ("test", "negative_frac", "negative markings", 4),
        ("test-large", "nrmse@40", "larger nets @40", 3),
        ("test-tokens", "nrmse@40", "3x tokens @40", 3),
        ("test", "firing_corr", "hidden firing vs truth (r)", 3),
    ],
}


def cell(values, decimals):
    values = [v for v in values if v is not None]
    if not values:
        return "-"

    # the spread over seeds is the sample standard deviation, the population one understates it with few seeds
    mean, std = np.mean(values), np.std(values, ddof=1) if len(values) > 1 else 0.0
    if abs(mean) < 10**-decimals and mean != 0:
        return f"{mean:.0e}"

    return f"{mean:.{decimals}f}" + (
        f" +- {std:.{decimals}f}" if len(values) > 1 else ""
    )


def main(root="results"):
    runs = defaultdict(list)
    for path in sorted(
        q for task in TABLES for q in (Path(root) / task).glob("*/*.json")
    ):
        r = json.loads(path.read_text())
        for m in r["metrics"].values():
            if "r2" in m:
                m["nrmse"] = float(np.sqrt(max(0.0, 1 - m["r2"])))

        runs[r["task"], r["regime"], r["model"]].append(r)

    for task, columns in TABLES.items():
        for regime in sorted({k[1] for k in runs if k[0] == task}):
            print(f"\n### {TASK_TITLE[task]}, {REGIME_TITLE[regime]}\n")
            print("| model | params | " + " | ".join(c[2] for c in columns) + " |")
            print("|---|---:|" + "---:|" * len(columns))

            for model in ORDER:
                rs = runs.get((task, regime, model))
                if rs:
                    cells = [
                        cell([r["metrics"].get(s, {}).get(k) for r in rs], d)
                        for s, k, _, d in columns
                    ]
                    print(
                        f"| {LABEL[model]} | {rs[0]['params']:,} | "
                        + " | ".join(cells)
                        + " |"
                    )

    protocol = Path(root) / "paper_protocol.json"
    if protocol.exists():
        results = json.loads(protocol.read_text())
        keys = [
            ("rmse", "RMSE"),
            ("mae", "MAE"),
            ("rmse_row", "RMSE in im C^T"),
            ("rmse_ker", "RMSE in ker C"),
            ("consistent", "takes A to B"),
            ("npf_wins", "NPF better in (paired runs)"),
        ]
        for noise, title in (
            ("0.0", "exact states"),
            ("1.0", "states observed with noise U(-1, 1)"),
        ):
            print(
                f"\n### paper protocol (one fixed net, 70 training samples, 300 epochs) / {title}\n"
            )
            print("| model | params | " + " | ".join(h for _, h in keys) + " |")
            print("|---|---:|" + "---:|" * len(keys))

            for name, label in PROTOCOL_LABEL.items():
                m = results.get(f"{name}|noise={noise}")
                if m:
                    print(
                        f"| {label} | {m['params'][0]:,.0f} | "
                        + " | ".join(
                            (
                                (
                                    f"{m[k][0]:.0%}"
                                    if k == "npf_wins"
                                    else f"{m[k][0]:.3f} +- {m[k][1]:.3f}"
                                )
                                if k in m
                                else "-"
                            )
                            for k, _ in keys
                        )
                        + " |"
                    )


CHEM_LABEL = {
    "npf": "**NPF (Petri semantics)**",
    "npf-nogate": "NPF without participation gate (ablation)",
    "pgnn": "generic counterpart: same message passing, generic readout",
    "drfp": "DRFP + MLP (reproduced)",
    "npf-sigma": "**NPF, explicit firing vector from the maps of the exact mapper**",
    "pgnn-sigma": "generic head on the same maps, one term per seated atom, same width",
    "npf-nettargets": "**NPF token game**",
    "npf-noenabling-nettargets": "NPF token game without the enabling mask (ablation)",
    "npf-oneshot-nettargets-matched": "one-shot labelling with enabling masks and valence repair, encoder of the token game",
    "pgnn-nettargets-matched": "one-shot labelling, generic, encoder of the token game (equal capacity)",
}
CHEM_COLUMNS = {
    "classify": [("accuracy", "accuracy"), ("macro_f1", "macro F1")],
    "forward": [
        ("product_top1_beam", "product top-1"),
        ("product_top2_beam", "top-2"),
        ("product_top3_beam", "top-3"),
        ("product_top5_beam", "top-5"),
        ("product_top1", "top-1 (greedy / one-shot)"),
        ("valence_valid", "valence-valid"),
    ],
}
CHEM_MODELS = {
    "classify": ("drfp", "pgnn", "npf-nogate", "npf", "pgnn-sigma", "npf-sigma"),
    "forward": (
        "pgnn-nettargets-matched",
        "npf-oneshot-nettargets-matched",
        "npf-noenabling-nettargets",
        "npf-nettargets",
    ),
}
FORWARD_RUNS = r"(npf-nettargets|npf-noenabling-nettargets|pgnn-nettargets-matched)"


def chemistry(root="results"):
    """Schneider 50k, the internal protocol. Every forward model trains on the targets of the net and every classifier
    reads the maps of the exact mapper."""
    for task, columns in CHEM_COLUMNS.items():
        runs = defaultdict(list)
        for path in sorted((Path(root) / "chem" / task).glob("*-[0-9].json")):
            # exact name, no label / size / ablation variants
            runs[path.stem.rsplit("-", 1)[0]].append(json.loads(path.read_text()))

        if not runs:
            continue

        print(f"\n### Schneider 50k / {task}\n")
        print("| model | params | seeds | " + " | ".join(h for _, h in columns) + " |")
        print("|---|---:|---:|" + "---:|" * len(columns))

        for model in CHEM_MODELS[task]:
            if model in runs:
                rs = runs[model]
                print(
                    f"| {CHEM_LABEL[model]} | {rs[0]['params']:,} | {len(rs)} | "
                    + " | ".join(
                        cell([r["metrics"].get(k) for r in rs], 4) for k, _ in columns
                    )
                    + " |"
                )

        # each ensemble under the name of the model it averages
        for name in ("npf", "npf-sigma"):
            extra = Path(root) / "chem" / task / f"{name}-ensemble.json"
            if task == "classify" and extra.exists():
                m = json.loads(extra.read_text())
                print(
                    f"| {CHEM_LABEL[name]}, ensemble of {m['members']} | - | - | {m['accuracy']:.4f} | - |"
                )

    data_efficiency(root)
    insights(root)


def data_efficiency(root="results"):
    folder = Path(root) / "chem" / "forward"
    rows = defaultdict(list)
    for path in sorted(folder.glob("*-[0-9]*.json")):
        # the file name decides the row, since variants and the matched size carry the same model field
        name = re.fullmatch(FORWARD_RUNS + r"-[0-9](-n\d+)?", path.stem)
        r = json.loads(path.read_text())
        if name and "product_top1" in r["metrics"]:
            rows[name[1], r["n_train"]].append(r["metrics"]["product_top1"])

    sizes = sorted({n for _, n in rows})
    if len(sizes) > 1:
        print(
            "\n### Schneider 50k / forward: product top-1 (greedy) against the number of training reactions\n"
        )
        print("| model | " + " | ".join(f"{n:,}" for n in sizes) + " |")
        print("|---|" + "---:|" * len(sizes))

        for model in (
            "pgnn-nettargets-matched",
            "npf-noenabling-nettargets",
            "npf-nettargets",
        ):
            if any((model, n) in rows for n in sizes):
                print(
                    f"| {CHEM_LABEL[model]} | "
                    + " | ".join(
                        cell(rows[model, n], 4) if (model, n) in rows else "-"
                        for n in sizes
                    )
                    + " |"
                )


def insights(root="results"):
    folder = Path(root) / "chem"
    read = lambda name: (
        json.loads((folder / name).read_text()) if (folder / name).exists() else None
    )
    print("\n### Schneider 50k / what the Petri formulation gives beyond accuracy\n")
    fwd, attr, audit = (
        read(f"insights_{n}.json") for n in ("forward", "attribution", "mapping_audit")
    )
    if fwd:
        violations = lambda pattern: [
            1 - json.loads(q.read_text())["metrics"]["valence_valid"]
            for q in sorted((folder / "forward").glob(pattern))
        ]
        matched = violations("pgnn-nettargets-matched-[0-9].json")
        print(
            f"* **validity**: {fwd['beam_candidates_that_are_valid_molecules']:.2%} of the top-5 beam candidates are valid "
            f"molecules. One-shot counterpart of equal capacity, {np.mean(matched) if matched else float('nan'):.2%} "
            f"valence violations ({len(matched)} seeds). Token game, none by construction"
        )
        print(
            f"* **calibration**: expected calibration error {fwd['expected_calibration_error']:.3f} with the traces of a product merged"
        )
        print(
            f"* **firing order**: in sequences that break and form, a bond breaks first in {fwd['firing_order'].get('break first', 0)} "
            f"and forms first in {fwd['firing_order'].get('form first', 0)} cases"
        )
        shown = [
            f"{c}: {', '.join(f'{smi} ({n})' for smi, n in v[:2])}"
            for c, v in list(fwd["byproducts_by_class"].items())[:12]
            if v
        ]
        print(
            "* **by-products**, never in the targets, most frequent leaving fragments per class: "
            + "; ".join(shown)
        )

    if attr:
        rows = attr["attribution_norm_by_distance_to_reaction_centre"]
        print(
            "* **attribution**: mean contribution of an atom to the readout by bonds from the reaction centre, "
            + ", ".join(f"{d}: {v['mean']:.3f}" for d, v in list(rows.items())[:6])
            + f", exactly zero beyond 3 bonds for {rows['4']['share_exactly_zero']:.1%} of the atoms"
        )

    if audit:
        print(
            f"* **data audit**: for {audit['recorded_mapping_changes_more_places']:.2%} of the test reactions the recorded map "
            f"makes and breaks more bonds than the minimum firing vector ({audit['examples'][0]['places_recorded']} vs "
            f"{audit['examples'][0]['places_exact']} for `{audit['examples'][0]['reaction'][:60]}`)"
        )


def mean_std(values, decimals=3):
    return cell(list(values), decimals)


def load_bearing(root="results"):
    """Ablations and experiments that test whether each Petri component carries weight (see THEORY.md)."""
    root = Path(root)
    read = lambda path: (
        json.loads((root / path).read_text()) if (root / path).exists() else None
    )
    runs = lambda folder, pattern: [
        json.loads(q.read_text()) for q in sorted((root / folder).glob(pattern))
    ]
    print("\n## Is every Petri component load-bearing?\n")

    rows = [
        ("token game (full)", "npf-nettargets-[0-9].json"),
        (
            "token game without enabling (no valence capacities)",
            "npf-noenabling-nettargets-[0-9].json",
        ),
        (
            "one-shot labelling with enabling masks and valence repair, encoder of the token game",
            "npf-oneshot-nettargets-matched-[0-9].json",
        ),
        (
            "one-shot labelling, generic, encoder of the token game",
            "pgnn-nettargets-matched-[0-9].json",
        ),
    ]
    if any(runs("chem/forward", pat) for _, pat in rows):
        print(
            "### forward prediction (Schneider 50k), which part of the token game matters\n\n"
            "| model | product top-1 (greedy) | valence-valid |\n|---|---:|---:|"
        )

        for label, pat in rows:
            rs = runs("chem/forward", pat)
            if rs:
                print(
                    f"| {label} | {mean_std([r['metrics']['product_top1'] for r in rs], 4)} | {mean_std([r['metrics']['valence_valid'] for r in rs], 4)} |"
                )

    loc = read("locality.json")
    if loc:
        lengths = [k for k in loc["npf"][0] if k.isdigit()]
        print(
            "\n### locality lower bound: RMSE on chain nets of length L after training on random graphs\n"
        )
        print(
            "| model | "
            + " | ".join(f"L = {k}" for k in lengths)
            + " | random graphs |\n|---|"
            + "---:|" * (len(lengths) + 1)
        )

        for name, label in (
            ("gnn", "GNN"),
            ("pgnn", "PGNN (paper)"),
            ("pgnn+", "PGNN+"),
            ("pgnn+se", "PGNN+ with state equation"),
            ("npf", "NPF"),
        ):
            if name in loc:
                print(
                    f"| {label} | "
                    + " | ".join(
                        mean_std([r[k]["rmse"] for r in loc[name]])
                        for k in lengths + ["random graphs (in distribution)"]
                    )
                    + " |"
                )

    eq = read("equilibrium.json")
    if eq:
        print(
            "\n### thermodynamic equilibrium layer, equilibrium marking of a reversible net (nRMSE, 1 = no change)\n"
        )
        print(
            "| model | params | test | larger nets | 3x tokens | conservation drift | negative markings |\n|---|---:|---:|---:|---:|---:|---:|"
        )

        for name, label in (
            ("gnn", "GNN"),
            ("pgnn", "PGNN (paper)"),
            ("pgnn+", "PGNN+"),
            ("pgnn+se", "PGNN+ with state equation"),
            ("npf@16", "NPF token game, 16 rounds"),
            ("npf-thermo", "**NPF equilibrium layer**"),
        ):
            if name in eq:
                rs = eq[name]
                print(
                    f"| {label} | {rs[0]['params']:,} | "
                    + " | ".join(
                        mean_std([r[k]["nrmse"] for r in rs])
                        for k in ("test", "test-large", "test-tokens")
                    )
                    + f" | {np.mean([r['test']['conservation_drift'] for r in rs]):.0e} | {np.mean([r['test']['negative'] for r in rs]):.4f} |"
                )

    sh = read("sheaf.json")
    if sh:
        print(
            "\n### learned incidence, conversion ratios hidden, scored against the true net\n"
        )
        print(
            "| model | RMSE | MAE | RMSE in im C^T | takes A to B | larger nets: RMSE |\n|---|---:|---:|---:|---:|---:|"
        )

        for name, label in (
            ("pgnn", "PGNN (paper)"),
            ("pgnn+", "PGNN+"),
            ("npf-unit", "NPF assuming unit ratios"),
            (
                "npf-learned-projection-only",
                "NPF, incidence learned through the projection only (ablation)",
            ),
            (
                "npf-learned",
                "**NPF, learned incidence (state-equation consistency loss)**",
            ),
            ("npf-oracle", "NPF with the true incidence (oracle)"),
        ):
            if name in sh:
                rs = sh[name]
                print(
                    f"| {label} | "
                    + " | ".join(
                        mean_std([r["test"][k] for r in rs])
                        for k in ("rmse", "mae", "rmse_row", "consistent")
                    )
                    + f" | {mean_std([r['test-large']['rmse'] for r in rs])} |"
                )

        for name, label in (
            ("npf-learned", "with the consistency loss"),
            ("npf-learned-projection-only", "through the projection only"),
        ):
            if name in sh:
                print(
                    f"\nLearned conversion ratios, {label}. Largest relative error {np.mean([r['ratio_relative_error'] for r in sh[name]]):.2%}, "
                    f"largest violation of R[a,b]R[b,c] = R[a,c] (no arbitrage) {np.mean([r['arbitrage_gap'] for r in sh[name]]):.2%}, means over seeds."
                )

    col = read("coloured.json")
    if col:
        print(
            "\n### coloured tokens (masses and colours one step and eight steps ahead)\n"
        )
        keys = (
            "mass_nrmse@1",
            "colour_rmse@1",
            "mass_nrmse@8",
            "colour_rmse@8",
            "colour_change_without_inflow",
        )
        print(
            "| model | params | "
            + " | ".join(keys)
            + " | larger nets: colour_rmse@8 |\n|---|---:|"
            + "---:|" * (len(keys) + 1)
        )

        for name, label in (
            ("pgnn+", "PGNN+ on [mass, colour]"),
            ("npf-free", "NPF masses, free colour update (ablation)"),
            ("npf-coloured", "**NPF, coloured tokens**"),
        ):
            if name in col:
                rs = col[name]
                print(
                    f"| {label} | {rs[0]['params']:,} | "
                    + " | ".join(
                        mean_std([r["test"].get(k, float("nan")) for r in rs], 4)
                        for k in keys
                    )
                    + f" | {mean_std([r['test-large']['colour_rmse@8'] for r in rs], 4)} |"
                )

    inv = read("chem/insights_invariance.json")
    if inv:
        print(
            f"\n### classifier, counterfactual invariance of the readout, {inv['reactions_with_a_remote_site']} reactions, "
            f"methyl added >= {inv['min_distance_bonds']} bonds from every change\n"
        )
        print(
            "| readout | max logit change | mean logit change | predictions changed |\n|---|---:|---:|---:|"
        )

        for name, label in (
            ("npf-nogate", "state-equation readout (pure)"),
            ("npf", "state-equation readout + participation gate"),
            ("pgnn", "generic readout"),
        ):
            if name in inv:
                print(
                    f"| {label} | {np.mean(inv[name]['max_logit_change']):.1e} | {np.mean(inv[name]['mean_logit_change']):.1e} | "
                    f"{np.mean(inv[name]['predictions_changed']):.4%} |"
                )

    variants = [
        ("DRFP + MLP", "drfp"),
        ("generic readout", "pgnn"),
        ("state-equation readout, no gate", "npf-nogate"),
        ("state-equation readout + gate", "npf"),
        ("explicit firing vector from the maps of the exact mapper", "npf-sigma"),
        ("generic head on the same maps, same width", "pgnn-sigma"),
    ]
    columns = [
        ("standard split", ""),
        ("size extrapolation", "-sizesplit"),
        ("250 labels", "-labels250"),
        ("250 labels + firing task", "-firing-labels250"),
        ("1000 labels", "-labels1000"),
        ("1000 labels + firing task", "-firing-labels1000"),
    ]
    table = {
        (m, tag): runs("chem/classify", f"{m}{tag}-[0-9].json")
        for _, m in variants
        for _, tag in columns
    }
    if any(v for (m, tag), v in table.items() if tag or m.startswith("npf-sigma")):
        print(
            "\n### classifier variants (accuracy)\n\n| model | "
            + " | ".join(c for c, _ in columns)
            + " |\n|---|"
            + "---:|" * len(columns)
        )

        for label, m in variants:
            print(
                f"| {label} | "
                + " | ".join(
                    (
                        mean_std([r["metrics"]["accuracy"] for r in table[m, tag]], 4)
                        if table[m, tag]
                        else "-"
                    )
                    for _, tag in columns
                )
                + " |"
            )


def uspto_label(name):
    """The row of a USPTO-MIT run from its file name, the model, its width and where its training targets came from."""
    label = "NPF token game" + (", width 256" if "deep" in name else "")

    if "nettargets-single" in name:
        return label + ", one minimum firing vector per reaction"

    return label + (
        ", targets of the net" if "nettargets" in name else ", recorded maps"
    )


def benchmarks_with_published_protocols(root="results"):
    """USPTO-MIT on the official split with all 40,000 test reactions in the denominator, and the Golden atom mapping
    set. These are the numbers that may be compared with published ones."""
    folder = Path(root) / "uspto_mit"

    # the seeds of one run share a row, whose name drops the seed
    forward = defaultdict(list)
    for path in sorted((folder / "forward").glob("*-[0-9].json")):
        r = json.loads(path.read_text())
        if "product_top1_official" in r["metrics"]:
            forward[path.stem.rsplit("-", 1)[0]].append(r)

    baselines = [
        json.loads(path.read_text())
        for path in sorted(folder.glob("baselines/*/result.json"))
    ]
    if forward or baselines:
        print(
            "\n### USPTO-MIT forward prediction, official split, mixed setting, share of all 40,000 test reactions\n"
        )
        print(
            "Token game rule, the recorded major product is among the molecules the firings touch and every "
            "recorded molecule is in the final marking. Major, the largest molecule made is the recorded major "
            "product. Molecular Transformer, exact match of the product side and both rules. Training reactions "
            "in lines of the file for the transformer and usable reactions for the token game.\n"
        )
        print(
            "| model | params | seeds | training reactions | top-1 greedy | top-1 | top-2 | top-3 | top-5 | major top-1 | valence-valid | training (h) |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        keys = (
            ["product_top1_official"]
            + [f"product_top{j}_beam_official" for j in (1, 2, 3, 5)]
            + ["product_major_top1_beam_official", "valence_valid"]
        )

        for name, rs in sorted(
            forward.items(), key=lambda item: (item[1][0]["n_train"], item[0])
        ):
            print(
                f"| {uspto_label(name)} | {rs[0]['params']:,} | {len(rs)} | {rs[0]['n_train']:,} | "
                + " | ".join(cell([r["metrics"].get(k) for r in rs], 4) for k in keys)
                + f" | {np.mean([r['train_seconds'] for r in rs]) / 3600:.1f} |"
            )

        for r in sorted(baselines, key=lambda r: r["subset_lines"]):
            for rule, label in (
                ("product", "exact match"),
                ("product_found", "rule of the token game"),
            ):
                if f"{rule}_top1" in r:
                    print(
                        f"| Molecular Transformer, trained here, {label} | - | 1 | {r['subset_lines']:,} lines | - | "
                        + " | ".join(
                            cell([r.get(f"{rule}_top{j}")], 4) for j in (1, 2, 3, 5)
                        )
                        + f" | {cell([r.get('product_major_top1')], 4) if rule == 'product' else '-'} | - | - |"
                    )

    exact = [
        (
            json.loads(path.read_text()),
            json.loads(
                path.with_name(path.name.replace(".report.json", ".json")).read_text()
            ),
        )
        for path in sorted(
            (Path(root) / "chem" / "exact_map").glob("golden-*.report.json")
        )
    ]
    if exact:
        print(
            "\n### Golden atom mapping set, exact minimum firing vector of the valence net, no learning and no recorded map\n"
        )
        print(
            "Cost chosen on 200 reactions, 1,560 held out. Intervals are 95 % bootstrap intervals.\n"
        )
        print(
            "| cost on the net | all usable | interval | held out | proved optimal | correct when proved | median time (s) |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|")
        hydrogen = lambda c: (
            ""
            if c["ch_places"] and not c["labile_h"]
            else ", hydrogen" + ("" if c["labile_h"] else " on carbon")
        )
        label = (
            lambda c: "bond places"
            + (" incl. C-H" if c["ch_places"] else "")
            + (
                ""
                if not any(c["secondary"])
                else ", then tokens on kept bonds" + hydrogen(c) + " and charge"
            )
            + (", then redox, aromatic and sink" if c.get("third_level") else "")
        )

        exact = sorted(exact, key=lambda x: x[0]["all"]["exact_net_mapper"])
        for r, c in exact:
            a, h = r["all"], r["held_out"]
            print(
                f"| {label(c)} | {a['exact_net_mapper']:.4f} | {a['exact_net_mapper_ci95'][0]:.3f} to {a['exact_net_mapper_ci95'][1]:.3f} | "
                f"{h['exact_net_mapper']:.4f} | {a['proved']:.4f} | {a['exact_net_mapper_when_proved']:.4f} | {c['median_seconds']:.2f} |"
            )

        # the cost of the mapper was chosen on the dev reactions, the comparison with RXNMapper is reported for it, with
        # the third level when it was run
        chosen = (
            sorted(
                [
                    x
                    for x in exact
                    if tuple(x[1]["secondary"]) == tuple(CHOSEN["secondary"])
                    and x[1]["labile_h"] == CHOSEN["labile_h"]
                    and x[1]["ch_places"] == CHOSEN["ch_places"]
                ],
                key=lambda x: not x[1].get("third_level", False),
            )
            or exact[-1:]
        )
        a, h = chosen[0][0]["all"], chosen[0][0]["held_out"]
        print(
            f"| RXNMapper, same reactions and scorer | {a['rxnmapper']:.4f} | {a['rxnmapper_ci95'][0]:.3f} to {a['rxnmapper_ci95'][1]:.3f} | "
            f"{h['rxnmapper']:.4f} | - | - | - |"
        )
        print(
            f"| either of the two (oracle) | {a['either']:.4f} | - | {h['either']:.4f} | - | - | - |"
        )
        best = chosen[0][0]
        if "by_places" in best:
            print(
                "\n### Golden atom mapping set by the number of places that the curated map changes\n"
            )
            print(
                "Many changed places indicate one-pot and multi-step reactions. A curated map that is not minimal "
                "changes more places than the cheapest correspondence.\n"
            )
            print(
                "| places changed | reactions | minimum firing vector | RXNMapper | curated map is not minimal | proved optimal |"
            )
            print("|---|---:|---:|---:|---:|---:|")

            for b in best["by_places"]:
                print(
                    f"| {b['places']} | {b['reactions']:,} | {b['exact_net_mapper']:.4f} | {b['rxnmapper']:.4f} | "
                    f"{b['curated_not_minimal']:.4f} | {b['proved']:.4f} |"
                )

        ties = Path(root) / "chem" / "exact_map" / "ties-golden.json"
        if ties.exists():
            t = json.loads(ties.read_text())
            print(
                "\n### Golden atom mapping set, the mappings that the net cannot tell apart\n"
            )
            print(
                f"All optimal mappings are enumerated and merged by condensed graph. Complete for {t['enumerated']:.1%} "
                f"of the {t['reactions']:,} usable reactions, {t['complete']:.1%} of them below the limit of 512 mappings.\n"
            )
            print("| on the enumerated reactions | share |\n|---|---:|")

            for key, label in (
                ("mappings_per_reaction", "optimal mappings per reaction (mean)"),
                ("classes_per_reaction", "classes after merging (mean)"),
                ("true_tie", "more than one class"),
                (
                    "curated_among_optima",
                    "curated map among the optima (ceiling of a tie-breaker)",
                ),
                ("uniform_choice", "uniform choice among the classes"),
                ("first_optimum", "the first optimum of the search"),
                (
                    "curated_among_optima_when_tied",
                    "of the tied reactions, curated map among the optima",
                ),
                ("uniform_choice_when_tied", "of the tied reactions, uniform choice"),
            ):
                print(f"| {label} | {t[key]:.4f} |")

        print(
            f"\nSign test against RXNMapper on all usable reactions, {a['only_ours']} only the net maps correctly, "
            f"{a['only_rxnmapper']} only RXNMapper, p = {a['sign_test_p']:.2f}."
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
    chemistry(*sys.argv[1:])
    benchmarks_with_published_protocols(*sys.argv[1:])
    load_bearing(*sys.argv[1:])
