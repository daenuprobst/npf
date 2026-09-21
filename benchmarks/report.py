"""Aggregate results/<task>/<regime>/<model>-<seed>.json into markdown tables (mean +- std over seeds).

    uv run python -m benchmarks.report > results/REPORT.md
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDER = ["se-only", "gnn", "pgnn-eq10", "pgnn", "pgnn@8", "pgnn+", "pgnn+se", "npf-prior", "npf-mlp", "npf", "npf-kl", "npf@16"]
LABEL = {
    "se-only": "state equation only (no learning)", "gnn": "GNN (Eqs. 3-6)", "pgnn-eq10": "PGNN, Eq. 10 literal",
    "pgnn": "**PGNN (paper, Eqs. 9-12)**", "pgnn@8": "PGNN, 8 rounds (1 seed)", "pgnn+": "PGNN+ (strengthened)", "pgnn+se": "PGNN+ with state equation (ablation)",
    "npf-prior": "NPF, rate law + Petri semantics only (ablation)", "npf-mlp": "NPF, generic rate law (ablation)", "npf": "**NPF (ours)**", "npf-kl": "NPF with the I-projection in place of the weighted step", "npf@16": "NPF, 16 time steps",
}
PROTOCOL_LABEL = {
    "se-only": "state equation only (no learning)", "gnn": "GNN (Eqs. 3-6)", "pgnn-linear": "**PGNN, linear special case (paper, Eq. 13)**",
    "pgnn": "**PGNN (paper, Eqs. 9-12)**", "pgnn+": "PGNN+ (strengthened)", "npf-hard": "NPF, state equation forced (ablation)", "npf": "**NPF (ours)**",
}

# (split, metric, header, decimals), nrmse = RMSE / std of the target = sqrt(1 - R^2)
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
              "pgnn-matched": "PGNN-style with the encoder of the token game (equal capacity)",
              "npf-sigma": "**NPF, explicit firing vector from our own mapper**",
              "pgnn-sigma": "generic head on the same maps, one term per seated atom, same width",
              "npf-sigma-exact-netmaps-firing": "**NPF, explicit firing vector from the exact net mapper, no recorded map**",
              "npf-exact-netmaps": "**NPF (Petri semantics), gate taught by the exact net mapper, no recorded map**"}
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
                # exact name, no label / size / ablation variants
                runs[path.stem.rsplit("-", 1)[0]].append(json.loads(path.read_text()))

            if not runs:
                continue

            print(f"\n### {title} / {task}\n")
            print("| model | params | seeds | " + " | ".join(h for _, h in columns) + " |")
            print("|---|---:|---:|" + "---:|" * len(columns))

            for model in ("drfp", "pgnn", "pgnn-matched", "npf-nogate", "npf", "npf-exact-netmaps", "npf-large", "pgnn-sigma", "npf-sigma", "npf-sigma-exact-netmaps-firing"):
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
        violations = lambda pattern: [1 - json.loads(q.read_text())["metrics"]["valence_valid"] for q in sorted((folder / "forward").glob(pattern))]
        matched, small = violations("pgnn-matched-[0-9].json"), violations("pgnn-[0-9].json")
        print(f"* **validity**: {fwd['beam_candidates_that_are_valid_molecules']:.2%} of all top-5 beam candidates are valid molecules; valence violations of the "
              f"one-shot counterpart: {np.mean(matched) if matched else float('nan'):.2%} at equal capacity ({len(matched)} seeds), "
              f"{np.mean(small) if small else float('nan'):.2%} for the smaller published one ({len(small)} seed) (token game: 0 by construction)")
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
            ("one-shot labelling with enabling masks and valence repair", "npf-oneshot-[0-9].json"), ("one-shot labelling, generic", "pgnn-[0-9].json"),
            ("one-shot labelling with enabling masks and valence repair, encoder of the token game", "npf-oneshot-matched-[0-9].json"),
            ("one-shot labelling, generic, encoder of the token game", "pgnn-matched-[0-9].json")]
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
                ("explicit firing vector from our clean mapper", "npf-sigma"), ("generic head on the same maps, same width", "pgnn-sigma"),
                ("explicit firing vector from the exact net mapper, no recorded map", "npf-sigma-exact-netmaps"),
                ("explicit firing vector from the recorded mapping (reference)", "npf-sigma-recorded")]
    columns = [("standard split", ""), ("size extrapolation", "-sizesplit"), ("250 labels", "-labels250"), ("250 labels + firing task", "-firing-labels250"),
               ("1000 labels", "-labels1000"), ("1000 labels + firing task", "-firing-labels1000"), ("all labels + firing task", "-firing")]
    table = {(m, tag): runs("chem/classify", f"{m}{tag}-[0-9].json") for _, m in variants for _, tag in columns}
    if any(v for (m, tag), v in table.items() if tag or m.startswith("npf-sigma")):
        print("\n### classifier variants (accuracy)\n\n| model | " + " | ".join(c for c, _ in columns) + " |\n|---|" + "---:|" * len(columns))

        for label, m in variants:
            print(f"| {label} | " + " | ".join(mean_std([r["metrics"]["accuracy"] for r in table[m, tag]], 4) if table[m, tag] else "–" for _, tag in columns) + " |")


def benchmarks_with_published_protocols(root="results"):
    """USPTO-MIT on the official split with all 40,000 test reactions in the denominator, and the Golden atom mapping
    set. These are the numbers that may be compared with published ones."""
    folder = Path(root) / "uspto_mit"
    forward = [json.loads(path.read_text()) | {"name": path.stem} for path in sorted((folder / "forward").glob("*-[0-9].json"))]
    forward = [r for r in forward if "product_top1_official" in r["metrics"]]
    baselines = [json.loads(path.read_text()) for path in sorted(folder.glob("baselines/*/result.json"))]
    if forward or baselines:
        print("\n### USPTO-MIT forward prediction, official split, mixed setting, share of all 40,000 test reactions\n")
        print("| model | params | training reactions | top-1 greedy | top-1 | top-2 | top-3 | top-5 | valence-valid | training (h) |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        for r in sorted(forward, key=lambda r: r["n_train"]):
            m = r["metrics"]
            beam = [f"{m[k]:.4f}" if k in m else "–" for k in (f"product_top{j}_beam_official" for j in (1, 2, 3, 5))]
            print(f"| NPF token game ({r['name']}) | {r['params']:,} | {r['n_train']:,} | {m['product_top1_official']:.4f} | " + " | ".join(beam)
                  + f" | {m['valence_valid']:.4f} | {r['train_seconds'] / 3600:.1f} |")

        for r in sorted(baselines, key=lambda r: r["subset_lines"]):
            print(f"| Molecular Transformer, trained here ({r['selected']}) | – | {r['subset_lines']:,} | – | {r['product_top1']:.4f} | "
                  f"{r['product_top2']:.4f} | {r['product_top3']:.4f} | {r['product_top5']:.4f} | – | – |")

    golden = [(name, json.loads(path.read_text())) for name, path in (("trained on USPTO-MIT", folder / "golden.json"),
                                                                     ("trained on Schneider 50k", Path(root) / "chem" / "golden.json")) if path.exists()]
    if golden:
        print("\n### Golden atom mapping set, maps with the curated condensed graph of reaction, share of the usable reactions\n")
        print("| mapper | usable reactions | correct |\n|---|---:|---:|")

        for name, g in golden:
            n = g["n_with_complete_curated_mapping"]
            for key, label in (("npf_decoded", f"NPF mapper {name}, as decoded"), ("npf", f"NPF mapper {name}, with token descent"),
                               ("rxnmapper", "RXNMapper, same code"), ("token_audit", f"RXNMapper unless NPF ({name}) moves fewer tokens"),
                               ("either", "either of the two (oracle)")):
                if f"{key}_correct_on_usable" in g:
                    print(f"| {label} | {n:,} | {g[f'{key}_correct_on_usable']:.4f} |")

    exact = [(json.loads(path.read_text()), json.loads(path.with_name(path.name.replace(".report.json", ".json")).read_text()))
             for path in sorted((Path(root) / "chem" / "exact_map").glob("golden-*.report.json"))]
    if exact:
        print("\n### Golden atom mapping set, exact minimum firing vector of the valence net, no learning and no recorded map\n")
        print("The second level of the cost was chosen on 200 reactions, the other 1,560 are held out. Intervals are 95 % bootstrap intervals.\n")
        print("| cost on the net | all usable | interval | held out | proved optimal | correct when proved | median time (s) |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        hydrogen = lambda c: "" if c["ch_places"] and not c["labile_h"] else ", hydrogen" + ("" if c["labile_h"] else " on carbon")
        label = lambda c: "bond places" + (" incl. C-H" if c["ch_places"] else "") + (
            "" if not any(c["secondary"]) else ", then tokens on kept bonds" + hydrogen(c) + " and charge")

        exact = sorted(exact, key=lambda x: x[0]["all"]["exact_net_mapper"])
        for r, c in exact:
            a, h = r["all"], r["held_out"]
            print(f"| {label(c)} | {a['exact_net_mapper']:.4f} | {a['exact_net_mapper_ci95'][0]:.3f} to {a['exact_net_mapper_ci95'][1]:.3f} | "
                  f"{h['exact_net_mapper']:.4f} | {a['proved']:.4f} | {a['exact_net_mapper_when_proved']:.4f} | {c['median_seconds']:.2f} |")

        a, h = exact[-1][0]["all"], exact[-1][0]["held_out"]
        print(f"| RXNMapper, same reactions and scorer | {a['rxnmapper']:.4f} | {a['rxnmapper_ci95'][0]:.3f} to {a['rxnmapper_ci95'][1]:.3f} | "
              f"{h['rxnmapper']:.4f} | – | – | – |")
        print(f"| either of the two (oracle) | {a['either']:.4f} | – | {h['either']:.4f} | – | – | – |")
        best = exact[-1][0]
        if "by_places" in best:
            print("\n### Golden atom mapping set by the number of places that the curated map changes\n")
            print("Many changed places indicate one-pot and multi-step reactions. A curated map that is not minimal is another correspondence of "
                  "the atoms, one that changes more places than the cheapest correspondence does.\n")
            print("| places changed | reactions | exact net mapper | RXNMapper | curated map is not minimal | proved optimal |")
            print("|---|---:|---:|---:|---:|---:|")

            for b in best["by_places"]:
                print(f"| {b['places']} | {b['reactions']:,} | {b['exact_net_mapper']:.4f} | {b['rxnmapper']:.4f} | {b['curated_not_minimal']:.4f} | {b['proved']:.4f} |")

        ties = Path(root) / "chem" / "exact_map" / "ties-golden.json"
        if ties.exists():
            t = json.loads(ties.read_text())
            print("\n### Golden atom mapping set, the mappings that the net cannot tell apart\n")
            print(f"Every optimal mapping of a reaction is enumerated and mappings with the same condensed graph are merged. Enumeration completed for "
                  f"{t['enumerated']:.1%} of the {t['reactions']:,} usable reactions, {t['complete']:.1%} of them below the limit of 512 mappings.\n")
            print("| on the enumerated reactions | share |\n|---|---:|")

            for key, label in (("mappings_per_reaction", "optimal mappings per reaction (mean)"), ("classes_per_reaction", "classes after merging (mean)"),
                               ("true_tie", "more than one class"), ("curated_among_optima", "curated map among the optima (ceiling of a tie-breaker)"),
                               ("uniform_choice", "uniform choice among the classes"), ("first_optimum", "the mapping the solver returns"),
                               ("curated_among_optima_when_tied", "of the tied reactions, curated map among the optima"),
                               ("uniform_choice_when_tied", "of the tied reactions, uniform choice")):
                print(f"| {label} | {t[key]:.4f} |")

        print(f"\nPaired sign test of the best cost against RXNMapper on all usable reactions, {a['only_ours']} reactions only the net maps "
              f"correctly, {a['only_rxnmapper']} only RXNMapper does, p = {a['sign_test_p']:.2f}.")


if __name__ == "__main__":
    main(*sys.argv[1:])
    chemistry(*sys.argv[1:])
    benchmarks_with_published_protocols(*sys.argv[1:])
    load_bearing(*sys.argv[1:])
