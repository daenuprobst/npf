"""Figures for the paper that only the Petri formulation can draw.

tokengame     The numbers of the token game figure, drawn in TikZ in paper_v2/figures/tokengame.tex. For one test
              reaction the rates of the enabled transitions after every subset of its firing vector, the orders
              that the enabling rule excludes, and the lattice in which count-equivalent orders merge.
attribution   Panel (a) of the attribution figure, drawn in TikZ in paper_v2/figures/attribution.tex. The product of a
              test reaction shaded by the exact contribution of every atom to the state-equation readout.

    CUDA_VISIBLE_DEVICES="" uv run python -m benchmarks.chemistry.figures tokengame 28317   # results/chem/tokengame.json
    CUDA_VISIBLE_DEVICES="" uv run python -m benchmarks.chemistry.figures attribution 956   # paper_v2/figures/attribution_mol.*
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from npflow import chem

from .experiment import splits

SYMBOL = {5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br", 53: "I"}


def precursor_mol(a, bonds, atoms, slack_change=None):
    """RDKit molecule of a marking restricted to `atoms` (hydrogens as in the precursors, tokens are drawn separately)."""
    mol = Chem.RWMol()
    index = {int(i): k for k, i in enumerate(atoms)}
    for i in atoms:
        atom = Chem.Atom(int(a["element"][i]))
        atom.SetFormalCharge(int(a["q"][i]))
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(int(a["h"][i]))
        mol.AddAtom(atom)

    for i, j in zip(*np.nonzero(np.triu(bonds, 1))):
        if int(i) in index and int(j) in index:
            mol.AddBond(index[int(i)], index[int(j)], chem.RD_BOND[int(bonds[i, j])])

            if bonds[i, j] == 4:
                for k in (i, j):
                    mol.GetAtomWithIdx(index[int(k)]).SetIsAromatic(True)

    return mol, index


def name(a, before, firing):
    i, j, t = firing
    pair = "-".join(
        sorted(
            (SYMBOL.get(int(a["element"][k]), "X") for k in (i, j)),
            key=lambda e: (e != "C", e),
        )
    )

    return (
        f"{'form' if before[i, j] == 0 else ('break' if t == 0 else 'retype')} {pair}"
    )


@torch.no_grad()
def tokengame(reaction_id, weights="results/chem/forward/npf-nettargets-0.pt", out="results/chem/tokengame.json"):
    """The numbers of the token game figure, whose drawing is paper_v2/figures/tokengame.tex. Every subset of the
    firing vector is a marking, and for each the rate law gives the probability of every transition of the vector
    that is enabled there, so the orders that the enabling rule excludes and the count-equivalent orders that merge
    are read off the lattice of markings."""
    data = chem.load()
    _, _, test = splits(data, "forward")
    r = next(x for x in test if x["id"] == int(reaction_id))
    before = chem.dense_bonds(r["a"])
    game = chem.TokenGame()
    game.load_state_dict(torch.load(weights, map_location="cpu"))
    game.eval()
    b = chem.collate([r], "cpu")
    n, N = b["ba"].shape[1], chem.N_BOND
    firings = [(min(int(i), int(j)), max(int(i), int(j)), int(t)) for i, j, t in r["edits"]]

    def step(done):
        cur, fired = b["ba"].clone(), torch.zeros_like(b["ba"], dtype=torch.bool)
        for i, j, t in done:
            cur[0, i, j] = cur[0, j, i] = t
            fired[0, i, j] = fired[0, j, i] = True

        logits, enabled, stop = game.rates(b, cur, fired)
        p = torch.softmax(torch.cat([logits.masked_fill(~enabled, -1e4).flatten(1), stop[:, None]], 1), 1)[0]

        return (
            {f: float(p[(f[0] * n + f[1]) * N + f[2]]) for f in firings if f not in done},
            {f: bool(enabled[0, f[0], f[1], f[2]]) for f in firings},
            float(p[-1]),
        )

    # lattice of markings, subsets of the firing vector, and the probability of reaching each by enabled firings
    subsets = [frozenset(c) for k in range(len(firings) + 1) for c in itertools.combinations(range(len(firings)), k)]
    edge, stop_p, reach = {}, {}, {frozenset(): 1.0}
    for s in sorted(subsets, key=len):
        probs, enabled, stop_p[s] = step([firings[k] for k in s])
        for k in range(len(firings)):
            if k not in s:
                f = firings[k]
                edge[s, s | {k}] = probs[f] if enabled[f] else None

                if enabled[f] and s in reach:
                    reach[s | {k}] = reach.get(s | {k}, 0.0) + reach[s] * probs[f]

    full = frozenset(range(len(firings)))

    def orders(s):
        if s == full:
            yield [], 1.0
            return

        for k in range(len(firings)):
            if k not in s and edge[s, s | {k}] is not None:
                for rest, p in orders(s | {k}):
                    yield [k] + rest, edge[s, s | {k}] * p

    enabled_orders = list(orders(frozenset()))
    best, p_best = max(enabled_orders, key=lambda x: x[1])
    letters = "abcdefgh"
    word = lambda s: "".join(letters[k] for k in sorted(s)) or "0"
    result = {
        "reaction": int(reaction_id),
        "smiles": r["smiles"],
        "transitions": {letters[k]: name(r["a"], before, f) for k, f in enumerate(firings)},
        "edges": {f"{word(s)}-{word(t)}": p for (s, t), p in edge.items()},
        "reached": {word(s): p for s, p in reach.items()},
        "stop": stop_p[full],
        "best_order": [letters[k] for k in best],
        "best": p_best * stop_p[full],
        "merged": reach[full] * stop_p[full],
        "enabled_orders": len(enabled_orders),
        "orders": len(list(itertools.permutations(firings))),
    }
    Path(out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


@torch.no_grad()
def molecule_tikz(product, level, changed, length):
    """TikZ lines of one product, shaded discs for the atoms with a level, bonds of length cm with the second line of a
    double bond inside its ring, the bonds in changed in magenta, heteroatoms labelled with their hydrogens on the side
    with the fewest bonds, and the size of the drawing in cm."""
    Chem.Kekulize(product, clearAromaticFlags=True)
    xy = product.GetConformer().GetPositions()[:, :2]
    bonds = list(product.GetBonds())
    xy = xy / np.mean([np.linalg.norm(xy[bd.GetBeginAtomIdx()] - xy[bd.GetEndAtomIdx()]) for bd in bonds]) * length
    xy = xy - xy.min(0)
    rings = [np.array(ring) for ring in product.GetRingInfo().AtomRings()]
    label, hydrogen = {}, {}
    for atom in product.GetAtoms():
        if atom.GetSymbol() != "C":
            k, h = atom.GetIdx(), atom.GetTotalNumHs()
            label[k] = atom.GetSymbol()
            if h:
                out = [xy[n.GetIdx()] - xy[k] for n in atom.GetNeighbors()]
                sides = [np.array(d) for d in ((1, 0), (-1, 0), (0, -1), (0, 1))]
                side = max(sides, key=lambda d: min((-np.dot(d, v) / np.linalg.norm(v) for v in out), default=1))
                hydrogen[k] = (xy[k] + side * 0.2, "H" if h == 1 else f"H$_{h}$")

    point = lambda v: f"({v[0]:.3f}, {v[1]:.3f})"
    lines = [f"\\fill[attrbase!{20 * (lv + 1)}!white] {point(xy[k])} circle (0.16);" for k, lv in sorted(level.items())]

    # a bond ends short of a labelled atom
    gap, offset = 0.13, 0.065
    for bd in bonds:
        i, j = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        a, e = xy[i].copy(), xy[j].copy()
        u = (e - a) / np.linalg.norm(e - a)
        a = a + u * gap if i in label else a
        e = e - u * gap if j in label else e
        style = "draw=magenta!75!black, line width=1.1pt" if bd.GetIdx() in changed else "draw=black"

        # a changed bond runs between the darkest discs, a white casing keeps it visible on them
        if bd.GetIdx() in changed:
            lines.append(f"\\draw[white, line width=2.6pt] {point(a)} -- {point(e)};")
        lines.append(f"\\draw[{style}] {point(a)} -- {point(e)};")
        if bd.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            n = np.array([-u[1], u[0]])
            ring = next((ring for ring in rings if i in ring and j in ring), None)
            if ring is not None:
                n = n if np.dot(xy[ring].mean(0) - (xy[i] + xy[j]) / 2, n) > 0 else -n
                pairs = [(a + n * offset + u * 0.07, e + n * offset - u * 0.07)]
            elif bd.GetBondType() == Chem.BondType.TRIPLE:
                pairs = [(a + n * offset, e + n * offset), (a - n * offset, e - n * offset)]
            else:
                pairs = [(a + n * offset, e + n * offset)]
            lines += [f"\\draw[{style}] {point(a2)} -- {point(e2)};" for a2, e2 in pairs]

    lines += [f"\\node[font=\\scriptsize, inner sep=0pt] at {point(xy[k])} {{{t}}};" for k, t in sorted(label.items())]
    lines += [f"\\node[font=\\scriptsize, inner sep=0pt] at {point(at)} {{{t}}};" for k, (at, t) in sorted(hydrogen.items())]

    return lines, (float(xy[:, 0].max()), float(xy[:, 1].max()))


def attribution(reaction_ids="956,2284,1091", weights="results/chem/classify/npf-nogate-0.pt",
                figure="paper_v2/figures/attribution.tex", out="paper_v2/figures/attribution_mol", length=0.42):
    """The molecules of the attribution figure of App E. For each test reaction the product, every atom shaded by its
    contribution to the atom terms of the class readout, psi_B(i) - psi_A(pi(i)) along the map of the mapper, in five
    steps of the magenta of the other figures on one log scale shared by all molecules, and the bonds that the firing
    vector changes in magenta. The TikZ goes into the figure file between the lines '% >>> molecule ID' and
    '% <<< molecule ID', and <out>.json holds the facts that paper_v2/tables.py reads."""
    import re

    from .experiment import use_predicted_firing

    data = chem.load()
    _, _, test = splits(data, "classify")
    use_predicted_firing(test, blank_missing=True)
    # the pure readout, w = 1, for which the atom terms decompose exactly, in float64 so that no rounding passes for a
    # contribution
    clf = chem.Classifier(len(data["classes"]), gate=False)
    clf.load_state_dict(torch.load(weights, map_location="cpu"))
    clf.double().eval()

    found = []
    for rid in reaction_ids.split(","):
        r = next(x for x in test if x["id"] == int(rid))
        with torch.no_grad():
            b = {k: v.double() if torch.is_tensor(v) and v.is_floating_point() else v
                 for k, v in chem.collate([r], "cpu").items()}
            ha = clf.encoder(b["xa"], b["ba"], b["mask_a"], all_depths=True)
            hb = clf.encoder(b["xb"], b["bb"], b["mask_b"], all_depths=True)
            pa = torch.cat([f(h) for f, h in zip(clf.atom, ha)], -1)[0]
            pb = torch.cat([f(h) for f, h in zip(clf.atom, hb)], -1)[0]
            seat = r["target"].astype(np.int64)
            c = (pb[: len(seat)] - pa[torch.as_tensor(seat)]).norm(dim=-1).numpy()
            correct = clf(b).argmax(-1).item() == r["label"]

        product, _ = precursor_mol(r["b"], chem.dense_bonds(r["b"]), np.arange(len(r["b"]["x"])))
        Chem.SanitizeMol(product)

        # of the default layout and CoordGen, the one whose closest pair of atoms that share no bond is farthest apart,
        # the default where both leave room, as it spreads chains horizontally
        from rdkit.Chem import rdDepictor

        def spread(mol):
            xy = mol.GetConformer().GetPositions()[:, :2]
            bonded = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()}
            gaps = [np.linalg.norm(xy[i] - xy[j]) for i in range(len(xy)) for j in range(i + 1, len(xy))
                    if (i, j) not in bonded and (j, i) not in bonded]
            return min(gaps) / np.mean([np.linalg.norm(xy[i] - xy[j]) for i, j in bonded])

        layouts = []
        for coordgen in (False, True):
            rdDepictor.SetPreferCoordGen(coordgen)
            rdDepictor.Compute2DCoords(product)
            layouts.append((spread(product), Chem.Mol(product)))
        rdDepictor.SetPreferCoordGen(False)
        product = layouts[0][1] if layouts[0][0] >= 0.75 or layouts[0][0] >= layouts[1][0] else layouts[1][1]

        # the bonds of the product that the firing vector changes, from the precursor numbering of the edits
        inv = {int(i): k for k, i in enumerate(seat)}
        changed = set()
        for e in r["edits"]:
            i, j = inv.get(int(e[0])), inv.get(int(e[1]))
            bond = None if i is None or j is None else product.GetBondBetweenAtoms(i, j)
            if bond is not None:
                changed.add(bond.GetIdx())

        # the distance of every product atom from the nearest re-typed bond, to count the atoms beyond the K = 3 rounds
        from scipy.sparse.csgraph import shortest_path

        touched = sorted({inv[int(x)] for e in r["edits"] for x in e[:2] if int(x) in inv})
        distance = shortest_path(chem.dense_bonds(r["b"]) > 0, unweighted=True)[:, touched].min(1)
        far = distance > 3
        found.append((r, c, correct, product, changed, far, distance))

    # five steps on one log scale over every molecule, step k is attrbase!(20k)!white in TikZ as in the legend of the
    # figure, the base magenta!75!black as it renders, RGB (190, 50, 129)
    base, steps = np.array([190, 50, 129]) / 255, 5
    ramp = [tuple(1 - (1 - base) * (k + 1) / steps) for k in range(steps)]
    every = np.concatenate([c[c > 1e-5] for _, c, _, _, _, _, _ in found])
    lo, hi = np.log10(every.min()), np.log10(every.max())

    text = Path(figure).read_text()
    molecules = []
    for r, c, correct, product, changed, far, distance in found:
        nonzero = c > 1e-5
        level = {int(k): int(np.clip((np.log10(c[k]) - lo) / max(hi - lo, 1e-9) * (steps - 1) + 0.5, 0, steps - 1))
                 for k in np.nonzero(nonzero)[0]}
        lines, (width, height) = molecule_tikz(product, level, changed, length)
        rid = int(r["id"])
        block = [f"% >>> molecule {rid}, written by benchmarks.chemistry.figures attribution, "
                 f"{width:.2f} by {height:.2f} cm"] + lines + [f"% <<< molecule {rid}"]
        pattern = re.compile(rf"^[ \t]*% >>> molecule {rid}\b.*?^[ \t]*% <<< molecule {rid}[^\n]*$", re.S | re.M)
        if not pattern.search(text):
            raise SystemExit(f"{figure} has no block for molecule {rid}")
        indent = re.search(rf"^([ \t]*)% >>> molecule {rid}", text, re.M).group(1)
        text = pattern.sub(lambda _: "\n".join(indent + line for line in block), text)
        molecules.append({
            "id": rid,
            "class": data["classes"][r["label"]],
            "smiles": r["smiles"],
            "correct": bool(correct),
            "atoms": len(c),
            "zero": int((~nonzero).sum()),
            "edits": len(r["edits"]),
            "far_contributing": int((far & nonzero).sum()),
            "far_zero": int((far & ~nonzero).sum()),
            "farthest_contributing": int(distance[nonzero & np.isfinite(distance)].max()),
            "width_cm": width,
            "height_cm": height,
        })

    Path(figure).write_text(text)
    facts = {"molecules": molecules, "lo": float(10**lo), "hi": float(10**hi), "ramp": ramp}
    Path(out).with_suffix(".json").write_text(json.dumps(facts, indent=1))
    print(json.dumps({k: v for k, v in facts.items() if k != "ramp"}))


if __name__ == "__main__":
    {"tokengame": tokengame, "attribution": attribution}[sys.argv[1]](*sys.argv[2:])
