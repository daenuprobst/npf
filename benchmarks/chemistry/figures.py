"""Figures for the paper that only the Petri formulation can draw.

tokengame     A reaction as a token game on the valence net. The marking after every firing with free valence tokens
              as dots on the atoms, the rates of the enabled transitions, the transitions that enabling forbids, and
              the lattice of markings in which count-equivalent firing sequences merge and their probabilities add.
attribution   Exact per-atom contributions to the state-equation readout of the classifier, which are zero beyond
              the receptive field of the reaction centre.
loadbearing   The ablations that show where the net is load-bearing.

    CUDA_VISIBLE_DEVICES="" uv run python -m benchmarks.chemistry.figures tokengame 14440
"""
import io
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D, Point3D

from npf import chem
from .experiment import splits

OUT = Path("paper/figures")
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#d9d8d2"
rgb = lambda h: tuple(int(h[k:k + 2], 16) / 255 for k in (1, 3, 5))
SYMBOL = {5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br", 53: "I"}
DETACH = 1.45


def touched_atoms(r):
    """Atoms of the precursor molecules that take part in a firing (spectators are not drawn)."""
    a = r["a"]
    frag = np.asarray(a["fragment"]) if "fragment" in a else chem.collate([r], "cpu")["frag_a"][0].numpy()[:len(a["x"])]
    keep = np.isin(frag, np.unique(frag[np.unique(r["edits"][:, :2])]))

    return np.nonzero(keep)[0]


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


def layout(r, atoms):
    """2D coordinates in which atoms never move, product atoms sit where the product is drawn, leaving groups stay
    attached to their old neighbour, rotated away from the rest."""
    a, b, target = r["a"], r["b"], r["target"].astype(int)
    product, _ = precursor_mol(b, chem.dense_bonds(b), np.arange(len(b["x"])))
    Chem.SanitizeMol(product)
    AllChem.Compute2DCoords(product)
    pos = {int(target[k]): np.array(product.GetConformer().GetAtomPosition(k))[:2] for k in range(len(target))}

    # leaving groups stay out of rings
    rings = [np.mean([pos[int(target[k])] for k in ring], 0) for ring in product.GetRingInfo().AtomRings()]
    before = chem.dense_bonds(a)
    whole, index = precursor_mol(a, before, atoms)
    Chem.SanitizeMol(whole)
    AllChem.Compute2DCoords(whole)
    old = {int(i): np.array(whole.GetConformer().GetAtomPosition(index[int(i)]))[:2] for i in atoms}
    leaving = [int(i) for i in atoms if int(i) not in pos]
    groups, seen = [], set()

    # connected leaving groups
    for i in leaving:
        if i in seen:
            continue

        group, todo = {i}, [i]
        while todo:
            for k in np.nonzero(before[todo.pop()])[0]:
                if int(k) in leaving and int(k) not in group:
                    group.add(int(k))
                    todo.append(int(k))

        seen |= group
        groups.append(sorted(group))

    for group in groups:
        anchor = [(x, int(c)) for x in group for c in np.nonzero(before[x])[0] if int(c) in pos]

        # a whole molecule leaves, put it to the right
        if not anchor:
            shift = np.array([max(p[0] for p in pos.values()) + 3.0, 0.0]) - np.mean([old[x] for x in group], 0)
            for x in group:
                pos[x] = old[x] + shift

            continue

        x, c = anchor[0]
        others = np.array([p for k, p in pos.items() if k != c] + rings)

        # room for labels like NH2
        margin = np.array([0.9 if (a["h"][k] > 0 and a["element"][k] != 6) else 0.0 for k in pos if k != c] + [1.6] * len(rings))
        rel = np.array([old[y] - old[c] for y in group])

        # smallest distance of the rotated group to everything that is already placed
        def clearance(t):
            new_dir, old_dir = np.array([np.cos(t), np.sin(t)]), (old[x] - old[c]) / np.linalg.norm(old[x] - old[c])
            ang = np.arctan2(new_dir[1], new_dir[0]) - np.arctan2(old_dir[1], old_dir[0])
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            placed = pos[c] + (rel @ rot.T) * DETACH

            return (np.linalg.norm(others[None] - placed[:, None], axis=2) - margin[None]).min()

        best = max(np.linspace(0, 2 * np.pi, 72, endpoint=False), key=clearance)
        new_dir, old_dir = np.array([np.cos(best), np.sin(best)]), (old[x] - old[c]) / np.linalg.norm(old[x] - old[c])
        angle = np.arctan2(new_dir[1], new_dir[0]) - np.arctan2(old_dir[1], old_dir[0])
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for y in group:
            # leaving groups are drawn slightly detached
            pos[y] = pos[c] + R @ (old[y] - old[c]) * DETACH

    return pos


def draw_marking(r, atoms, pos, bonds, next_firing, tokens, size=(520, 400)):
    """PNG of one marking, bonds as in `bonds`, the transition that fires next highlighted, free tokens as dots."""
    a = r["a"]
    mol, index = precursor_mol(a, bonds, atoms)
    i, j, t = next_firing if next_firing is not None else (None, None, None)
    forming = next_firing is not None and bonds[i, j] == 0

    # a bond place without tokens, drawn dotted
    if forming:
        mol.AddBond(index[i], index[j], Chem.BondType.ZERO)

    mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for k, atom in enumerate(atoms):
        conf.SetAtomPosition(k, Point3D(float(pos[int(atom)][0]), float(pos[int(atom)][1]), 0.0))

    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.bondLineWidth, opts.padding, opts.fixedBondLength = 2, 0.08, 34
    opts.clearBackground, opts.highlightBondWidthMultiplier = True, 14
    opts.useBWAtomPalette()
    highlight_bonds, colours = [], {}

    if next_firing is not None:
        bond = mol.GetBondBetweenAtoms(index[i], index[j])
        highlight_bonds, colours = [bond.GetIdx()], {bond.GetIdx(): rgb(BLUE if forming else ORANGE)}

    rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False, addChiralHs=False, wedgeBonds=False)
    drawer.DrawMolecule(mol, highlightAtoms=[], highlightBonds=highlight_bonds, highlightBondColors=colours)
    drawer.SetFillPolys(True)
    drawer.SetColour(rgb(INK))
    centre = np.mean([pos[int(k)] for k in atoms], 0)

    # free valence tokens of the slack places
    for atom, n in tokens.items():
        p = np.array(pos[atom])
        out = (p - centre) / max(np.linalg.norm(p - centre), 1e-6)
        for m in range(int(round(n))):
            q = p + 0.55 * out + 0.22 * (m - (n - 1) / 2) * np.array([-out[1], out[0]])
            drawer.DrawEllipse(Point2D(q[0] - 0.09, q[1] - 0.09), Point2D(q[0] + 0.09, q[1] + 0.09), rawCoords=False)

    drawer.FinishDrawing()

    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def name(a, before, firing):
    i, j, t = firing
    pair = "–".join(sorted((SYMBOL.get(int(a["element"][k]), "X") for k in (i, j)), key=lambda e: (e != "C", e)))

    return f"{'form' if before[i, j] == 0 else ('break' if t == 0 else 'retype')} {pair}"


@torch.no_grad()
def tokengame(reaction_id, weights="results/chem/forward/npf-0.pt"):
    data = chem.load()
    _, _, test = splits(data, "forward")
    r = next(x for x in test if x["id"] == int(reaction_id))
    a, before = r["a"], chem.dense_bonds(r["a"])
    game = chem.TokenGame()
    game.load_state_dict(torch.load(weights, map_location="cpu"))
    game.eval()
    b = chem.collate([r], "cpu")
    n, N = b["ba"].shape[1], chem.N_BOND
    firings = [tuple(int(v) for v in e) for e in r["edits"]]
    firings = [(min(i, j), max(i, j), t) for i, j, t in firings]

    def step(done):
        cur, fired = b["ba"].clone(), torch.zeros_like(b["ba"], dtype=torch.bool)
        for i, j, t in done:
            cur[0, i, j] = cur[0, j, i] = t
            fired[0, i, j] = fired[0, j, i] = True

        logits, enabled, stop = game.rates(b, cur, fired)
        p = torch.softmax(torch.cat([logits.masked_fill(~enabled, -1e4).flatten(1), stop[:, None]], 1), 1)[0]

        return {f: float(p[(f[0] * n + f[1]) * N + f[2]]) for f in firings if f not in done}, {f: bool(enabled[0, f[0], f[1], f[2]]) for f in firings}, float(p[-1])

    # lattice of markings, subsets of the firing vector
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

    # the most probable enabled order for the film strip
    order, s = [], frozenset()
    while s != full:
        k = max((k for k in range(len(firings)) if k not in s and edge[s, s | {k}] is not None), key=lambda k: edge[s, s | {k}])
        order.append(k)
        s = s | {k}

    atoms = touched_atoms(r)
    pos = layout(r, atoms)

    fig = plt.figure(figsize=(7.1, 3.45))
    grid = fig.add_gridspec(3, len(order) + 1, height_ratios=[1.0, 0.2, 1.05], hspace=0.0, wspace=0.04, left=0.005, right=0.995, top=0.945, bottom=0.01)
    bonds, done, slack = before.copy(), [], {}
    for col in range(len(order) + 1):
        ax = fig.add_subplot(grid[0, col])
        ax.axis("off")
        nxt = firings[order[col]] if col < len(order) else None
        ax.imshow(draw_marking(r, atoms, pos, bonds, nxt, slack))
        probs, enabled, p_stop = step(done)

        if nxt is not None:
            text = f"{name(a, before, nxt)}  ({probs[nxt]:.2f})"
            blocked = [name(a, before, f) for f in firings if f not in done and not enabled[f]]
            sub = ("not enabled: " + ", ".join(blocked)) if blocked else "all remaining transitions enabled"
        else:
            text, sub = f"stop  ({p_stop:.2f})", "recorded product reached"

        ax.set_title(f"$m_{col}$", fontsize=8, color=INK, pad=1)
        tx = fig.add_subplot(grid[1, col])
        tx.axis("off")
        tx.text(0.5, 0.95, text, transform=tx.transAxes, ha="center", va="top", fontsize=7.5, color=INK)
        tx.text(0.5, 0.35, sub, transform=tx.transAxes, ha="center", va="top", fontsize=6.5, color=MUTED)

        if nxt is not None:
            i, j, t = nxt
            delta = chem.BOND_ORDER[t] - chem.BOND_ORDER[bonds[i, j]]
            for k in (i, j):
                slack[k] = slack.get(k, 0) - delta

            slack = {k: v for k, v in slack.items() if v > 0}
            bonds[i, j] = bonds[j, i] = t
            done.append(nxt)

    ax = fig.add_subplot(grid[2, :])
    ax.axis("off")
    ax.set_xlim(-1.9, len(firings) + 2.4)
    ax.set_ylim(-1.3, 1.25)
    ypos, letters = {}, "abcdef"
    for size in range(len(firings) + 1):
        level = sorted((s for s in subsets if len(s) == size), key=lambda s: (s not in reach, sorted(s)))
        for m, s in enumerate(level):
            ypos[s] = (size, ((len(level) - 1) / 2 - m) * 0.85)

    for (s, s2), p in edge.items():
        (x1, y1), (x2, y2) = ypos[s], ypos[s2]
        live = p is not None and s in reach
        ax.plot([x1, x2], [y1, y2], color=BLUE if live else GRID, lw=1.6 if live else 0.9, ls="-" if live else (0, (2, 2)), zorder=1,
                solid_capstyle="round")

        if live:
            ax.text(x1 + 0.42 * (x2 - x1), y1 + 0.42 * (y2 - y1), f"{p:.2f}", fontsize=6.5, color=INK, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"), zorder=2)

    labels = [name(a, before, f) for f in firings]

    for s, (x, y) in ypos.items():
        live = s in reach
        ax.scatter([x], [y], s=46, color=BLUE if live else "#ffffff", edgecolor=BLUE if live else "#b9b8b0", linewidth=1.1, zorder=3)
        what = "$m_A$" if not s else "{" + ", ".join(letters[k] for k in sorted(s)) + "}"
        above = y > 0.2
        ax.text(x, y + (0.17 if above else -0.17), what + (f"  $P$ = {reach[s]:.2f}" if live and s else ""), fontsize=6.5,
                color=INK if live else MUTED, ha="center", va="bottom" if above else "top")

    x, y = ypos[full]
    ax.annotate("", xy=(x + 0.5, y), xytext=(x + 0.07, y), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4))
    ax.text(x + 0.27, y + 0.07, f"stop {stop_p[full]:.2f}", fontsize=6.5, color=INK, ha="center", va="bottom")
    orders = [o for o in itertools.permutations(range(len(firings)))]
    allowed = [o for o in orders if all(edge[frozenset(o[:k]), frozenset(o[:k + 1])] is not None for k in range(len(o)))]
    best = max(np.prod([edge[frozenset(o[:k]), frozenset(o[:k + 1])] for k in range(len(o))]) for o in allowed)
    ax.text(x + 0.58, y, f"$P$(product) = {reach[full] * stop_p[full]:.2f}\nbest single sequence {best * stop_p[full]:.2f}\n"
            f"{len(allowed)} of {len(orders)} orders enabled", fontsize=6.8, color=INK, va="center", linespacing=1.3)
    ax.text(-1.85, 1.15, "lattice of markings", fontsize=7.5, color=INK, va="top", fontweight="bold")
    ax.text(-1.85, 0.85, "\n".join(f"{letters[k]} = {labels[k]}" for k in range(len(firings))), fontsize=6.8, color=INK, va="top", linespacing=1.35)
    ax.plot([-1.85, -1.55], [-0.62, -0.62], color=BLUE, lw=1.6)
    ax.text(-1.48, -0.62, "enabled, with probability", fontsize=6.3, color=MUTED, va="center")
    ax.plot([-1.85, -1.55], [-0.88, -0.88], color="#b9b8b0", lw=0.9, ls=(0, (2, 2)))
    ax.text(-1.48, -0.88, "not enabled (no free token)", fontsize=6.3, color=MUTED, va="center")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "tokengame.pdf")
    fig.savefig(OUT / "tokengame.png", dpi=200)
    print("reaction", r["smiles"], "\nfirings", labels, "\nP(product) merged", reach[full] * stop_p[full], "best single", best * stop_p[full], "enabled orders", len(allowed))


@torch.no_grad()
def attribution(reaction_id=None, weights="results/chem/classify/npf-nogate-0.pt"):
    """Exact decomposition of the state-equation readout into per-atom contributions psi_B(i) - psi_A(pi(i))."""
    import json
    data = chem.load()
    _, _, test = splits(data, "classify")
    clf = chem.Classifier(len(data["classes"]), gate=False)
    clf.load_state_dict(torch.load(weights, map_location="cpu"))
    clf.eval()

    def contributions(r):
        b = chem.collate([r], "cpu")
        ha = clf.encoder(b["xa"], b["ba"], b["mask_a"], all_depths=True)
        hb = clf.encoder(b["xb"], b["bb"], b["mask_b"], all_depths=True)
        pa = torch.cat([f(h) for f, h in zip(clf.atom, ha)], -1)[0]
        pb = torch.cat([f(h) for f, h in zip(clf.atom, hb)], -1)[0]
        m = len(r["b"]["x"])

        return (pb[:m] - pa[torch.as_tensor(r["target"].astype(np.int64))]).norm(dim=-1).numpy()

    # a product with a long tail away from the reaction centre, correctly classified
    if reaction_id is None:
        best = None
        for r in test[:4000]:
            if r["target"] is None or not len(r["edits"]) or not (24 <= len(r["b"]["x"]) <= 34):
                continue

            c = contributions(r)
            if clf(chem.collate([r], "cpu")).argmax(-1).item() == r["label"] and (best is None or (c < 1e-6).mean() > best[0]):
                best = ((c < 1e-6).mean(), r)

        r = best[1]
    else:
        r = next(x for x in test if x["id"] == int(reaction_id))

    c = contributions(r)
    b = r["b"]
    product, _ = precursor_mol(b, chem.dense_bonds(b), np.arange(len(b["x"])))
    Chem.SanitizeMol(product)
    AllChem.Compute2DCoords(product)

    # one hue, light -> dark (magnitude)
    ramp = ["#cfe0f7", "#9cc0ee", "#5f9be2", "#2a78d6", "#174f96"]
    scale = np.log10(np.maximum(c, 1e-12))
    lo, hi = np.log10(max(c[c > 1e-6].min(), 1e-3)), np.log10(c.max())
    colours, radii = {}, {}
    for k, v in enumerate(c):
        if v > 1e-6:
            level = int(np.clip((scale[k] - lo) / max(hi - lo, 1e-9) * (len(ramp) - 1) + 0.5, 0, len(ramp) - 1))
            colours[k], radii[k] = rgb(ramp[level]), 0.42

    drawer = rdMolDraw2D.MolDraw2DCairo(900, 520)
    opts = drawer.drawOptions()
    opts.bondLineWidth, opts.padding, opts.clearBackground = 2, 0.06, True
    opts.useBWAtomPalette()
    opts.fillHighlights = True
    drawer.DrawMolecule(product, highlightAtoms=list(colours), highlightAtomColors=colours, highlightAtomRadii=radii, highlightBonds=[])
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText()))

    stats = json.loads(Path("results/chem/insights_attribution.json").read_text())["attribution_norm_by_distance_to_reaction_centre"]
    dist = sorted(int(k) for k in stats)
    fig = plt.figure(figsize=(7.1, 2.35))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.12, left=0.005, right=0.985, top=0.9, bottom=0.2)
    ax = fig.add_subplot(grid[0, 0])
    ax.axis("off")
    ax.imshow(image)
    ax.set_title(f"contribution of each product atom to the class readout: {(c < 1e-6).sum()} of {len(c)} exactly 0", fontsize=7.2, color=INK, pad=2)

    for k, colour in enumerate(ramp):
        ax.add_patch(plt.Rectangle((0.02 + 0.045 * k, -0.06), 0.043, 0.045, transform=ax.transAxes, color=colour, clip_on=False))

    ax.text(0.02, -0.075, f"{10 ** lo:.2f}", transform=ax.transAxes, fontsize=6.3, color=MUTED, va="top", ha="left")
    ax.text(0.02 + 0.045 * len(ramp), -0.075, f"{10 ** hi:.0f}", transform=ax.transAxes, fontsize=6.3, color=MUTED, va="top", ha="right")
    ax.text(0.03 + 0.045 * len(ramp), -0.037, r"$\|\psi_B(i)-\psi_A(\pi(i))\|$, log scale; no colour: exactly zero", transform=ax.transAxes, fontsize=6.5, color=MUTED, va="center")

    ax = fig.add_subplot(grid[0, 1])
    share = [100 * stats[str(d)]["share_exactly_zero"] for d in dist]
    ax.plot(dist, share, color=BLUE, lw=2, marker="o", ms=4.5, mfc=BLUE, mec="white", mew=1.0, clip_on=False, zorder=3)
    ax.axvline(3.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(3.6, 50, "receptive field\n($K=3$ rounds)", fontsize=6.5, color=MUTED, va="center")

    for d, v in zip(dist, share):
        if d in (0, 3, 4, 8):
            ax.text(d - (0.25 if d == 3 else 0), v + 5, f"{v:.1f}", fontsize=6.5, color=INK, ha="right" if d == 3 else "center", va="bottom")

    ax.set_ylim(0, 112)
    ax.set_xlim(-0.4, 8.4)
    ax.set_xticks(dist)
    ax.set_xticklabels([str(d) if d < 8 else "8+" for d in dist])
    ax.set_yticks([0, 50, 100])
    ax.tick_params(labelsize=6.5, colors=MUTED, length=0)
    ax.set_xlabel("bonds from the nearest re-typed bond", fontsize=7, color=INK)
    ax.set_title("atoms with contribution exactly 0 (%)", fontsize=7.2, color=INK, pad=2)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "attribution.pdf")
    fig.savefig(OUT / "attribution.png", dpi=200)
    print("reaction", r["id"], r["smiles"], data["classes"][r["label"]], "zeros", int((c < 1e-6).sum()), "of", len(c))


def loadbearing():
    """Two synthetic load-bearing tests, chain nets (projection) and equilibria of reversible nets (equilibrium layer)."""
    import json
    YELLOW = "#eda100"
    loc = json.loads(Path("results/locality.json").read_text())
    eq = json.loads(Path("results/equilibrium.json").read_text())
    lengths = [k for k in loc["npf"][0] if k.isdigit()]
    fig = plt.figure(figsize=(7.1, 2.05))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.95, left=0.05, right=0.875, top=0.87, bottom=0.2)

    ax = fig.add_subplot(grid[0, 0])
    series = [("gnn", "GNN", AQUA), ("pgnn", "PGNN", ORANGE), ("pgnn+", "PGNN+", YELLOW), ("npf", "NPF, PGNN+SE", BLUE)]
    x = np.arange(len(lengths))
    for key, label, colour in series:
        y = [np.mean([run[k]["rmse"] for run in loc[key]]) for k in lengths]
        ax.plot(x, y, color=colour, lw=2, marker="o", ms=4.5, mec="white", mew=1.0, clip_on=False, zorder=3)

    ends = {"gnn": -0.045, "pgnn": 0.0, "pgnn+": 0.045, "npf": 0.0}
    for key, label, colour in series:
        y = np.mean([run[lengths[-1]]["rmse"] for run in loc[key]])
        ax.text(x[-1] + 0.18, y + ends[key], label, fontsize=6.8, color=INK, va="center")
        ax.plot([x[-1] + 0.08, x[-1] + 0.15], [y, y + ends[key]], color=colour, lw=1.0, clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels(lengths)
    ax.set_ylim(-0.02, 0.5)
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_xlabel("length $L$ of the chain net", fontsize=7, color=INK)
    ax.set_title("RMSE of the inferred firing counts", fontsize=7.5, color=INK, pad=3, loc="left")

    ax2 = fig.add_subplot(grid[0, 1])
    models = [("gnn", "GNN"), ("pgnn", "PGNN"), ("pgnn+", "PGNN+"), ("pgnn+se", "PGNN+SE"), ("npf@16", "NPF token game"), ("npf-thermo", "NPF equilibrium layer")]
    splits_ = [("test", "unseen nets", BLUE, "o"), ("test-large", "larger nets", ORANGE, "s"), ("test-tokens", "3$\\times$ tokens", AQUA, "D")]
    rows = np.arange(len(models))[::-1]
    for (key, label), row in zip(models, rows):
        values = [np.mean([run[sp]["nrmse"] for run in eq[key]]) for sp, *_ in splits_]
        ax2.plot([min(values), max(values)], [row, row], color=GRID, lw=1.2, zorder=1)

        for (sp, name_, colour, marker), v in zip(splits_, values):
            ax2.scatter([v], [row], s=26, color=colour, marker=marker, edgecolor="white", linewidth=0.8, zorder=3, clip_on=False)

    ax2.set_xscale("log")
    ax2.set_xlim(0.003, 1.2)
    ax2.set_yticks(rows)
    ax2.set_yticklabels([m[1] for m in models])
    ax2.set_xlabel("nRMSE of the predicted equilibrium (log scale)", fontsize=7, color=INK)
    ax2.set_title("equilibria of reversible nets", fontsize=7.5, color=INK, pad=3, loc="left")

    for k, (sp, name_, colour, marker) in enumerate(splits_):
        ax2.scatter([1.06], [0.95 - 0.13 * k], s=22, color=colour, marker=marker, edgecolor="white", linewidth=0.8, transform=ax2.transAxes, clip_on=False)
        ax2.text(1.10, 0.95 - 0.13 * k, name_, fontsize=6.8, color=INK, va="center", transform=ax2.transAxes)

    for a_ in (ax, ax2):
        a_.tick_params(labelsize=6.8, colors=MUTED, length=0)

        for side in ("top", "right", "left"):
            a_.spines[side].set_visible(False)

        a_.spines["bottom"].set_color(GRID)
        a_.set_axisbelow(True)

    ax.grid(axis="y", color=GRID, lw=0.6)
    ax2.grid(axis="x", color=GRID, lw=0.6)

    for tick in ax2.get_yticklabels():
        tick.set_color(INK)

    fig.savefig(OUT / "loadbearing.pdf")
    fig.savefig(OUT / "loadbearing.png", dpi=200)


if __name__ == "__main__":
    {"tokengame": tokengame, "attribution": attribution, "loadbearing": loadbearing}[sys.argv[1]](*sys.argv[2:])
