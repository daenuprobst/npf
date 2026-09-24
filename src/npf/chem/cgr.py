"""The condensed graph of reaction of a mapping, the precursor graph with every bond labelled by its order before and
after the firing. Two mappings with isomorphic condensed graphs describe the same reaction, which is how optimal
mappings are merged into classes and how a map is scored against a reference.
"""

import networkx as nx
import numpy as np

from .featurisation import dense_bonds


def cgr(reaction, mapping):
    """Condensed graph of reaction under a product -> precursor mapping, over all precursor atoms."""
    a, bb = reaction["a"], dense_bonds(reaction["b"])
    before = dense_bonds(a)
    after = np.zeros_like(before)
    after[np.ix_(mapping, mapping)] = bb
    kept = np.zeros(len(before), bool)
    kept[mapping] = True
    g = nx.Graph()
    for i, el in enumerate(a["element"]):
        g.add_node(i, label=(int(el), bool(kept[i])))

    for i, j in zip(*np.nonzero(np.triu((before > 0) | (after > 0), 1))):
        # bonds between two atoms that both leave are not part of the product, their fate is not recorded
        new = (
            int(after[i, j])
            if (kept[i] and kept[j])
            else (0 if (kept[i] or kept[j]) else int(before[i, j]))
        )
        g.add_edge(int(i), int(j), label=(int(before[i, j]), new))

    return g


def same_cgr(reaction, mapping, reference):
    g1, g2 = cgr(reaction, mapping), cgr(reaction, reference)
    centre = lambda g: sorted(
        d["label"] for _, _, d in g.edges(data=True) if d["label"][0] != d["label"][1]
    )
    if centre(g1) != centre(g2):
        return False

    return nx.is_isomorphic(
        g1,
        g2,
        node_match=lambda x, y: x["label"] == y["label"],
        edge_match=lambda x, y: x["label"] == y["label"],
    )


def key(reaction, mapping):
    """A cheap invariant of the condensed graph, equal for isomorphic graphs."""
    g = cgr(reaction, mapping)
    for _, data in g.nodes(data=True):
        data["text"] = str(data["label"])

    for _, _, data in g.edges(data=True):
        data["text"] = str(data["label"])

    return nx.weisfeiler_lehman_graph_hash(
        g, node_attr="text", edge_attr="text", iterations=3
    )
