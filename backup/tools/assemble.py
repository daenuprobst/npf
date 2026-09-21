"""Assemble new modules from named top-level definitions of the old flat package, verbatim (no retyping)."""
import ast
import sys
from pathlib import Path

OLD = Path("/home/daenu/Code/pgnn/npf")
NEW = Path("/home/daenu/Code/pgnn/refactor/src/npf")


def segments(path):
    """name -> source text of every top-level def / class / assignment, with the comment lines directly above it."""
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    out = {}
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        elif isinstance(node, ast.Assign):
            names = []
            for t in node.targets:
                names += [t.id] if isinstance(t, ast.Name) else [e.id for e in getattr(t, "elts", []) if isinstance(e, ast.Name)]
            start = node.lineno
        else:
            continue
        first = start - 1
        while first > 0 and lines[first - 1].lstrip().startswith("#"):
            first -= 1
        src = "".join(lines[first:node.end_lineno])
        for n in names:
            out[n] = src
    return out


def build(target, header, parts):
    chunks = [header.rstrip() + "\n"]
    cache = {}
    previous_kind = None
    for source, names in parts:
        segs = cache.setdefault(source, segments(OLD / source))
        for n in names:
            src = segs[n]
            kind = "def" if src.lstrip().startswith(("def ", "class ", "@")) else "assign"
            gap = "\n\n" if kind == "def" or previous_kind == "def" else ""
            chunks.append(gap + src)
            previous_kind = kind
    (NEW / target).parent.mkdir(parents=True, exist_ok=True)
    (NEW / target).write_text("".join(chunks))
    print("wrote", target)


if __name__ == "__main__":
    build("chem/featurise.py", '''"""Reactions as pairs of attributed graphs. State A is the precursor graph, state B the product graph."""
import csv
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
''', [("chem_data.py", ["ELEMENTS", "BOND_TYPE", "BOND_ORDER", "EXTRA_CAPACITY", "HYPERVALENT", "N_ATOM_FEAT", "MAX_PRECURSOR_ATOMS",
                        "RD_BOND", "one_hot", "atom_features", "skeleton_classes", "graph", "dense_bonds", "featurise", "build",
                        "_featurise_mit", "_without_maps", "build_uspto_mit", "load", "collate"])])
    build("chem/decode.py", '''"""From a marking of the valence net back to molecules."""
import numpy as np
from rdkit import Chem

from .featurise import BOND_ORDER, HYPERVALENT, RD_BOND, dense_bonds
''', [("chem_data.py", ["RECONSTRUCTION", "NORMAL_VALENCE", "ANION_PRIORITY", "NEUTRALISE_MIN_ATOMS", "canonical_product",
                        "marking_to_products", "marking_fragments", "_mol", "mapping_from_marking"])])
    build("chem/encoder.py", '''"""Message passing on the valence net of a molecule."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .featurise import N_ATOM_FEAT
''', [("chem_models.py", ["N_BOND", "mlp", "PetriLayer", "Encoder"])])
    build("chem/classifier.py", '''"""Reaction classification from the firing vector."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import N_BOND, Encoder, mlp
''', [("chem_models.py", ["Classifier"])])
    build("chem/mapper.py", '''"""Atom mapping as the equilibrium of an assignment net."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .encoder import N_BOND, Encoder, PetriLayer, mlp
''', [("chem_models.py", ["sinkhorn", "Mapper"])])
    build("chem/token_game.py", '''"""Forward reaction prediction as a firing sequence of the valence net."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import N_BOND, Encoder, mlp
from .featurise import BOND_ORDER
''', [("chem_models.py", ["TokenGame"])])
    build("chem/one_shot.py", '''"""One shot counterpart of the token game. Every bond place is labelled independently."""
import numpy as np
import torch
import torch.nn as nn

from .encoder import N_BOND, Encoder, mlp
from .featurise import BOND_ORDER
''', [("chem_models.py", ["Forward"])])
