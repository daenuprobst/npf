"""Copy the old benchmark scripts to refactor/benchmarks and rewrite their imports and module references."""
import ast
import re
from pathlib import Path

OLD = Path("/home/daenu/Code/pgnn/npf")
NEW = Path("/home/daenu/Code/pgnn/refactor/benchmarks")

MOVES = {
    "experiment.py": "synthetic/experiment.py", "sweep.py": "synthetic/sweep.py", "paper_protocol.py": "synthetic/paper_protocol.py",
    "locality.py": "synthetic/locality.py", "thermo.py": "synthetic/equilibrium.py", "sheaf.py": "synthetic/learned_incidence.py",
    "coloured.py": "synthetic/coloured.py",
    "chem_experiment.py": "chemistry/experiment.py", "chem_ensemble.py": "chemistry/ensemble.py", "chem_rxnmapper.py": "chemistry/rxnmapper.py",
    "chem_golden.py": "chemistry/golden.py", "chem_insights.py": "chemistry/insights.py", "chem_invariance.py": "chemistry/invariance.py",
    "chem_review.py": "chemistry/review.py", "chem_figures.py": "chemistry/figures.py",
    "report.py": "report.py", "paper_tables.py": "paper_tables.py", "theory_checks.py": "checks/theory_checks.py", "paper_checks.py": "checks/paper_checks.py",
}
DROP = {  # definitions that now live in the package
    "thermo.py": ["ThermoNPF"], "sheaf.py": ["SheafNPF", "RANK_TOLERANCE"], "coloured.py": ["ColouredNPF", "ColouredPGNN", "transition_features", "mean_input_colour"],
    "locality.py": ["chain"],
}
DATA = {"random_net": "nets", "Net": "nets", "Group": "datasets", "make_pairs": "datasets", "make_flow_pairs": "datasets", "make_flows": "datasets",
        "collate": "batching", "flat_targets": "batching", "gillespie_pairs": "simulate"}
LAYERS = ("apply_incidence", "mlp", "project_state_equation", "scatter_sum", "marking_features")


def drop(text, names):
    lines = text.splitlines(keepends=True)
    kill = set()
    for node in ast.parse(text).body:
        hit = (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names) or \
              (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))
        if hit:
            first = node.lineno - 1
            while first > 0 and lines[first - 1].lstrip().startswith("#"):
                first -= 1
            kill |= set(range(first, node.end_lineno))
    return "".join(l for k, l in enumerate(lines) if k not in kill)


for old, new in MOVES.items():
    s = (OLD / old).read_text()
    if old in DROP:
        s = drop(s, DROP[old])
    for name, module in DATA.items():
        s = re.sub(rf"\bdata\.{name}\b", f"{module}.{name}", s)
    for name in LAYERS:
        s = re.sub(rf"\bmodels\.{name}\b", f"layers.{name}", s)
    s = re.sub(r"\bchem_data\.", "chem.", s)
    s = re.sub(r"\bchem_models\.(?!py)", "chem.", s)
    s = s.replace("from . import chem_data, chem_models", "from npf import chem")
    s = s.replace("from . import data, models", "from npf import batching, datasets, layers, models, nets, simulate")
    s = s.replace("from . import data, locality, models, thermo", "from npf import batching, datasets, layers, models, nets, simulate")
    s = s.replace("from . import data\n", "from npf import batching, datasets, nets\n")
    s = re.sub(r"from \.chem_experiment import", "from .experiment import", s)
    s = re.sub(r"from \.models import ([^\n]+)", r"from npf.layers import \1", s)
    s = s.replace("from .experiment import", "from .experiment import" if new.startswith("synthetic/") else "from ..synthetic.experiment import" if not new.startswith("chemistry/") else "from .experiment import")
    (NEW / new).parent.mkdir(parents=True, exist_ok=True)
    (NEW / new).write_text(s)
for d in ("", "synthetic", "chemistry", "checks"):
    (NEW / d / "__init__.py").write_text("")
print("moved", len(MOVES), "files")
