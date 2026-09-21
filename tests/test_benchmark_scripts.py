"""Static checks of the moved benchmark scripts. No training is run here.

A script that was only moved can still break in three ways that no import of the package notices, a relative import of
a module that was renamed, an attribute that the new package does not export, and a local variable that now has the
name of an imported module (the old module was called data, the new ones are called nets, datasets and so on).
"""
import ast
import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p for p in (ROOT / "benchmarks").rglob("*.py") if p.name != "__init__.py")

# these two run their checks at import time
RUN_ON_IMPORT = {"paper_checks.py", "theory_checks.py"}


def module_name(path):
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def imported_modules(path, tree):
    """Local name to module object for every from-import of npf or benchmarks that binds a module."""
    package = module_name(path).rpartition(".")[0]
    found, problems = {}, []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue

        base = node.module or ""

        if node.level:
            parts = package.split(".")
            base = ".".join(parts[:len(parts) - (node.level - 1)] + ([node.module] if node.module else []))

        if not base.startswith(("npf", "benchmarks")):
            continue

        try:
            module = importlib.import_module(base)
        except Exception as error:
            problems.append(f"line {node.lineno}: cannot import {base} ({type(error).__name__}: {error})")
            continue

        for alias in node.names:
            value = getattr(module, alias.name, None)
            if value is None:
                try:
                    value = importlib.import_module(f"{base}.{alias.name}")
                except Exception:
                    problems.append(f"line {node.lineno}: {base} has no attribute {alias.name}")
                    continue

            if isinstance(value, types.ModuleType):
                found[alias.asname or alias.name] = value

    return found, problems


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_imports_and_module_attributes(path):
    tree = ast.parse(path.read_text())
    modules, problems = imported_modules(path, tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in modules:
            if not hasattr(modules[node.value.id], node.attr):
                problems.append(f"line {node.lineno}: {node.value.id}.{node.attr} does not exist in {modules[node.value.id].__name__}")

    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        stored = {n.id: n.lineno for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in modules}
        for name, line in stored.items():
            uses = sorted({n.lineno for n in ast.walk(function)
                           if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == name
                           and hasattr(modules[name], n.attr)})
            if uses:
                problems.append(f"line {line}: local variable {name} in {function.name} shadows the module {modules[name].__name__}, "
                                f"which the same function uses on lines {uses} (UnboundLocalError)")

    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", [p for p in SCRIPTS if p.name not in RUN_ON_IMPORT], ids=lambda p: str(p.relative_to(ROOT)))
def test_script_imports(path):
    """Import in a fresh interpreter from a neutral directory, the way python -m resolves the modules."""
    code = f"import sys; sys.path[:0] = [{str(ROOT / 'src')!r}, {str(ROOT)!r}]; import matplotlib; matplotlib.use('Agg'); import {module_name(path)}"
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "PATH": "/usr/bin:/bin"}
    done = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd="/", env=env)
    assert done.returncode == 0, done.stderr[-1500:]


def test_no_reference_to_old_module_paths():
    """python -m npf.<script> and the flat module names are gone, so no code, docstring or message may mention them."""
    import re
    flat = "chem_[a-z]+|experiment|sweep|report|paper_[a-z]+|theory_checks|thermo|sheaf|coloured|locality|data"
    old_name = re.compile(rf"\bnpf[./]({flat})\b|\bnpf/models\.py")
    hits = []
    for path in sorted((ROOT / "benchmarks").rglob("*.py")) + sorted((ROOT / "src").rglob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if old_name.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:140]}")

    assert not hits, "\n".join(hits)
