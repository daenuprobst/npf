"""Equivalence tests of the package against the flat code that produced the earlier published numbers.

The flat package is imported under the name npf_old through a symlink in a temporary directory. It is expected under
backup/npf_flat/npf, set NPF_OLD if it lives elsewhere. Tests that need it are skipped when it is missing, and so are
tests that need the data sets or saved weights. Everything runs on the CPU.

    uv run pytest -q

NPF_TEST_TOL=0 asks for bit equality of all model outputs instead of 1e-6. style_audit.py in this directory is a
script and not a test.
"""
import atexit
import importlib
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# the old tree must stay untouched, so no bytecode is written next to it
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))

ROOT = Path(__file__).resolve().parents[1]

# staged under <repo>/refactor before the swap, the repository root afterwards
STAGED = ROOT.name == "refactor"
REPO = Path(os.environ.get("NPF_REPO", ROOT.parent if STAGED else ROOT))
OLD = Path(os.environ.get("NPF_OLD", REPO / "npf" if STAGED else REPO / "backup" / "npf_flat" / "npf"))
DATA = Path(os.environ.get("NPF_DATA", REPO / "data"))
RESULTS = Path(os.environ.get("NPF_RESULTS", REPO / "results"))

for entry in (str(ROOT), str(ROOT / "src")):
    if entry in sys.path:
        sys.path.remove(entry)

    sys.path.insert(0, entry)


def _old_is_flat_package():
    return (OLD / "models.py").exists() and (OLD / "data.py").exists()


def _remove_alias(alias):
    # only the link and its directory are removed, never anything behind the link
    (alias / "npf_old").unlink(missing_ok=True)
    alias.rmdir()


if _old_is_flat_package():
    _alias = Path(tempfile.mkdtemp(prefix="npf-old-"))
    (_alias / "npf_old").symlink_to(OLD, target_is_directory=True)
    sys.path.append(str(_alias))
    atexit.register(_remove_alias, _alias)


def pytest_configure(config):
    # the data loaders of the chemistry script fork worker processes while torch threads exist
    config.addinivalue_line("filterwarnings", "ignore:This process .* is multi-threaded, use of fork:DeprecationWarning")


def old_module(name):
    if not _old_is_flat_package():
        pytest.skip(f"old flat package not found at {OLD}")

    return importlib.import_module(f"npf_old.{name}")


@pytest.fixture(scope="session")
def old():
    """Lazy access to the modules of the old package, old("models") is npf_old.models."""
    return old_module


@pytest.fixture(scope="session")
def schneider_all():
    """The whole data set, about 220 MB. Tests must not modify the reactions in place."""
    path = DATA / "schneider50k.pkl"
    if not path.exists():
        pytest.skip(f"{path} not found")

    import pickle

    return pickle.loads(path.read_bytes())


@pytest.fixture(scope="session")
def schneider(schneider_all):
    # a fixed spread over the file, mapped and unmapped reactions, train and test
    index = np.linspace(0, len(schneider_all["reactions"]) - 1, 360).astype(int)

    return {"reactions": [schneider_all["reactions"][i] for i in index], "classes": schneider_all["classes"]}


def assert_same_arrays(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} != {b.dtype}"
    assert np.array_equal(a, b, equal_nan=True), f"{what}: values differ, max abs diff {np.abs(a.astype(float) - b.astype(float)).max()}"


def assert_same_tensors(a, b, what="", tol=0.0):
    assert a.shape == b.shape, f"{what}: shape {tuple(a.shape)} != {tuple(b.shape)}"
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} != {b.dtype}"

    if tol == 0.0 or not a.dtype.is_floating_point:
        assert torch.equal(a, b), f"{what}: values differ" + (
            f", max abs diff {(a.double() - b.double()).abs().max().item():.3e}" if a.dtype.is_floating_point else "")
    else:
        diff = (a.double() - b.double()).abs().max().item() if a.numel() else 0.0
        assert diff <= tol, f"{what}: max abs diff {diff:.3e} > {tol}"


def assert_same_state(old_model, new_model):
    so, sn = old_model.state_dict(), new_model.state_dict()
    assert list(so) == list(sn), f"state_dict keys differ\n old only: {[k for k in so if k not in sn]}\n new only: {[k for k in sn if k not in so]}"

    for k in so:
        assert_same_tensors(so[k], sn[k], f"parameter {k}")
