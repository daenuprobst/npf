"""The mechanism nets: STOP needs whole pairs on the electron net, the octet capacities of the arrow net are in pairs,
and untrained games end only in valid molecules."""
from pathlib import Path

import pytest
import torch

from benchmarks.chemistry import mechanism as M
from npf.chem import arrows
from npf.chem.electron_game import ElectronGame

ROOT = Path(__file__).resolve().parents[1]
FLOWER = ROOT / "data/flower/2025/data/flower_dataset/train.txt"


def two_atom_state(bond_electrons):
    """Two atoms and one bond place holding the given number of electrons, charges whole and inside their window."""
    zeros = torch.zeros(1, 2, dtype=torch.long)
    cur = torch.tensor([[[0, bond_electrons], [bond_electrons, 0]]])

    return {"nb": zeros, "q2": zeros, "cur": cur, "mask": torch.ones(1, 2, dtype=torch.bool), "cap": zeros + 4,
            "lo": zeros, "hi": zeros, "fired_a": torch.zeros_like(cur), "fired_b": torch.zeros_like(cur),
            "n_fired": torch.zeros(1, dtype=torch.long)}


@pytest.mark.parametrize("electrons,stop", [(2, True), (4, True), (1, False), (3, False)])
def test_stop_needs_whole_pairs_on_every_bond(electrons, stop):
    game = ElectronGame(d=16, rounds=1, attention=0, pair=8)
    assert bool(game.enabled(two_atom_state(electrons))[1][0]) is stop


def test_without_enabling_stop_is_always_allowed():
    game = ElectronGame(d=16, rounds=1, attention=0, pair=8, enabling=False)
    assert bool(game.enabled(two_atom_state(1))[1][0])


def test_arrow_octet_capacities_are_in_pairs(monkeypatch):
    if not FLOWER.exists():
        pytest.skip(f"{FLOWER} not found")

    monkeypatch.setattr(M, "NET", M.NETS["arrow"])
    lines = FLOWER.read_text().splitlines()[:300]
    octet, _ = arrows.tables(*M._stats(lines))

    # four pairs around C, N and O: bonds and lone pairs of methane, ammonium and water
    assert octet[6] == 4 and octet[7] >= 4 and octet[8] >= 4


@pytest.mark.parametrize("net", ["arrow", "electron"])
def test_untrained_games_end_in_valid_molecules(net, monkeypatch):
    if not (ROOT / M.NETS[net]["data"] / "test").exists():
        pytest.skip(f"prepared {net} data not found")

    from benchmarks.chemistry import mechanism_validity as V

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(M, "NET", M.NETS[net])
    data = M.Steps("test")
    covered = [k for k in range(2000) if data.a["status"][k] == M.OK][:24]
    for forced in (False, True):
        result = V.rollout(data, covered, 0, True, forced, "cpu")
        assert result["valid"] in (1.0, None) and result["octet"] in (1.0, None)
