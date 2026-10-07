"""The command line: one reaction, and a file whose lines keep their order and identifiers."""
from npflow import cli


def test_map_one_reaction(capsys):
    assert cli.main(["map", "CC(=O)Cl.CN>>CC(=O)NC"]) == 0
    out = capsys.readouterr().out.strip()

    # the nitrogen and the carbonyl carbon carry map numbers on both sides, the chloride leaves unmapped
    assert out.endswith(">>[CH3:1][NH:4][C:5]([CH3:2])=[O:3]")
    assert "Cl[C:5]" in out


def test_map_file_keeps_order_identifiers_and_failures(tmp_path):
    source, target = tmp_path / "in.smi", tmp_path / "out.smi"
    source.write_text("CC(=O)Cl.CN>>CC(=O)NC\tr1\nnot a reaction\tr2\n>>CCO\tr3\nCCO.CC(=O)O>>CCOC(C)=O\tr4\n")

    assert cli.main(["map", str(source), "-o", str(target), "--processes", "2"]) == 0
    lines = target.read_text().splitlines()

    assert [line.rsplit("\t", 1)[1] for line in lines] == ["r1", "r2", "r3", "r4"]
    assert lines[0].startswith("Cl[C:5]") and lines[3].count(":") > 0
    assert lines[1] == "\tr2" and lines[2] == "\tr3"
