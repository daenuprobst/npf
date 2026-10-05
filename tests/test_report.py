"""The tables of the paper come out of the committed result files with the numbers the paper quotes."""
from benchmarks import report


def test_paper_benchmarks_reproduce_the_quoted_numbers(capsys):
    report.paper_benchmarks()
    out = capsys.readouterr().out

    for number in (
        "87.71",            # USPTO-480K top-1, targets of the net
        "99.897",           # its RDKit validity
        "70.23 +- 1.59",    # CARE EC level 4 with the maps of the mapper
        "90.20 +- 0.15",    # ECREACT EC level 3
        "91.04 +- 0.74",    # EnzymeMap EC level 3 with the maps of the mapper
        "88.70",            # EnzymeMap mapping, SynRXN validator
        "90.47 +- 0.09",    # FlowER step top-1, electron net
        "94.49 +- 0.36",    # FlowER pathway top-1, electron net
    ):
        assert number in out, number
