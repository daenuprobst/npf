"""The enzymatic data: the holdout of whole EC3 classes that fixes the reaction set of EnzymeMap, and the data sets of
care.py."""
from benchmarks.chemistry import care
from benchmarks.chemistry.enzymes import ec_holdout


def test_ec_holdout_keeps_only_reactions_whose_classes_share_a_split():
    # ten reactions of 1.1.1, one of 2.1.1 and one of 3.1.1, largest class first to train, then val, then test
    records = [{"ec3": "1.1.1", "all_ec3": ["1.1.1"]} for _ in range(10)]
    records += [{"ec3": "2.1.1", "all_ec3": ["2.1.1"]}, {"ec3": "3.1.1", "all_ec3": ["3.1.1"]}]
    # recorded under 1.1.1 and 2.1.1 as well, which went to different splits
    records += [{"ec3": "1.1.1", "all_ec3": ["1.1.1", "2.1.1"]}]

    kept = ec_holdout(records)

    assert len(kept) == 12
    assert all(r["all_ec3"] == [r["ec3"]] for r in kept)


def test_official_test_sets():
    assert care.DATASETS["care_easy"][2] == care.CARE_EASY_TEST == 393
    assert care.DATASETS["ecreact_enzyformer"][2] == 4907
