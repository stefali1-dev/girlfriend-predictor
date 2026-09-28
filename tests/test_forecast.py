"""build_rows and split_time: pairing single interviews with follow-ups, and the split."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from forecast import TEST_ROUND, build_rows, split_time  # noqa: E402


def frame(rows):
    """rows: (person, round, year, age, partnered) -> minimal clean-schema frame."""
    return pd.DataFrame(rows, columns=["person_id", "round", "year", "age",
                                       "partnered", "asvab_percentile"])


def one_row(got, person, rounded):
    return got[(got.person_id == person) & (got["round"] == rounded)].iloc[0]


def test_prefers_the_interview_2_years_later():
    got = build_rows(frame([
        ("A", 2, 2000, 24, 0, 50),
        ("A", 3, 2001, 25, 1, 50),   # 1 year later: too close
        ("A", 4, 2002, 26, 0, 50),   # 2 years later: the pick
        ("A", 5, 2003, 27, 1, 50),   # 3 years later: only a fallback
    ]))
    row = one_row(got, "A", 2)
    assert row.follow_round == 4 and row.found_partner == 0


def test_falls_back_to_3_years_and_drops_other_gaps():
    got = build_rows(frame([
        ("B", 2, 2000, 30, 0, 50),
        ("B", 4, 2003, 33, 1, 50),   # 3-year gap: still "about 2 years"
        ("C", 2, 2000, 30, 0, 50),
        ("C", 6, 2005, 35, 1, 50),   # 5-year gap: no follow-up, row dropped
        ("D", 2, 2000, 30, 0, 50),
        ("D", 3, 2001, 31, 1, 50),   # only 1 year later: no valid follow-up either
    ]))
    assert set(got.person_id) == {"B"}
    assert one_row(got, "B", 2).found_partner == 1


def test_year_tie_goes_to_the_earlier_round():
    got = build_rows(frame([
        ("E", 2, 2000, 28, 0, 50),
        ("E", 4, 2002, 30, 1, 50),   # two interviews in 2002 (fieldwork spill)
        ("E", 5, 2002, 30, 0, 50),
    ]))
    row = one_row(got, "E", 2)
    assert row.follow_round == 4 and row.found_partner == 1


def test_only_single_adults_are_base_rows():
    got = build_rows(frame([
        ("F", 1, 1998, 18, 0, 55),   # round 1: asvab is blanked (tested the next school year)
        ("F", 3, 2000, 20, 1, 55),
        ("G", 5, 2001, 17, 0, 50),   # under 18 at the single interview
        ("G", 7, 2003, 19, 1, 50),
        ("H", 5, 2001, 42, 0, 50),   # over 41
        ("H", 7, 2003, 44, 1, 50),
        ("I", 5, 2001, 30, 1, 50),   # already partnered: not a base row
        ("I", 7, 2003, 32, 1, 50),
    ]))
    assert set(got.person_id) == {"F"}
    row = one_row(got, "F", 1)
    assert pd.isna(row.asvab_percentile) and row.found_partner == 1


def test_split_time_keeps_rounds_and_people_disjoint():
    base = frame([
        ("P1", 2, 2000, 24, 0, 50), ("P1", 18, 2017, 40, 0, 50),
        ("P2", 5, 2001, 24, 0, 50), ("P2", 19, 2019, 41, 0, 50),
        ("P3", 2, 2000, 24, 0, 50), ("P3", 10, 2006, 30, 0, 50),
        ("P4", 18, 2017, 35, 0, 50), ("P4", 20, 2021, 39, 0, 50),
    ])
    train, test, unused = split_time(base)
    assert (train["round"] < TEST_ROUND).all() and (test["round"] >= TEST_ROUND).all()
    assert not set(train.person_id) & set(test.person_id)
    # every row lands in exactly one of train, test or an unused bucket
    assert len(train) + len(test) + sum(len(v) for v in unused.values()) == len(base)
