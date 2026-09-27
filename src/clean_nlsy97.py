"""Clean NLSY97: wide raw extract -> long table, one row per person per interview round.

Rows: interviewed rounds (CV_MARSTAT 1-10) at age 18+.
A feature value at round t only uses information known at or before t
(carry forward, never backward), so the table can also serve the
"partnered 2 years later" question later. Carried-forward questions
(religion, height, ...) are filled from every round, including the
rounds before age 18: those answers were known, even though the rows
are not kept.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

RAW = Path("data/raw/nlsy97")
OUT = Path("data/clean/nlsy97.parquet")

# NLSY97 interviews were annual 1997-2011, biennial after.
ROUNDS = list(range(1997, 2012)) + [2013, 2015, 2017, 2019, 2021, 2023]
ROUND_NUM = {year: i + 1 for i, year in enumerate(ROUNDS)}
MISSING = [-1, -2, -3, -4, -5]  # refused, don't know, invalid skip, valid skip, non-interview

# Midpoints of the show-card brackets, in dollars (asked when the exact amount
# is refused/unknown). Top bracket is open-ended; 300000 is an assumption.
WAGE_BRACKETS = {1: 2500, 2: 7500, 3: 17500, 4: 37500, 5: 75000, 6: 175000, 7: 300000}
BIZ_BRACKETS = {2: 2500, 3: 7500, 4: 17500, 5: 37500, 6: 75000, 7: 175000, 8: 300000}

IN_TO_CM = 2.54
LB_TO_KG = 0.45359237

REGION = {1: "northeast", 2: "midwest", 3: "south", 4: "west"}
MSA = {1: "not_in_cbsa", 2: "cbsa_not_central_city", 3: "cbsa_central_city", 4: "cbsa_not_known"}
EDU_FROM_ENROLL = {
    1: "less_than_hs", 2: "hs", 3: "hs", 4: "some_college", 5: "some_college",
    6: "bachelor_plus", 7: "bachelor_plus", 8: "less_than_hs",
    9: "some_college", 10: "some_college", 11: "bachelor_plus",
}
EDU_FROM_DEGREE = {0: "less_than_hs", 1: "hs", 2: "hs", 3: "some_college",
                   4: "bachelor_plus", 5: "bachelor_plus", 6: "bachelor_plus", 7: "bachelor_plus"}
ENROLLED = {8: "high_school", 9: "college_2yr", 10: "college_4yr", 11: "graduate"}


def religion_map(coding):
    """Flat code -> category dict from {category: [codes]}."""
    return {code: cat for cat, codes in coding.items() for code in codes}


# Current-religion codings differ by era; collapsed to a stable small set.
# Mormon, Unitarian, Quaker, Jehovah's Witness and Seventh Day Adventist are
# "other" in every era (each era codes them differently, sometimes not at all).
RELIGION_1997 = religion_map({"catholic": [1], "protestant": list(range(2, 14)),
                              "jewish": list(range(14, 18)), "none": [25, 26, 27]})
RELIGION_2005 = religion_map({"catholic": [1], "protestant": list(range(2, 14)),
                              "jewish": list(range(14, 18)), "none": [25, 26, 27]})
RELIGION_2017 = religion_map({"catholic": [101],
                              "protestant": [103, 104, 105, 106, 107, 111, 112],
                              "jewish": [102], "none": [116]})
# every other valid code in each era (other religions, other-specify) -> "other"


def load_columns():
    """CSV columns are reference numbers; the Stata renames map them to
    <question>_<year> (e.g. R1210200 -> CV_MARSTAT_1997)."""
    text = (RAW / "partner-model-value-labels.do").read_text()
    pairs = re.findall(r"rename (\w+) (\S+)", text)
    renames = {rnum: name for rnum, name in pairs if re.fullmatch(r"[A-Z]\d+", rnum)}
    wide = pd.read_csv(RAW / "partner-model.csv").set_index("R0000100", drop=False)  # PUBID
    return {name: wide[rnum] for rnum, name in renames.items() if rnum in wide.columns}


def num(cols, name):
    """Column as float with the NLSY missing codes blanked."""
    s = cols[name]
    return s.mask(s.isin(MISSING)).astype(float)


def flag(series):
    """0/1 with NaN kept missing (a plain .eq().astype(float) would turn NaN into 0)."""
    return series.eq(1).map({True: 1.0, False: 0.0}).where(series.notna())


def main():
    cols = load_columns()
    everyone = cols["PUBID_1997"].index

    # ---- person-level constants ----
    person = pd.DataFrame(index=everyone)
    person["sex"] = num(cols, "KEY!SEX_1997").map({1: "male", 2: "female"})
    person["race_ethnicity"] = num(cols, "KEY!RACE_ETHNICITY_1997").map(
        {1: "black", 2: "hispanic", 3: "other", 4: "other"})
    person["asvab_percentile"] = num(cols, "ASVAB_MATH_VERBAL_SCORE_PCT_XRND") / 1000  # stored x1000
    for who, col_name in [("MOM", "mother_educ_grade"), ("DAD", "father_educ_grade")]:
        grades = num(cols, f"CV_HGC_BIO_{who}_1997")
        person[col_name] = grades.mask(grades == 95)  # 95 = ungraded
    person["lived_with_both_parents_age12"] = flag(num(cols, "CV_YTH_REL_HH_AGE_12_1997"))  # 1 = both bio parents
    person["family_income_1997"] = num(cols, "CV_INCOME_GROSS_YR_1997")

    # Big Five from the 2008 TIPI items: mean of the two items per trait,
    # the reverse item flipped (8 - x).
    tipi = {i: num(cols, f"YTEL_TIPIA_{i:06d}_2008") for i in range(1, 11)}
    flip = lambda i: 8 - tipi[i]
    person["big5_extraversion"] = (tipi[1] + flip(6)) / 2
    person["big5_agreeableness"] = (tipi[7] + flip(2)) / 2
    person["big5_conscientiousness"] = (tipi[3] + flip(8)) / 2
    person["big5_emotional_stability"] = (tipi[9] + flip(4)) / 2
    person["big5_openness"] = (tipi[5] + flip(10)) / 2

    # ---- one frame per round: kept rows, plus carried-forward values for everyone ----
    rounds = []
    carried_rounds = []
    for y in ROUNDS:
        marstat = num(cols, f"CV_MARSTAT_{y}")
        age = num(cols, f"CV_AGE_INT_DATE_{y}")
        keep = marstat.isin(range(1, 11)) & age.ge(18)
        part = pd.DataFrame(index=marstat[keep].index)
        part["round"] = ROUND_NUM[y]
        # fieldwork can spill into the next calendar year; the actual year matters
        # for the calendar-year work variables below
        part["year"] = num(cols, f"CV_INTERVIEW_DATE_Y_{y}").fillna(y)
        part["age"] = age[keep]
        # 1,5,7,9 cohabiting; 3,4 married (spouse present/absent); 2,6,8,10 not partnered
        part["partnered"] = marstat[keep].isin([1, 3, 4, 5, 7, 9]).astype(np.int8)
        part["weight"] = num(cols, f"SAMPLING_WEIGHT_CC_{y}") / 100  # two implied decimals

        enroll = num(cols, f"CV_ENROLLSTAT_{y}")
        part["enrolled"] = enroll.map(ENROLLED).fillna("not_enrolled").where(enroll.notna())

        degree = num(cols, f"CV_HIGHEST_DEGREE_EVER_EDT_{y}") if f"CV_HIGHEST_DEGREE_EVER_EDT_{y}" in cols \
            else num(cols, f"CV_HIGHEST_DEGREE_EVER_{y}")
        grade = num(cols, f"CV_HGC_EVER_EDT_{y}") if f"CV_HGC_EVER_EDT_{y}" in cols \
            else num(cols, f"CV_HGC_EVER_{y}")
        education = enroll.map(EDU_FROM_ENROLL).fillna(degree.map(EDU_FROM_DEGREE))
        from_grade = pd.Series(np.select([grade >= 13, grade >= 12],
                                          ["some_college", "hs"], default="less_than_hs"),
                               index=everyone)
        part["education"] = education.fillna(from_grade).where(education.notna() | grade.notna())

        part["census_region"] = num(cols, f"CV_CENSUS_REGION_{y}").map(REGION)
        part["urban"] = num(cols, f"CV_URBAN_RURAL_{y}").map({1: 1.0, 0: 0.0})
        part["msa"] = num(cols, f"CV_MSA_{y}").map(MSA)
        part["general_health"] = num(cols, f"YHEA_100_{y}")  # 1 excellent .. 5 poor

        # own earnings in the year before the interview: exact amount, else bracket midpoint
        wages = num(cols, f"YINC_1700_{y}").fillna(
            num(cols, f"YINC_1800_{y}").map(WAGE_BRACKETS))
        biz = num(cols, f"YINC_2100_{y}")  # negative = business loss
        biz = biz.fillna(num(cols, f"YINC_2000_{y}").eq(0).map({True: 0.0}))  # answered "no business income"
        biz = biz.fillna(num(cols, f"YINC_2200_{y}").map(BIZ_BRACKETS))  # bracket 1 "lost money" has no amount
        part["earnings"] = wages + biz

        # weeks/hours worked are calendar-year variables; take the calendar year
        # before each person's actual interview year
        weeks = pd.Series(np.nan, index=part.index)
        hours = pd.Series(np.nan, index=part.index)
        for yy in ((part["year"] - 1) % 100).unique():
            asked = ((part["year"] - 1) % 100 == yy)
            weeks[asked] = num(cols, f"CVC_WKSWK_YR_ALL_{int(yy):02d}_XRND")
            hours[asked] = num(cols, f"CVC_HOURS_WK_YR_ALL_{int(yy):02d}_XRND")
        part["weeks_worked"] = weeks
        part["hours_worked"] = hours
        part["employed"] = weeks.gt(0).map({True: 1.0, False: 0.0}).where(weeks.notna())

        # carried forward from every round, including before age 18
        carry = pd.DataFrame(index=everyone)
        # height was asked 1997-2011 under changing question names
        ft = pd.Series(np.nan, index=everyone)
        inch = pd.Series(np.nan, index=everyone)
        for ft_name, in_name in [(f"YHEA_2000_{y}", f"YHEA_2100_{y}"),
                                 (f"YHEA_2050_{y}", f"YHEA_2100_{y}"),
                                 (f"YSAQ_000A_000001_{y}", f"YSAQ_000A_000002_{y}"),
                                 (f"YSAQ_000A000001_{y}", f"YSAQ_000A000002_{y}")]:
            if ft_name in cols and in_name in cols:
                ft = ft.fillna(num(cols, ft_name))
                inch = inch.fillna(num(cols, in_name))
        total_in = ft * 12 + inch
        carry["height_cm"] = total_in.where(total_in.between(48, 84)) * IN_TO_CM  # drop implausible

        lb = pd.Series(np.nan, index=everyone)
        for lb_name in [f"YHEA_2200_{y}", f"YHEA_2300_{y}", f"YSAQ_000B_{y}", f"YHEA_SAQ_000B_{y}"]:
            if lb_name in cols:
                lb = lb.fillna(num(cols, lb_name))
        carry["weight_kg"] = lb.where(lb.between(70, 600)) * LB_TO_KG  # drop implausible

        # religion: asked only in some rounds, with era-specific codings
        for pref_name, pref_map in [(f"YINF_3600_{y}", RELIGION_1997),
                                    (f"YHHI_55709_{y}", RELIGION_2005),
                                    (f"YHHI_55708_REV_{y}", RELIGION_2017)]:
            if pref_name in cols:
                v = num(cols, pref_name)
                carry["religion"] = v.map(pref_map).fillna("other").where(
                    v.notna() & ~v.isin([995, 999]))  # supervisor review / uncodable
        for q in [f"YSAQ_282A_{y}", f"YHHI_SAQ_282A_{y}"]:
            if q in cols:
                carry["attendance_worship"] = num(cols, q)  # 1 never .. 8 everyday
        for q in [f"YSAQ_282A7_{y}", f"YHHI_SAQ_282A7_{y}"]:
            if q in cols:
                carry["importance_faith"] = num(cols, q)  # 1 extremely .. 5 not at all
        carried_rounds.append(carry)

        # non-resident biological children: all-ages count to 2017, under-18 after
        nr_name = f"CV_BIO_CHILD_NR_{y}" if f"CV_BIO_CHILD_NR_{y}" in cols else f"CV_BIO_CHILD_NR_U18_{y}"
        raw = cols[nr_name]
        # -4 valid skip = no biological child ever reported; other negatives stay missing
        part["nonresident_children"] = raw.replace(-4, 0).mask(raw.isin([-1, -2, -3, -5])).astype(float)[keep]

        part["_year_key"] = y  # the round's nominal calendar year, internal join key
        rounds.append(part)

    long = pd.concat(rounds)

    # forward-fill the carried columns across all rounds (never backward),
    # then bring each kept row the value that was latest at its round
    carried_names = ["religion", "attendance_worship", "importance_faith", "height_cm", "weight_kg"]
    carried = pd.concat(carried_rounds)
    carried["_year_key"] = [y for y in ROUNDS for _ in range(len(everyone))]
    carried = carried.sort_index(kind="stable")
    carried[carried_names] = carried.groupby(level=0, sort=False)[carried_names].ffill()
    carried = carried.reset_index().set_index(["R0000100", "_year_key"])
    key = pd.MultiIndex.from_arrays([long.index, long["_year_key"]])
    for name in carried_names:
        long[name] = carried[name].reindex(key).to_numpy()
    long["bmi"] = long["weight_kg"] / (long["height_cm"] / 100) ** 2
    # independent height/weight bounds can still combine into an implausible BMI
    implausible = long["bmi"].notna() & ~long["bmi"].between(13, 60)
    long.loc[implausible, ["bmi", "height_cm", "weight_kg"]] = np.nan

    for name, s in person.items():
        long[name] = s.reindex(long.index)
    long.insert(0, "person_id", "nlsy97-" + long.index.astype(str))
    long.insert(1, "source", "nlsy97")

    # TIPI was measured in round 12 (2008); earlier rounds must not use it
    for trait in ["extraversion", "agreeableness", "conscientiousness",
                  "emotional_stability", "openness"]:
        long[f"big5_{trait}"] = long[f"big5_{trait}"].where(long["round"] >= ROUND_NUM[2008])

    columns = [
        "person_id", "source", "year", "round", "weight", "age", "sex", "race_ethnicity",
        "education", "employed", "height_cm", "bmi", "partnered",
        "earnings", "weeks_worked", "hours_worked", "enrolled", "census_region",
        "urban", "msa", "general_health", "asvab_percentile", "mother_educ_grade",
        "father_educ_grade", "lived_with_both_parents_age12", "family_income_1997",
        "religion", "attendance_worship", "importance_faith",
        "big5_extraversion", "big5_agreeableness", "big5_conscientiousness",
        "big5_emotional_stability", "big5_openness", "weight_kg",
        "nonresident_children",
    ]
    long = long[columns].reset_index(drop=True)
    long["year"] = long["year"].astype(int)
    long["age"] = long["age"].astype(int)
    long["round"] = long["round"].astype(np.int8)
    # person_id x round is the key the "partnered 2 years later" step will join on
    assert not long.duplicated(["person_id", "round"]).any()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(OUT, index=False)

    print(f"rows {len(long):,}  people {long['person_id'].nunique():,}  -> {OUT}")
    print("\npartnered rate by sex:")
    print(long.groupby("sex")["partnered"].agg(["count", "mean"]).round(3).to_string())
    bands = pd.cut(long["age"], [17, 24, 29, 34, 39, 43],
                   labels=["18-24", "25-29", "30-34", "35-39", "40-43"])
    print("\npartnered rate by age band:")
    print(long.groupby(bands, observed=True)["partnered"].agg(["count", "mean"]).round(3).to_string())
    print(f"\nrows at age > 43: {(long['age'] > 43).sum()}")
    print("\nmissing share per column:")
    print(long.isna().mean().round(3).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
