"""Clean the HCMST 2017 wave (wave 1) into data/clean/hcmst.parquet: one row
per respondent with the shared clean columns from decisions.md, plus
any_partner (a romantic/sexual partner they don't live with also counts).

Height and BMI are not asked in HCMST, so those columns are all missing.
"""

import pandas as pd

RAW = "data/raw/hcmst/HCMST_2017_to_2022_small_public_version_2.2.dta"
OUT = "data/clean/hcmst.parquet"

RACE = {
    "Hispanic": "hispanic",
    "Black, Non-Hispanic": "black",
    "White, Non-Hispanic": "other",
    "Other, Non-Hispanic": "other",
    "2+ Races, Non-Hispanic": "other",
}
EDUCATION = {
    "No formal education": "less_than_hs",
    "1st, 2nd, 3rd, or 4th grade": "less_than_hs",
    "5th or 6th grade": "less_than_hs",
    "7th or 8th grade": "less_than_hs",
    "9th grade": "less_than_hs",
    "10th grade": "less_than_hs",
    "11th grade": "less_than_hs",
    "12th grade NO DIPLOMA": "less_than_hs",
    "HIGH SCHOOL GRADUATE - high school DIPLOMA or the equivalent (GED)": "hs",
    "Some college, no degree": "some_college",
    "Associate degree": "some_college",
    "Bachelors degree": "bachelor_plus",
    "Masters degree": "bachelor_plus",
    "Professional or Doctorate degree": "bachelor_plus",
}

df = pd.read_stata(RAW, columns=[
    "caseid_new", "w1_ppage", "w1_ppgender", "w1_ppethm", "w1_ppeduc", "w1_ppwork",
    "w1_partnership_status_cohab", "w1_weight_combo",
])

# w1_partnership_status_cohab (no missing values): married / in unmarried cohab
# partnership / in unmarried noncohab partnership / unpartnered. Derived by
# Stanford from the S1/S2/S3 screens and "are you currently living with partner".
status = df.w1_partnership_status_cohab

out = pd.DataFrame({
    "person_id": "hcmst-" + df.caseid_new.astype(str),
    "source": "hcmst",
    "year": 2017,
    "age": df.w1_ppage.astype(int),
    "sex": df.w1_ppgender.map({"Male": "male", "Female": "female"}),
    "race_ethnicity": df.w1_ppethm.map(RACE),
    "education": df.w1_ppeduc.map(EDUCATION),
    "employed": df.w1_ppwork.isin(["Working - as a paid employee", "Working - self-employed"]).astype(int),
    "height_cm": float("nan"),
    "bmi": float("nan"),
    "partnered": status.isin(["married", "in unmarried cohab partnership"]).astype(int),
    "weight": df.w1_weight_combo,
    "any_partner": status.isin(
        ["married", "in unmarried cohab partnership", "in unmarried noncohab partnership"]
    ).astype(int),
})
out.to_parquet(OUT, index=False)

print(f"{OUT}: {len(out)} rows, ages {out.age.min()}-{out.age.max()}")
by_sex = out.groupby("sex")[["partnered", "any_partner"]].mean().round(3)
print("partnered / any_partner by sex (share):")
print(by_sex.to_string())
