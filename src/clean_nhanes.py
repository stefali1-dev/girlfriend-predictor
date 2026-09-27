"""Clean NHANES into data/clean/nhanes.parquet: one row per adult (18+)
with the shared clean columns from decisions.md.

Cycles 2007-2018 (letters E-J) and 2021-2023 (letter L). Height and BMI are
measured in the MEC exam, so the person weight is the MEC weight WTMEC2YR.
"""

import pandas as pd

RAW = "data/raw/nhanes"
OUT = "data/clean/nhanes.parquet"

CYCLE_YEAR = {  # file letter -> first year of the cycle
    "E": 2007, "F": 2009, "G": 2011, "H": 2013, "I": 2015, "J": 2017, "L": 2021,
}
RACE = {1: "hispanic", 2: "hispanic", 4: "black", 3: "other", 5: "other"}
EDUCATION = {  # DMDEDUC2 (adults 20+); refused/don't know dropped
    1: "less_than_hs", 2: "less_than_hs", 3: "hs", 4: "some_college", 5: "bachelor_plus",
}


def clean_cycle(letter):
    demo = pd.read_sas(f"{RAW}/DEMO_{letter}.xpt", format="xport")
    bmx = pd.read_sas(f"{RAW}/BMX_{letter}.xpt", format="xport")[["SEQN", "BMXHT", "BMXBMI"]]
    ocq = pd.read_sas(f"{RAW}/OCQ_{letter}.xpt", format="xport")[["SEQN", "OCD150"]]

    df = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "WTMEC2YR"]].copy()
    # 2021-2023 renamed the marital column and merged married with
    # living-with-partner into one code
    df["marital"] = demo["DMDMARTZ"] if letter == "L" else demo["DMDMARTL"]
    df = df.merge(bmx, on="SEQN", how="left").merge(ocq, on="SEQN", how="left")

    df = df[(df.RIDAGEYR >= 18) & df.marital.notna()]
    # DMDMARTL: 1 married, 2 widowed, 3 divorced, 4 separated, 5 never married,
    # 6 living with partner. DMDMARTZ: 1 married/living with partner,
    # 2 widowed/divorced/separated, 3 never married.
    partnered_codes = {1} if letter == "L" else {1, 6}
    known_codes = {1, 2, 3} if letter == "L" else {1, 2, 3, 4, 5, 6}
    df = df[df.marital.isin(known_codes)]

    return pd.DataFrame({
        "person_id": "nhanes-" + df.SEQN.astype(int).astype(str),
        "source": "nhanes",
        "year": CYCLE_YEAR[letter],
        "age": df.RIDAGEYR.astype(int),
        "sex": df.RIAGENDR.map({1: "male", 2: "female"}),
        "race_ethnicity": df.RIDRETH1.map(RACE),
        "education": df.DMDEDUC2.map(EDUCATION),
        # OCD150 work status last week: 1 working, 2 has a job but not at work,
        # 3 looking for work, 4 not working; missing for proxy interviews
        "employed": df.OCD150.map({1: 1, 2: 1, 3: 0, 4: 0}).astype("Int64"),
        "height_cm": df.BMXHT,
        "bmi": df.BMXBMI,
        "partnered": df.marital.isin(partnered_codes).astype(int),
        "weight": df.WTMEC2YR,
    })


def main():
    df = pd.concat([clean_cycle(letter) for letter in CYCLE_YEAR], ignore_index=True)
    df.to_parquet(OUT, index=False)
    print(f"{OUT}: {len(df)} rows, ages {df.age.min()}-{df.age.max()}")
    print("partnered by sex (share):")
    print(df.groupby("sex").partnered.mean().round(3).to_string())


if __name__ == "__main__":
    main()
