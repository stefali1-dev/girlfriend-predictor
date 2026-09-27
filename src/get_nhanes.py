"""Download NHANES demographics, body measures, and occupation files."""
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "nhanes"
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{file}_{letter}.xpt"

# cycle letter -> first year. Skips the 2017-March 2020 pre-pandemic cycle (prefix
# P_), which overlaps 2017-2018.
CYCLES = [(2007, "E"), (2009, "F"), (2011, "G"), (2013, "H"), (2015, "I"), (2017, "J"), (2021, "L")]
FILES = ["DEMO", "BMX", "OCQ"]
MARITAL_VARS = {"DMDMARTL", "DMDMARTZ"}  # name changes starting with the 2021-2023 cycle


def download(year: int, letter: str, file: str) -> Path:
    dest = RAW_DIR / f"{file}_{letter}.xpt"
    if dest.exists():
        print(f"skip {dest.name} (already downloaded)")
        return dest
    url = BASE_URL.format(year=year, file=file, letter=letter)
    print(f"downloading {url}")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def check_marital_var(dest: Path) -> None:
    df = pd.read_sas(dest, format="xport")
    if not MARITAL_VARS & set(df.columns):
        raise ValueError(f"no marital status variable found in {dest.name}")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year, letter in CYCLES:
        for file in FILES:
            dest = download(year, letter, file)
            if file == "DEMO":
                check_marital_var(dest)
