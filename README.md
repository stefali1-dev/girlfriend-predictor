# girlfriend-predictor

Predicts the chance that a person has a partner — "partnered" means married or living
with a partner — from facts about them: age, sex, education, work, earnings, height,
personality, family background. It is trained on NLSY97, so it describes US people born
1980–84 at ages 18–43 and nothing outside that group. A second model asks a related
question: among people who are single, who has a partner about two years later.

## Setup

Runs on any recent Python (built on 3.14). All scripts are run from the repo root.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Data

`data/` and `models/` are not in git; these steps build them.

- **NLSY97** (main training data) — the one manual step. In your NLS Investigator account,
  load the saved tagset `partner-model`, download the extract, and unzip it into
  `data/raw/nlsy97/`. The site can't be scripted.
- **NHANES** (US health survey) — `python src/get_nhanes.py` downloads it into `data/raw/nhanes/`.
- **HCMST 2017–2022** (Stanford, How Couples Meet and Stay Together) — `python src/get_hcmst.py`
  downloads it into `data/raw/hcmst/`.

## Run order

    python src/clean_nlsy97.py   # -> data/clean/nlsy97.parquet (one row per person per survey round)
    python src/clean_nhanes.py   # -> data/clean/nhanes.parquet (one row per adult)
    python src/clean_hcmst.py    # -> data/clean/hcmst.parquet (one row per respondent)
    python src/split.py          # -> data/curated/{train,test}.parquet (20% of people held out for test)
    python src/train.py          # -> models/partnered.joblib, results/cv_results.md
    python src/evaluate.py       # -> results/metrics.md and results/figures/
    python src/explain.py        # -> results/explain_numbers.md and results/figures/
    python src/forecast.py       # -> models/forecast.joblib, results/forecast.md, results/figures/
    python src/outside_check.py  # -> results/outside_check.md (same findings checked on NHANES and HCMST)

## Tests

    pytest

`tests/test_predict.py` runs `src/predict.py` end to end and needs the trained model, so
`split.py` and `train.py` must have run. `tests/test_forecast.py` needs no data.

## Predict one person

    echo '{"age": 29, "sex": "male", "education": "bachelor_plus", "earnings": 55000}' | python src/predict.py

Prints one number, the probability of being partnered (0 to 1). `age` is required; every
other field may be left out or set to `null` and is treated as unknown, exactly like a
missing survey answer. The allowed fields and category values are listed in `src/train.py`
(`FEATURES`, `CATEGORICAL`).

## Results

Read `results/insights.md` first, then `results/forecast.md`, `results/outside_check.md`
and `results/metrics.md`. Charts are in `results/figures/`.
