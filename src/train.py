"""Compare models with grouped cross-validation, stack the useful ones, save the final model.

Train set only; the test set stays closed until evaluate.py.
Writes models/partnered.joblib and results/cv_results.md.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, OrdinalEncoder,
                                   SplineTransformer, StandardScaler)
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

TRAIN = Path("data/curated/train.parquet")
MODEL_OUT = Path("models/partnered.joblib")
CV_OUT = Path("results/cv_results.md")
SEED = 42
TARGET = "partnered"

CATEGORICAL = ["sex", "race_ethnicity", "education", "enrolled", "census_region", "msa", "religion"]
DOLLARS = ["earnings", "family_income_1997"]
NUMERIC = [
    "height_cm", "bmi", "weeks_worked", "hours_worked", "urban", "general_health",
    "asvab_percentile", "mother_educ_grade", "father_educ_grade",
    "lived_with_both_parents_age12", "attendance_worship", "importance_faith",
    "big5_extraversion", "big5_agreeableness", "big5_conscientiousness",
    "big5_emotional_stability", "big5_openness", "nonresident_children",
]
# Left out: ids, round/year (with a 5-year birth cohort, year is almost age again),
# sampling weight, employed (= weeks_worked > 0), weight_kg (height + bmi carry it).
FEATURES = ["age"] + CATEGORICAL + DOLLARS + NUMERIC

# Models whose out-of-fold predictions feed the final stack. Cross-validation showed every
# stack scoring the same as CatBoost alone (log loss 0.546), so the final "stack" is CatBoost
# with the stacker acting as its calibrator: one fast model that TreeSHAP can explain exactly.
# To stack more, list them here, e.g. ["catboost", "tabpfn"].
STACKED = ["catboost"]


def sex_age_cell(X):
    return (X["sex"] + "_" + X["age"].astype(str)).to_frame()


def make_models():
    # Floor to beat: one probability per (sex, age) cell. One-hot cells + a barely
    # regularised logistic regression = the partnered rate of each cell in the training folds.
    baseline = make_pipeline(
        FunctionTransformer(sex_age_cell),
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(C=100, max_iter=1000),
    )

    # Everything that happens to the data sits inside the Pipeline, so cross-validation
    # refits the imputer medians and scaler on each training fold (no peeking at the
    # validation fold), and predict-time data gets exactly the same treatment.
    missing_as_median = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                      StandardScaler())
    logistic = make_pipeline(
        ColumnTransformer([
            # Age is curved (steep rise in the 20s, flat after ~35): splines let a linear model bend.
            ("age", SplineTransformer(n_knots=5), ["age"]),
            # arcsinh is log-like but defined for the few negative (business loss) earnings.
            ("dollars", make_pipeline(FunctionTransformer(np.arcsinh), missing_as_median), DOLLARS),
            ("num", missing_as_median, NUMERIC),
            ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                                  OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
        ]),
        LogisticRegression(max_iter=5000),
    )

    # CatBoost handles missing numbers and text categories itself; it only needs
    # missing categories spelled out.
    catboost = make_pipeline(
        ColumnTransformer(
            [("cat", SimpleImputer(strategy="constant", fill_value="missing"), CATEGORICAL),
             ("num", "passthrough", ["age"] + DOLLARS + NUMERIC)],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas"),
        # cat_features as a tuple: CatBoost copies a list, which breaks scikit-learn's clone().
        CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=6,
                           cat_features=tuple(CATEGORICAL), random_seed=SEED, verbose=0,
                           allow_writing_files=False),
    )

    # TabPFN v2: a transformer pre-trained on millions of synthetic tables; it predicts by
    # "reading" training rows as context instead of fitting weights. Context size is its limit
    # (and CPU time), so each of its 4 ensemble members sees a random 5,000 training rows.
    # v2 weights download without a login; newer versions need a license account.
    tabpfn = make_pipeline(
        ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
             CATEGORICAL),
            ("num", "passthrough", ["age"] + DOLLARS + NUMERIC),
        ]),
        TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, device="cpu", n_estimators=4,
            categorical_features_indices=list(range(len(CATEGORICAL))),
            ignore_pretraining_limits=True, inference_config={"SUBSAMPLE_SAMPLES": 5000},
            random_state=SEED),
    )

    return {"age_sex_baseline": baseline, "logistic": logistic,
            "catboost": catboost, "tabpfn": tabpfn}


def make_stacker():
    # Stacking: a small logistic regression learns how much to trust each model, from
    # predictions the models made on rows they did not train on. Working on the logit
    # (log-odds) scale makes it Platt scaling of a weighted blend, so the stack is also
    # the calibration step: its output is fitted to match observed rates. With one model in
    # the stack it is plain Platt scaling.
    return make_pipeline(
        FunctionTransformer(np.clip, kw_args={"a_min": 1e-6, "a_max": 1 - 1e-6}),
        FunctionTransformer(logit),
        LogisticRegression(),
    )


def calibration_error(y, p, bins=10):
    """Expected calibration error: mean |predicted - observed rate| over 10 equal-size bins."""
    bin_of = pd.qcut(p, bins, labels=False, duplicates="drop")
    table = pd.DataFrame({"y": y, "p": p, "bin": bin_of}).groupby("bin")
    return float((table["p"].mean() - table["y"].mean()).abs().mul(table.size() / len(y)).sum())


def metrics(y, p):
    return {"log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "roc_auc": roc_auc_score(y, p), "calib_error": calibration_error(y, p)}


def markdown(table):
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


def cv_table(oof, y, folds):
    """Mean and standard deviation of each metric across the validation folds."""
    rows = []
    for name, p in oof.items():
        per_fold = pd.DataFrame([metrics(y[val], p[val]) for _, val in folds])
        rows.append({"model": name, **{
            m: f"{per_fold[m].mean():.4f} ± {per_fold[m].std():.4f}" for m in per_fold}})
    return pd.DataFrame(rows)


def main():
    train = pd.read_parquet(TRAIN)
    X, y = train[FEATURES], train[TARGET].to_numpy()

    # Folds grouped by person, for the same reason as the train/test split: all rows of one
    # person are in the same fold. Every model uses the same folds, so their out-of-fold
    # predictions line up row by row for stacking.
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                 .split(X, y, groups=train["person_id"]))

    models = make_models()
    oof = {}
    for name, model in models.items():
        print(f"cross-validating {name} ...", flush=True)
        oof[name] = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]

    # Stacks are scored with the same folds: the stacker is refitted on 4/5 of the
    # out-of-fold predictions and judged on the other 1/5.
    stacks = {"catboost + calibration (stacker on catboost alone)": ["catboost"],
              "stack: logistic + catboost": ["logistic", "catboost"],
              "stack: catboost + tabpfn": ["catboost", "tabpfn"],
              "stack: logistic + catboost + tabpfn": ["logistic", "catboost", "tabpfn"]}
    for name, members in stacks.items():
        Z = np.column_stack([oof[m] for m in members])
        oof[name] = cross_val_predict(make_stacker(), Z, y, cv=folds, method="predict_proba")[:, 1]

    table = cv_table(oof, y, folds)
    print(table.to_string(index=False))

    stacker = make_stacker().fit(np.column_stack([oof[m] for m in STACKED]), y)
    weights = {name: round(float(w), 3) for name, w in zip(STACKED, stacker[-1].coef_[0])}
    print("stacker weights on each model's log-odds:", weights)

    print("fitting final models on all of train ...", flush=True)
    base = {name: models[name].fit(X, y) for name in STACKED}
    MODEL_OUT.parent.mkdir(exist_ok=True)
    categories = {col: sorted(train[col].dropna().unique()) for col in CATEGORICAL}
    joblib.dump({"features": FEATURES, "categories": categories,
                 "base": base, "stacker": stacker}, MODEL_OUT)

    CV_OUT.parent.mkdir(exist_ok=True)
    CV_OUT.write_text(
        "# Cross-validation on train (5 folds grouped by person)\n\n"
        f"{len(train):,} rows, {train['person_id'].nunique():,} people. "
        "Mean ± standard deviation over the 5 validation folds. Lower is better except ROC-AUC.\n\n"
        + markdown(table)
        + f"\n\nFinal model: stack of {', '.join(STACKED)}; "
        f"stacker weights on each model's log-odds: {weights}.\n")
    print(f"saved {MODEL_OUT} and {CV_OUT}")


if __name__ == "__main__":
    main()
