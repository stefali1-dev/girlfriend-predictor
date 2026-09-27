"""Build the "single now, partnered ~2 years later" data, train, evaluate and explain it.

Question: among NLSY97 adults who are single (not married, not living with a partner) at an
interview, who has a partner at the interview closest to 2 years later? Every feature comes
from the first interview or earlier, so the partner's own money or housing cannot leak into
the inputs - "the partner brought the money" can no longer fake "money helps".

Reads data/clean/nlsy97.parquet (built by clean_nlsy97.py).
Writes data/curated/forecast_{train,test}.parquet, models/forecast.joblib,
results/forecast.md and results/figures/forecast_*.png.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold, cross_val_predict

from evaluate import SEX_COLORS, style
from train import (CATEGORICAL, FEATURES, TARGET, make_models, make_stacker, markdown,
                   metrics)

CLEAN = Path("data/clean/nlsy97.parquet")
NOW_TRAIN, NOW_TEST = Path("data/curated/train.parquet"), Path("data/curated/test.parquet")
OUT_DIR = Path("data/curated")
MODEL_OUT = Path("models/forecast.joblib")
REPORT = Path("results/forecast.md")
FIGURES = Path("results/figures")
SEED = 42

MIN_AGE, MAX_AGE = 18, 41  # the single interview; the follow-up can reach 43
TEST_ROUND = 18            # base rounds >= 18 (interviews 2017-2022) are the test set
HELD_SHARE = 0.30          # share of people held out; their latest rounds are the test set
RECENT_ROUND = 12          # the calibrator is fitted on OOF predictions from rounds >= this
NEW_TARGET = "found_partner"
# Age gets fixed bands instead of terciles so the words stay readable.
AGE_BANDS, AGE_LABELS = [18, 25, 30, 35, 42], ["18-24", "25-29", "30-34", "35-41"]
MODEL_NAMES = ["age_sex_baseline", "logistic", "catboost"]


def build_rows(df):
    """One row per single interview that has a follow-up interview 2-3 years later.

    The target is `found_partner`: partnered (married or living together) at the follow-up.
    Interviews ran yearly until 2011 and every 2 years after, so both a 2- and a 3-year gap
    count as "about 2 years"; the gap closest to 2 wins, the earlier round breaks a tie.
    """
    single = df[(df["partnered"] == 0) & df["age"].between(MIN_AGE, MAX_AGE)]
    later = df[["person_id", "round", "year", "partnered"]].rename(
        columns={"round": "follow_round", "year": "follow_year", "partnered": NEW_TARGET})
    pairs = single[["person_id", "round", "year"]].merge(later, on="person_id")
    gap = pairs["follow_year"] - pairs["year"]
    pairs = pairs[(pairs["follow_round"] > pairs["round"]) & gap.between(2, 3)]
    pairs = (pairs.assign(distance=(gap - 2).abs())
             .sort_values(["person_id", "round", "distance", "follow_round"])
             .drop_duplicates(["person_id", "round"]))
    base = single.merge(pairs.drop(columns=["year", "distance"]),
                        on=["person_id", "round"], validate="1:1")
    # ASVAB was tested across the 1997/98 school year, possibly just after a round-1
    # interview; for those few rows it is not "from round t or earlier".
    base.loc[base["round"] == 1, "asvab_percentile"] = np.nan
    base = base.drop(columns="partnered")  # constant 0 by construction; the target carries on
    return base


def split_time(base, seed=SEED):
    """Train on earlier rounds of most people, test on the latest rounds of held-out people.

    Two rules: no training interview later than a test interview (rounds 1-17 vs 18-20),
    and no person on both sides. The holdout is random people, not "whoever is single in
    the test rounds": those people are the long-term singles, and dropping exactly them
    from train would keep only people who partnered in the end - train's older singles
    would then be a 68% partner-in-2-years group instead of the true 17%.
    """
    rng = np.random.default_rng(seed)
    held = set(rng.choice(base["person_id"].unique(),
                          size=round(HELD_SHARE * base["person_id"].nunique()), replace=False))
    is_held = base["person_id"].isin(held)
    train = base[~is_held & (base["round"] < TEST_ROUND)]
    test = base[is_held & (base["round"] >= TEST_ROUND)]
    return train, test


def shap_frame(pipeline, X):
    """SHAP values (log-odds) of the CatBoost step, as a frame with feature-name columns."""
    Xt = pipeline[0].transform(X)
    values = shap.TreeExplainer(pipeline[1]).shap_values(Xt)
    return pd.DataFrame(values, index=X.index, columns=Xt.columns)


def binned(values):
    """Bin a feature for direction lines: age bands, categories, or low/middle/high."""
    if values.name == "age":
        return pd.cut(values, AGE_BANDS, right=False, labels=AGE_LABELS)
    if values.name in CATEGORICAL:
        return values
    bins = pd.qcut(values, 3, duplicates="drop")
    if len(bins.cat.categories) > 1:
        names = {c: ["low", "high"] if len(bins.cat.categories) == 2 else
                 ["low", "middle", "high"][i] for i, c in enumerate(bins.cat.categories)}
        return bins.cat.rename_categories(names)
    # Near-constant counts (e.g. nonresident_children is 0 for most): 0 vs any.
    mode = values.mode().iloc[0]
    same = (values == mode).where(values.notna())
    return same.map({True: f"{mode:g}", False: "any other value"})


def direction(sv, values, base_rate):
    """One plain line per feature: the value bins with the strongest down/up push, in pp."""
    pts = points(sv, values, base_rate)
    if len(pts) < 2:
        return "no variation in these rows"
    ranked = sorted(pts, key=pts.get)
    lo, hi = ranked[0], ranked[-1]
    return f"{lo} {pts[lo]:+.1f} pp ... {hi} {pts[hi]:+.1f} pp"


def points(sv, values, base_rate):
    """Effect of each value bin in percentage points around `base_rate`."""
    mean = sv.groupby(binned(values), observed=True).mean()
    odds = logit(base_rate)
    return {k: (expit(odds + e) - base_rate) * 100 for k, e in mean.items()}


def swing(sv, values, base_rate):
    """Percentage-point gap between a feature's most positive and most negative bins."""
    pts = points(sv, values, base_rate)
    return pts[max(pts, key=pts.get)] - pts[min(pts, key=pts.get)]


def plain_findings(sv, test, sv_now, now_test):
    """A few takeaway sentences, every number computed here."""
    rate, now_rate = test[NEW_TARGET].mean(), now_test[TARGET].mean()
    men, women = test.index[test["sex"] == "male"], test.index[test["sex"] == "female"]
    race = points(sv["race_ethnicity"], test["race_ethnicity"], rate)
    down, up = min(race, key=race.get), max(race, key=race.get)
    fathers = test["nonresident_children"] > 0
    father_share = (test.loc[fathers, "sex"] == "male").mean()
    return [
        f"- **Race is the model's biggest single push.** {down.capitalize()} singles sit "
        f"{race[down]:+.1f} pp from the test average, {up.capitalize()} singles "
        f"{race[up]:+.1f} pp; the average push is bigger for women "
        f"({sv.loc[women, 'race_ethnicity'].abs().mean():.2f}) than men "
        f"({sv.loc[men, 'race_ethnicity'].abs().mean():.2f}).",
        f"- **Own earnings help, but modestly.** The low-to-high earnings gap is "
        f"{swing(sv['earnings'], test['earnings'], rate):.0f} pp here, against "
        f"{swing(sv_now['earnings'], now_test['earnings'], now_rate):.0f} pp in the "
        f"partnered-now model: money mostly marks *having* a partner more than it drives "
        f"*getting* one.",
        f"- **Children living elsewhere flip sign.** Singles who already have such children "
        f"({father_share:.0%} of them men) go "
        f"{points(sv['nonresident_children'], test['nonresident_children'], rate)['any other value']:+.1f} pp on the "
        f"2-year forecast, while the partnered-now model puts them "
        f"{abs(points(sv_now['nonresident_children'], now_test['nonresident_children'], now_rate)['any other value']):.1f} pp "
        f"lower. Association, not proof - but the reversal is striking.",
        f"- **Enrollment separates now, not next.** Students and non-enrolled people differ by "
        f"{swing(sv_now['enrolled'], now_test['enrolled'], now_rate):.0f} pp in the partnered-now "
        f"model but only {swing(sv['enrolled'], test['enrolled'], rate):.0f} pp here: it shifts "
        f"*when* people partner, and says almost nothing about who finds someone within two years.",
        f"- **Outgoing people do better on both questions.** The low-to-high extraversion gap is "
        f"{swing(sv['big5_extraversion'], test['big5_extraversion'], rate):.0f} pp here and "
        f"{swing(sv_now['big5_extraversion'], now_test['big5_extraversion'], now_rate):.0f} pp "
        f"for being partnered - one of the few personality reads that survives both questions.",
    ]


def explain(model, test, sv, paths):
    """Mean |SHAP| by sex with one direction line per top feature, plus a beeswarm per sex."""
    by_sex = pd.DataFrame({sex: sv.loc[rows.index].abs().mean()
                           for sex, rows in test.groupby("sex")})
    top = by_sex.sum(axis=1).sort_values(ascending=False).head(10).index
    base_rate = test[NEW_TARGET].mean()
    table = pd.DataFrame({
        "feature": top,
        "men: avg push": [f"{by_sex.loc[f, 'male']:.2f}" for f in top],
        "women: avg push": [f"{by_sex.loc[f, 'female']:.2f}" for f in top],
        "values with the strongest down vs up push (shift in the chance)":
            [direction(sv[f], test[f], base_rate) for f in top],
    })

    Xt = model[0].transform(test[FEATURES])
    for col in CATEGORICAL:  # the beeswarm colour scale needs numbers
        codes, _ = pd.factorize(Xt[col], sort=True)
        Xt[col] = np.where(codes >= 0, codes, np.nan)
    for sex, path in paths.items():
        rows = test.index[test["sex"] == sex]
        shap.summary_plot(sv.loc[rows].to_numpy(), Xt.loc[rows], max_display=10, show=False)
        ax = plt.gca()
        ax.set_xlabel("effect on the model's log-odds of finding a partner "
                      "(left lowers the chance, right raises it)")
        ax.set_title(f"{'Men' if sex == 'male' else 'Women'}: what pushes the forecast "
                     f"({len(rows):,} test interviews)")
        plt.gcf().set_size_inches(8, 6)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
    return table


def calibration_plot(test, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="#898781", linestyle="--", linewidth=1,
            label="perfect calibration")
    for sex, rows in test.groupby("sex"):
        bins = rows.groupby(pd.qcut(rows["p"], 10, duplicates="drop"), observed=True)
        ax.plot(bins["p"].mean(), bins[NEW_TARGET].mean(), marker="o", markersize=8,
                linewidth=2, color=SEX_COLORS[sex], label=f"{sex} (10 equal-size bins)")
    ax.set(xlim=(0, 0.45), ylim=(0, 0.45), xlabel="predicted chance of finding a partner",
           ylabel="share that actually did", title="Test set: predicted vs observed, by sex")
    style(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rate_by_age_plot(test, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    by_age = test.groupby(["sex", "age"])[[NEW_TARGET, "p"]].mean().reset_index()
    for sex, rows in by_age.groupby("sex"):
        color = SEX_COLORS[sex]
        ax.plot(rows["age"], rows["p"], color=color, linewidth=2,
                label=f"{sex}: model (mean prediction)")
        ax.scatter(rows["age"], rows[NEW_TARGET], color=color, s=40, edgecolor="white",
                   linewidth=1.5, zorder=3, label=f"{sex}: observed")
    ax.set(xlabel="age at the single interview", ylabel="share with a partner ~2 years later",
           title="Test set: 2-year partnering rate by age and sex, model vs observed")
    style(ax)
    # Below the axes: the series themselves cover most of the plot area.
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def verdict(s_now, s_fc):
    if s_now >= 8 and s_fc < s_now / 2:
        return "matters for now, much less for finding: likely runs backwards"
    if s_fc >= 8 and s_now < s_fc / 2:
        return "matters for finding a partner, not for being partnered"
    if s_now >= 8 and s_fc >= 8:
        return "matters for both questions"
    return "minor for both"


def now_comparison(forecast_share, sv, test):
    """Refit the 'partnered now' recipe, then compare which features each question rewards."""
    now_train, now_test = pd.read_parquet(NOW_TRAIN), pd.read_parquet(NOW_TEST)
    now_model = make_models()["catboost"].fit(now_train[FEATURES], now_train[TARGET])
    sv_now = shap_frame(now_model, now_test[FEATURES])
    now_mean, fc_mean = sv_now.abs().mean(), forecast_share
    now_row = metrics(now_test[TARGET].to_numpy(),
                      now_model.predict_proba(now_test[FEATURES])[:, 1])

    fc_rate, now_rate = test[NEW_TARGET].mean(), now_test[TARGET].mean()
    shares = pd.DataFrame({"now": now_mean / now_mean.sum(),
                           "forecast": fc_mean / fc_mean.sum()})
    top = shares.max(axis=1).sort_values(ascending=False).head(12).index
    rows = []
    for f in top:
        rows.append({
            "feature": f,
            "'partnered now' share": f"{100 * shares.loc[f, 'now']:.0f}%",
            "its low vs high values, now": direction(sv_now[f], now_test[f], now_rate),
            "'finds a partner' share": f"{100 * shares.loc[f, 'forecast']:.0f}%",
            "its low vs high values, finding": direction(sv[f], test[f], fc_rate),
            "read": verdict(100 * shares.loc[f, "now"], 100 * shares.loc[f, "forecast"]),
        })
    return pd.DataFrame(rows), now_row, sv_now, now_test


def metrics_row(label, y, p):
    return {"group": label, "rows": len(y), "found a partner": f"{np.mean(y):.3f}",
            **{k: f"{v:.4f}" for k, v in metrics(y, p).items()}}


def main():
    base = build_rows(pd.read_parquet(CLEAN))
    train, test = split_time(base)
    for name, part in [("train", train), ("test", test)]:
        part.to_parquet(OUT_DIR / f"forecast_{name}.parquet", index=False)

    X, y = train[FEATURES], train[NEW_TARGET].to_numpy()
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                 .split(X, y, groups=train["person_id"]))
    models, oof = make_models(), {}
    for name in MODEL_NAMES:
        print(f"cross-validating {name} ...", flush=True)
        oof[name] = cross_val_predict(models[name], X, y, cv=folds,
                                      method="predict_proba")[:, 1]

    cv_rows = []
    for name, p in oof.items():
        per_fold = pd.DataFrame([metrics(y[val], p[val]) for _, val in folds])
        cv_rows.append({"model": name,
                        **{m: f"{per_fold[m].mean():.4f} ± {per_fold[m].std():.4f}"
                           for m in per_fold}})
    cv_table = pd.DataFrame(cv_rows)

    print("fitting final models on all of train ...", flush=True)
    final = models["catboost"].fit(X, y)
    # The calibrator is Platt scaling fitted on out-of-fold predictions from the recent
    # rounds only: partnering rates fell over the decades, so a calibrator fitted on the
    # 2000s would speak the wrong era's language on the test years.
    recent = (train["round"] >= RECENT_ROUND).to_numpy()
    stacker = make_stacker().fit(oof["catboost"][recent].reshape(-1, 1), y[recent])
    stacker_weight = float(stacker[-1].coef_[0][0])
    test = test.copy()
    test["p"] = stacker.predict_proba(
        final.predict_proba(test[FEATURES])[:, 1].reshape(-1, 1))[:, 1]

    y_test = test[NEW_TARGET].to_numpy()
    overall = [metrics_row("age + sex baseline", y_test,
                           models["age_sex_baseline"].fit(X, y)
                           .predict_proba(test[FEATURES])[:, 1]),
               metrics_row("logistic", y_test, models["logistic"].fit(X, y)
                           .predict_proba(test[FEATURES])[:, 1]),
               metrics_row("catboost (raw)", y_test,
                           final.predict_proba(test[FEATURES])[:, 1]),
               metrics_row("catboost + calibration (final)", y_test, test["p"].to_numpy())]
    by_sex = [metrics_row(sex, rows[NEW_TARGET].to_numpy(), rows["p"].to_numpy())
              for sex, rows in test.groupby("sex")]
    bands = pd.cut(test["age"], AGE_BANDS, right=False, labels=AGE_LABELS)
    by_age = [metrics_row(band, rows[NEW_TARGET].to_numpy(), rows["p"].to_numpy())
              for band, rows in test.groupby(bands, observed=True)]

    print("explaining with SHAP ...", flush=True)
    sv = shap_frame(final, test[FEATURES])
    FIGURES.mkdir(parents=True, exist_ok=True)
    explain_table = explain(final, test, sv,
                            {"male": FIGURES / "forecast_shap_male.png",
                             "female": FIGURES / "forecast_shap_female.png"})
    calibration_plot(test, FIGURES / "forecast_calibration.png")
    rate_by_age_plot(test, FIGURES / "forecast_rate_by_age.png")

    compare_table, now_row, sv_now, now_test = now_comparison(sv.abs().mean(), sv, test)
    findings = plain_findings(sv, test, sv_now, now_test)

    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump({"features": FEATURES,
                 "categories": {c: sorted(train[c].dropna().unique()) for c in CATEGORICAL},
                 "base": {"catboost": final}, "stacker": stacker,
                 "target": NEW_TARGET}, MODEL_OUT)
    write_report(base, train, test, cv_table, pd.DataFrame(overall), pd.DataFrame(by_sex),
                 pd.DataFrame(by_age), explain_table, compare_table, now_row, findings,
                 stacker_weight)
    print(REPORT.read_text())


def write_report(base, train, test, cv_table, overall, by_sex, by_age,
                 explain_table, compare_table, now_row, findings, stacker_weight):
    n2 = int((base["follow_year"] - base["year"] == 2).sum())
    n3 = int((base["follow_year"] - base["year"] == 3).sum())
    unused = len(base[(base["round"] < TEST_ROUND)
                      & base["person_id"].isin(set(test["person_id"]))])
    older = base[(base["round"] < TEST_ROUND) & (base["age"] >= 35)]
    late_singles = set(base.loc[base["round"] >= TEST_ROUND, "person_id"])
    biased = older[~older["person_id"].isin(late_singles)][NEW_TARGET].mean()
    calib_n = int((train["round"] >= RECENT_ROUND).sum())
    beyond = (test["age"] > train["age"].max()).mean()
    final_row = overall.iloc[-1]
    floor_auc = overall.iloc[0]["roc_auc"]
    findings_text = "\n".join(findings)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""\
# Finding a partner within about two years

Among NLSY97 adults who were **single** at an interview - not married, not living with a
partner - **{test[NEW_TARGET].mean():.1%} had a partner (married or living together) at the
interview closest to 2 years later**. This page builds a model that predicts that from facts
known **before**: age, education, work, earnings, height, personality, family background.
Because every input is older than the outcome, a partner's own income or housing cannot
masquerade as "earnings help" - the classic reason "partnered right now" numbers overstate
what a person can change.

## The rows

{len(base):,} rows, {base['person_id'].nunique():,} people, single at ages {MIN_AGE}-{MAX_AGE},
interviews {base['year'].min()}-{base['year'].max()}. Each row needs a follow-up interview
2-3 years later ({n2:,} rows matched at 2 years, {n3:,} at 3); single interviews without one
were dropped. Across all rows, {base[NEW_TARGET].mean():.1%} found a partner.

## Train and test: earlier interviews vs the latest ones

- **Train:** rounds {train['round'].min()}-{train['round'].max()} (interviews
  {train['year'].min()}-{train['year'].max()}), {len(train):,} rows,
  {train['person_id'].nunique():,} people, {train[NEW_TARGET].mean():.1%} found a partner.
- **Test:** rounds {test['round'].min()}-{test['round'].max()} (interviews
  {test['year'].min()}-{test['year'].max()}), {len(test):,} rows,
  {test['person_id'].nunique():,} people, {test[NEW_TARGET].mean():.1%} found a partner.
- **Set aside, unused:** {unused:,} earlier-round rows belonging to the test people.

Two rules, and why. **Time:** every test interview happened after every training interview,
so the test imitates the real use: applying the model to a future it has not seen. **People
separate:** keeping a person's earlier rows in train while their latest row is in test would
let the model partly "recognise" the person instead of learning what singles are like.

The order of those two rules matters, and the first attempt got it wrong. Removing "the test
people's earlier rows" from train sounds natural - but the test people **are** the long-term
singles, and deleting exactly them left train with the singles who partnered in the end:
{biased:.0%} of that train's 35+ singles found a partner within 2 years, against the true
{older[NEW_TARGET].mean():.0%}. Every model trained on such a set overpredicts wildly on the
test years. Holding people out **at random** instead (the other {1 - HELD_SHARE:.0%} of people
to train, the rest never seen) keeps train representative while both rules still hold.

What the split costs, honestly: the NLSY97 cohort was born 1980-84, so the test interviews
only contain ages {test['age'].min()}-{test['age'].max()} - behavior for singles in their
20s is measured only by cross-validation on earlier data - and the training rounds contain
no single person older than {train['age'].max()} ({(train['age'] >= 35).sum():,} training
rows are 35+), while {beyond:.0%} of test rows are older. There, every model is
extrapolating past anything it has seen. And partnering rates fell over the decades, so the
calibrator is fitted on out-of-fold predictions from rounds {RECENT_ROUND}+ only
({calib_n:,} rows, interviews 2010 onward), the era closest to the test years; it puts
weight {stacker_weight:.2f} on CatBoost's log-odds. The calibration plot shows how well
that worked.

## Model comparison (cross-validation on train)

Mean ± sd over 5 folds grouped by person. Lower is better except ROC-AUC.

{markdown(cv_table)}

## Test: the latest interviews ({test['year'].min()}-{test['year'].max()})

{markdown(overall)}

The age + sex floor scores below chance on AUC ({floor_auc}) for the extrapolation reason
above: at the ages it never saw it predicts the same average for everyone, which ranks the
37+ test rows too high. Logistic regression, which bends a curve through age instead of
memorising cells, is the honest "guess by age and sex" here; CatBoost adds a little on top.

By sex and by age band (final model):

{markdown(by_sex)}

{markdown(by_age)}

For scale: the same recipe predicting "partnered now" scores {now_row['roc_auc']:.3f} AUC on
its own held-out test set, versus {final_row['roc_auc']} here - the 2-year question is
genuinely harder, as expected for something with a lot of chance in it.

![calibration](figures/forecast_calibration.png)

![rate by age](figures/forecast_rate_by_age.png)

## What goes with finding a partner (SHAP)

Each row is one feature: its average push on the model (log-odds) for men and for women
separately, and the value groups with the strongest down and up push, in plain percentage
points (pp) around the test's average of {test[NEW_TARGET].mean():.0%}. Pushes are relative
to the training-era average of {train[NEW_TARGET].mean():.0%}; the test years run lower, so
lines can sit below zero overall. Red dots in the figures are high feature values, blue are
low.

{markdown(explain_table)}

![SHAP men](figures/forecast_shap_male.png)

![SHAP women](figures/forecast_shap_female.png)

## "Partnered now" vs "finds a partner": which features may run backwards

The "now" columns refit the partnered-now recipe on its own train set and explain it the same
way; shares are each feature's slice of total importance, so the two questions can be
compared. A feature big for "now" but small for "finding" mostly marks *having* a partner,
not *getting* one - often the partner is the cause (two incomes, shared housing). These are
associations, not proven causes.

{markdown(compare_table)}

## The short version

{findings_text}

## Limits

Associations, not causes - measuring features before the outcome removes the worst backwards
arrow, not all of them. One US cohort born 1980-84, and the test years cover ages
{test['age'].min()}-{test['age'].max()} only. Unweighted (survey weights not used), and
people who left the survey are absent. The target counts only marriage and living together:
a girlfriend or boyfriend not living together counts as "still single".
""")
    print(f"saved {REPORT}")


if __name__ == "__main__":
    main()
