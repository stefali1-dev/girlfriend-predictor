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

from evaluate import SEX_COLORS, metrics_row, style
from train import (CATEGORICAL, FEATURES, TARGET, cv_table, make_models, make_stacker,
                   markdown, metrics)

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
    # Rows used by neither side, in three buckets: the latest rounds of training people
    # (train must stay in the earlier rounds), the earlier rounds of the test people, and
    # everyone held out who never reaches a test-round interview.
    unused = {
        "late": base[~is_held & (base["round"] >= TEST_ROUND)],
        "early": base[is_held & base["person_id"].isin(set(test["person_id"]))
                 & (base["round"] < TEST_ROUND)],
        "never": base[is_held & ~base["person_id"].isin(set(test["person_id"]))],
    }
    return train, test, unused


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


def plain_findings(sv, rows, sv_now, matched, test):
    """A few takeaway sentences, every number computed here.

    `sv`/`rows`: the forecast model explained on its training rows. `sv_now`/`matched`:
    the partnered-now model explained at the forecast test's ages, so sizes compare.
    """
    rate, now_rate = rows[NEW_TARGET].mean(), matched[TARGET].mean()
    men, women = rows.index[rows["sex"] == "male"], rows.index[rows["sex"] == "female"]
    race = points(sv["race_ethnicity"], rows["race_ethnicity"], rate)
    race_line = ", ".join(f"{c} {race[c]:+.1f} pp" for c in ["black", "hispanic", "other"]
                          if c in race)
    kids = lambda frame: frame["nonresident_children"] > 0
    raw_train = 100 * (rows.loc[kids(rows), NEW_TARGET].mean()
                       - rows.loc[~kids(rows), NEW_TARGET].mean())
    raw_by_sex = {sex: 100 * (g.loc[kids(g), NEW_TARGET].mean()
                              - g.loc[~kids(g), NEW_TARGET].mean())
                  for sex, g in rows.groupby("sex")}
    raw_test = 100 * (test.loc[kids(test), NEW_TARGET].mean()
                      - test.loc[~kids(test), NEW_TARGET].mean())
    students = int((test["enrolled"] != "not_enrolled").sum())
    return [
        f"- **Race is the model's biggest push** ({race_line} around the training average; "
        f"bigger for women ({sv.loc[women, 'race_ethnicity'].abs().mean():.2f}) than men "
        f"({sv.loc[men, 'race_ethnicity'].abs().mean():.2f})). One caveat: the survey "
        f"oversamples Black respondents - {(test['race_ethnicity'] == 'black').mean():.0%} "
        f"of test rows - so unweighted shares flatter race.",
        f"- **Own earnings help, but modestly.** The low-to-high earnings gap is "
        f"{swing(sv['earnings'], rows['earnings'], rate):.0f} pp here, against "
        f"{swing(sv_now['earnings'], matched['earnings'], now_rate):.0f} pp in the "
        f"partnered-now model at matched ages: money mostly marks *having* a partner more "
        f"than it drives *getting* one.",
        f"- **Children living elsewhere flip sign - among otherwise similar singles.** The "
        f"forecast model puts such singles "
        f"{points(sv['nonresident_children'], rows['nonresident_children'], rate)['any other value']:+.1f} pp "
        f"up, the partnered-now model "
        f"{abs(points(sv_now['nonresident_children'], matched['nonresident_children'], now_rate)['any other value']):.1f} pp "
        f"down. The raw training gap is +{raw_train:.1f} pp (men +{raw_by_sex['male']:.1f}, "
        f"women +{raw_by_sex['female']:.1f}); in the small test slice the raw gap vanishes "
        f"({raw_test:+.1f} pp), so the numbers compare similar people, not raw rates.",
        f"- **Enrollment separates now, not next.** Students and non-enrolled differ by "
        f"{swing(sv_now['enrolled'], matched['enrolled'], now_rate):.0f} pp in the "
        f"partnered-now model at matched ages but only "
        f"{swing(sv['enrolled'], rows['enrolled'], rate):.0f} pp here: it shifts *when* "
        f"people partner, not who finds someone - and at the test ages students are rare "
        f"({students:,} of {len(test):,} singles).",
        f"- **Outgoing people do better on both questions.** The low-to-high extraversion "
        f"gap is {swing(sv['big5_extraversion'], rows['big5_extraversion'], rate):.0f} pp "
        f"here and {swing(sv_now['big5_extraversion'], matched['big5_extraversion'], now_rate):.0f} pp "
        f"for being partnered - one of the few personality reads that survives both questions.",
    ]


def explain(model, data, sv, paths, shown=5000, seed=SEED):
    """Mean |SHAP| by sex with one direction line per top feature, plus a beeswarm per sex.

    Computed on the training rows (in-era, tens of thousands) so rare feature values are
    not read off a thin test slice; the beeswarm dots subsample to `shown` rows.
    """
    by_sex = pd.DataFrame({sex: sv.loc[rows.index].abs().mean()
                           for sex, rows in data.groupby("sex")})
    top = by_sex.sum(axis=1).sort_values(ascending=False).head(10).index
    base_rate = data[NEW_TARGET].mean()
    table = pd.DataFrame({
        "feature": top,
        "men: avg push": [f"{by_sex.loc[f, 'male']:.2f}" for f in top],
        "women: avg push": [f"{by_sex.loc[f, 'female']:.2f}" for f in top],
        "values with the strongest down vs up push (shift in the chance)":
            [direction(sv[f], data[f], base_rate) for f in top],
    })

    Xt = model[0].transform(data[FEATURES])
    for col in CATEGORICAL:  # the beeswarm colour scale needs numbers
        codes, _ = pd.factorize(Xt[col], sort=True)
        Xt[col] = np.where(codes >= 0, codes, np.nan)
    rng = np.random.default_rng(seed)
    shown = rng.choice(data.index, size=min(shown, len(data)), replace=False)
    for sex, path in paths.items():
        rows = shown[data.loc[shown, "sex"].to_numpy() == sex]
        shap.summary_plot(sv.loc[rows].to_numpy(), Xt.loc[rows], max_display=10, show=False)
        ax = plt.gca()
        ax.set_xlabel("effect on the model's log-odds of finding a partner "
                      "(left lowers the chance, right raises it)")
        ax.set_title(f"{'Men' if sex == 'male' else 'Women'}: what pushes the forecast "
                     f"({len(rows):,} of {sum(data['sex'] == sex):,} training interviews)")
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


def now_comparison(forecast_share, sv, rows, test):
    """Refit the 'partnered now' recipe, then compare which features each question rewards.

    The now side is restricted to the forecast test's age range, so a feature's size means
    the same thing on both sides (the now test set otherwise spans 18-43).
    """
    now_train, now_test = pd.read_parquet(NOW_TRAIN), pd.read_parquet(NOW_TEST)
    now_model = make_models()["catboost"].fit(now_train[FEATURES], now_train[TARGET])
    matched = now_test[now_test["age"].between(test["age"].min(), test["age"].max())]
    sv_now = shap_frame(now_model, matched[FEATURES])
    now_mean = sv_now.abs().mean()
    now_row = metrics(now_test[TARGET].to_numpy(),
                      now_model.predict_proba(now_test[FEATURES])[:, 1])

    fc_rate, now_rate = rows[NEW_TARGET].mean(), matched[TARGET].mean()
    shares = pd.DataFrame({"now": now_mean / now_mean.sum(),
                           "forecast": forecast_share / forecast_share.sum()})
    top = shares.max(axis=1).sort_values(ascending=False).head(12).index
    out = []
    for f in top:
        out.append({
            "feature": f,
            "'partnered now' share": f"{100 * shares.loc[f, 'now']:.0f}%",
            "its low vs high values, now": direction(sv_now[f], matched[f], now_rate),
            "'finds a partner' share": f"{100 * shares.loc[f, 'forecast']:.0f}%",
            "its low vs high values, finding": direction(sv[f], rows[f], fc_rate),
            "read": verdict(100 * shares.loc[f, "now"], 100 * shares.loc[f, "forecast"]),
        })
    return pd.DataFrame(out), now_row, sv_now, matched


def loss_gap(y, p_final, p_other, people, boots=2000, seed=SEED):
    """Final-minus-other log-loss gap (>0 means the other model scores better): mean,
    standard error, and a 95% interval from bootstrapping whole people."""
    def row_loss(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    gap = row_loss(p_final) - row_loss(p_other)
    per_person = (pd.DataFrame({"gap": gap, "person": people})
                  .groupby("person")["gap"].agg(["mean", "size"]))
    means, sizes = per_person["mean"].to_numpy(), per_person["size"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(per_person), size=(boots, len(per_person)))
    boot = (means[picks] * sizes[picks]).sum(axis=1) / sizes[picks].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return gap.mean(), gap.std(ddof=1) / np.sqrt(len(gap)), lo, hi


def main():
    base = build_rows(pd.read_parquet(CLEAN))
    train, test, unused = split_time(base)
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
    cv_summary = cv_table(oof, y, folds)

    print("fitting final models on all of train ...", flush=True)
    fitted = {name: models[name].fit(X, y) for name in MODEL_NAMES}
    final = fitted["catboost"]
    # The calibrator is Platt scaling fitted on out-of-fold predictions from the recent
    # rounds only: partnering rates fell over the decades, so a calibrator fitted on the
    # 2000s would speak the wrong era's language on the test years.
    recent = (train["round"] >= RECENT_ROUND).to_numpy()
    stacker = make_stacker().fit(oof["catboost"][recent].reshape(-1, 1), y[recent])
    stacker_weight = float(stacker[-1].coef_[0][0])

    y_test = test[NEW_TARGET].to_numpy()
    predictions = {
        "age + sex baseline": fitted["age_sex_baseline"].predict_proba(test[FEATURES])[:, 1],
        "logistic": fitted["logistic"].predict_proba(test[FEATURES])[:, 1],
        "catboost (raw)": final.predict_proba(test[FEATURES])[:, 1],
    }
    test = test.copy()
    test["p"] = stacker.predict_proba(
        predictions["catboost (raw)"].reshape(-1, 1))[:, 1]
    overall = [metrics_row(name, y_test, p) for name, p in predictions.items()]
    overall.append(metrics_row("catboost + calibration (final)", y_test, test["p"].to_numpy()))
    noise = loss_gap(y_test, test["p"].to_numpy(), predictions["logistic"], test["person_id"])

    by_sex = [metrics_row(sex, rows[NEW_TARGET].to_numpy(), rows["p"].to_numpy())
              for sex, rows in test.groupby("sex")]
    bands = pd.cut(test["age"], AGE_BANDS, right=False, labels=AGE_LABELS)
    by_age = [metrics_row(band, rows[NEW_TARGET].to_numpy(), rows["p"].to_numpy())
              for band, rows in test.groupby(bands, observed=True)]

    print("explaining with SHAP (train rows, in-era) ...", flush=True)
    sv = shap_frame(final, train[FEATURES])
    FIGURES.mkdir(parents=True, exist_ok=True)
    explain_table = explain(final, train, sv,
                            {"male": FIGURES / "forecast_shap_male.png",
                             "female": FIGURES / "forecast_shap_female.png"})
    calibration_plot(test, FIGURES / "forecast_calibration.png")
    rate_by_age_plot(test, FIGURES / "forecast_rate_by_age.png")

    compare_table, now_row, sv_now, matched = now_comparison(sv.abs().mean(), sv, train, test)
    findings = plain_findings(sv, train, sv_now, matched, test)

    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump({"features": FEATURES,
                 "categories": {c: sorted(train[c].dropna().unique()) for c in CATEGORICAL},
                 "base": {"catboost": final}, "stacker": stacker,
                 "target": NEW_TARGET}, MODEL_OUT)
    write_report(base, train, test, unused, cv_summary, pd.DataFrame(overall),
                 pd.DataFrame(by_sex), pd.DataFrame(by_age), explain_table, compare_table,
                 now_row, findings, noise, stacker_weight)
    print(REPORT.read_text())


def write_report(base, train, test, unused, cv_summary, overall, by_sex, by_age,
                 explain_table, compare_table, now_row, findings, noise, stacker_weight):
    n2 = int((base["follow_year"] - base["year"] == 2).sum())
    n3 = int((base["follow_year"] - base["year"] == 3).sum())
    unused_n = {k: len(v) for k, v in unused.items()}
    unused_total = sum(unused_n.values())
    never_people = unused["never"]["person_id"].nunique()
    older = base[(base["round"] < TEST_ROUND) & (base["age"] >= 35)]
    late_singles = set(base.loc[base["round"] >= TEST_ROUND, "person_id"])
    biased = older[~older["person_id"].isin(late_singles)][NEW_TARGET].mean()
    calib_n = int((train["round"] >= RECENT_ROUND).sum())
    calib_year = int(train.loc[train["round"] >= RECENT_ROUND, "year"].min())
    beyond = (test["age"] > train["age"].max()).mean()
    final_row = overall.iloc[-1]
    logistic_row = overall.iloc[1]
    floor_auc = overall.iloc[0]["roc_auc"]
    gap_mean, gap_se, gap_lo, gap_hi = noise
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
- **Set aside, unused:** {unused_total:,} of {len(base):,} rows ({unused_total / len(base):.0%})
  - {unused_n['late']:,} latest-round rows of training people (train stays in the earlier
  rounds), {unused_n['early']:,} earlier-round rows of the test people, and
  {unused_n['never']:,} rows of the {never_people:,} held-out people who never reach a
  test-round interview (people are held out before anyone knows who will still be single
  and interviewed in {test['year'].min()}-{test['year'].max()}). This is the price of a
  train set that is not stacked toward people who partnered in the end.

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

**Where the knobs come from** (so the test set is not quietly tuning them). TEST_ROUND is
{TEST_ROUND} because rounds {TEST_ROUND}-{TEST_ROUND + 2} are simply the latest interviews with
a possible follow-up; HELD_SHARE is {HELD_SHARE:.0%}, fixed before any test number was
computed, to keep the test big enough to read (it lands at {len(test):,} rows); RECENT_ROUND is {RECENT_ROUND}, the first
round with Big Five answers, so the calibrator sees the feature world of the test era. None
of the three was adjusted after seeing test numbers. One design choice did use the test era:
the split itself - the random-people variant was adopted after the drop-variant's
overprediction was measured on test-era rows (the {biased:.0%}-vs-{older[NEW_TARGET].mean():.0%}
story above). It was not picked for score, but it is test-era information influencing design,
disclosed here.

What the split costs, honestly: the NLSY97 cohort was born 1980-84, so the test interviews
only contain ages {test['age'].min()}-{test['age'].max()} - behavior for singles in their
20s is measured only by cross-validation on earlier data - and the training rounds contain
no single person older than {train['age'].max()} ({(train['age'] >= 35).sum():,} training
rows are 35+), while {beyond:.0%} of test rows are older. There, every model is
extrapolating past anything it has seen. And partnering rates fell over the decades, so the
calibrator is fitted on out-of-fold predictions from rounds {RECENT_ROUND}+ only
({calib_n:,} rows, interviews {calib_year} onward), the era closest to the test years; it
puts weight {stacker_weight:.2f} on CatBoost's log-odds. The calibration plot shows how
well that worked.

## Model comparison (cross-validation on train)

The "new people, same era" measure: each fold's people are disjoint, but the interview
years mix, so it shows how well the model reads people in a familiar world.
Mean ± sd over 5 folds grouped by person. Lower is better except ROC-AUC.

{markdown(cv_summary)}

## Test: the latest interviews ({test['year'].min()}-{test['year'].max()})

The "future era" stress test: every interview later than every training interview, and more
than half at ages training never saw. Expect it to score below cross-validation; that gap
is the price of time.

{markdown(overall)}

Two honest notes on this table. First, **logistic regression beats the final model** on
log loss, Brier and calibration ({logistic_row['log_loss']} vs {final_row['log_loss']} log
loss). The gap is inside noise: the average per-row log-loss difference is {gap_mean:.4f}
with standard error {gap_se:.4f}, and a person-level bootstrap 95% interval runs
[{gap_lo:+.4f}, {gap_hi:+.4f}] across zero. CatBoost stays final anyway, because the
cross-validation above - {len(train):,} rows, {len(train) / len(test):.0f}× the test set,
same-era - favours it on log loss and AUC, and because the SHAP explanation needs its
trees. Second, the age + sex floor scores below chance on AUC ({floor_auc}) for the
extrapolation reason above: at the ages it never saw it predicts the same average for
everyone, which ranks the 37+ test rows too high. Logistic, which bends a curve through age
instead of memorising cells, is the honest "guess by age and sex" here.

By sex and by age band (final model):

{markdown(by_sex)}

{markdown(by_age)}

Read the bands as rough: 30-34 is {int(by_age.iloc[0]['rows']):,} rows, and 35-41 is mostly
ages the training rounds never saw ({(test.loc[test['age'] >= 35, 'age'] > train['age'].max()).mean():.0%}
of its rows are older than {train['age'].max()}).

For scale: the same recipe predicting "partnered now" scores {now_row['roc_auc']:.3f} AUC on
its own held-out test set, versus {final_row['roc_auc']} here - the 2-year question is
genuinely harder, as expected for something with a lot of chance in it.

![calibration](figures/forecast_calibration.png)

![rate by age](figures/forecast_rate_by_age.png)

## What goes with finding a partner (SHAP)

Each row is one feature: its average push on the model (log-odds) for men and for women
separately, and the value groups with the strongest down and up push, in plain percentage
points (pp) around the training average of {train[NEW_TARGET].mean():.0%}. The pushes are
computed on the {len(train):,} training rows (same era, {len(train) / len(test):.0f}× the
test set) so rare feature values are not read off a thin slice; the test set above stays
for scoring only. Red dots in the figures are high feature values, blue are low
(a sample of 5,000 rows each).

{markdown(explain_table)}

![SHAP men](figures/forecast_shap_male.png)

![SHAP women](figures/forecast_shap_female.png)

## "Partnered now" vs "finds a partner": which features may run backwards

The "now" columns refit the partnered-now recipe and explain it the same way, restricted to
ages {test['age'].min()}-{test['age'].max()} so a feature's size means the same thing on both
sides; the "finds a partner" columns use the training rows. Shares are each feature's slice
of total importance. A feature big for "now" but small for "finding" mostly marks *having*
a partner, not *getting* one - often the partner is the cause (two incomes, shared
housing). These are associations, not proven causes.

{markdown(compare_table)}

## The short version

{findings_text}

## Limits

Associations, not causes - measuring features before the outcome removes the worst backwards
arrow, not all of them. One US cohort born 1980-84, and the test years cover ages
{test['age'].min()}-{test['age'].max()} only. Unweighted (survey weights not used; the
oversample of Black respondents also enlarges race's importance share), and people who left
the survey are absent. The target counts only marriage and living together: a girlfriend or
boyfriend not living together counts as "still single".
""")
    print(f"saved {REPORT}")


if __name__ == "__main__":
    main()
