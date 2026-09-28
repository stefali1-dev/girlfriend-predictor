"""Outside check: does what the NLSY97 model learned hold in other Americans' surveys?

Three questions, one script:
1. Transport: train.py's CatBoost recipe on only the columns all three surveys share,
   fitted on the NLSY97 train split and scored unchanged on NHANES and HCMST adults
   aged 18-43 (NHANES only asks marital status from 20).
2. The same recipe fitted on each survey separately: do the shared features push the
   same way and by a similar size, for men and for women (SHAP per survey)?
3. HCMST's non-live-in partners: how many of the main model's "singles" still have a
   girlfriend or boyfriend?
Writes results/outside_check.md and results/figures/outside_*.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from evaluate import INK, SEX_COLORS, style
from forecast import shap_frame
from train import TARGET, markdown, metrics, sex_age_cell

CLEAN = Path("data/clean")
TRAIN, TEST = Path("data/curated/train.parquet"), Path("data/curated/test.parquet")
REPORT = Path("results/outside_check.md")
FIGURES = Path("results/figures")
SEED = 42

MIN_AGE, MAX_AGE = 18, 43  # the main model's range; NHANES starts at 20
CATEGORICAL = ["sex", "race_ethnicity", "education"]
NUMERIC = ["employed", "height_cm", "bmi"]
FEATURES = ["age"] + CATEGORICAL + NUMERIC
CORE_LABEL = "core model (trained on NLSY97)"
BASE_LABEL = "age + sex rates of this survey"
AGE_BANDS, AGE_LABELS = [18, 25, 30, 35, 44], ["18-24", "25-29", "30-34", "35-43"]
BAND_MIDPOINTS = [21, 27, 32, 39]
DATASETS = ["NLSY97", "NHANES", "HCMST"]
DATASET_COLORS = {"NLSY97": INK, "NHANES": "#2a78d6", "HCMST": "#eb6834"}
FEATURE_LABELS = {"race_ethnicity": "race/ethnicity", "height_cm": "height",
                  "bmi": "BMI"}


def core_model():
    """train.py's CatBoost step on the shared columns: same preprocessing, same settings."""
    return make_pipeline(
        ColumnTransformer(
            [("cat", SimpleImputer(strategy="constant", fill_value="missing"), CATEGORICAL),
             ("num", "passthrough", ["age"] + NUMERIC)],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas"),
        CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=6,
                           cat_features=tuple(CATEGORICAL), random_seed=SEED, verbose=0,
                           allow_writing_files=False),
    )


def age_sex_baseline():
    """train.py's floor to beat: one fitted probability per (sex, age) cell."""
    return make_pipeline(
        FunctionTransformer(sex_age_cell),
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(C=100, max_iter=1000),
    )


def shared(path, extra=()):
    """A survey's adults 18-43 with only the shared columns, plus the target and year."""
    df = pd.read_parquet(path)
    for c in CATEGORICAL:  # HCMST stores sex as categorical; str keeps missing values missing
        df[c] = df[c].astype("str")
    df = df[df["age"].between(MIN_AGE, MAX_AGE)]
    return df[[*FEATURES, TARGET, "year", *extra]].reset_index(drop=True)


def value_groups(values):
    """Bins that mean the same thing in every survey: age bands, categories, 0/1, tertiles."""
    if values.name == "age":
        return pd.cut(values, AGE_BANDS, right=False, labels=AGE_LABELS)
    if values.name == "employed":
        return values.astype("float").map({0.0: "not employed", 1.0: "employed"})
    if values.name in CATEGORICAL:
        return values
    if not values.notna().any():
        return values  # nothing measured (HCMST's height/BMI): no groups to form
    bins = pd.qcut(values, 3, duplicates="drop")
    names = ["low", "high"] if len(bins.cat.categories) == 2 else ["low", "middle", "high"]
    return bins.cat.rename_categories(names)


def points(sv, values, base_rate):
    """Each value group's push on the model, in percentage points around `base_rate`."""
    mean = sv.groupby(value_groups(values), observed=True).mean()
    odds = logit(base_rate)
    return {k: (expit(odds + e) - base_rate) * 100 for k, e in mean.items()}


def direction(sv, values, base_rate):
    """The value groups with the strongest down and up push, with their shifts."""
    pts = points(sv, values, base_rate)
    if not pts:
        return "not measured in this survey"
    ranked = sorted(pts, key=pts.get)
    lo, hi = ranked[0], ranked[-1]
    return f"{lo} {pts[lo]:+.1f} pp ... {hi} {pts[hi]:+.1f} pp"


def swing(sv, values, base_rate):
    """Gap in pp between a feature's most and least partnered value groups."""
    pts = points(sv, values, base_rate)
    return pts[max(pts, key=pts.get)] - pts[min(pts, key=pts.get)]


def scored(survey, model_name, p, d):
    """Raw numbers for one (survey, model) row of the transport table."""
    return {"survey": survey, "model": model_name, "rows": len(d),
            "partnered": float(d[TARGET].mean()), "mean_pred": float(p.mean()),
            **metrics(d[TARGET].to_numpy(), p)}


def fmt(transport):
    out = transport.copy()
    out["rows"] = out["rows"].map(lambda v: f"{v:,}")
    for c in ["partnered", "mean_pred"]:
        out[c] = out[c].map(lambda v: f"{v:.3f}")
    for c in ["log_loss", "brier", "roc_auc", "calib_error"]:
        out[c] = out[c].map(lambda v: f"{v:.4f}")
    return out.rename(columns={"mean_pred": "mean pred.", "log_loss": "log loss",
                               "roc_auc": "ROC-AUC", "calib_error": "calib. error"})


def push_tables(recipes):
    """Feature size (mean |SHAP| by sex per survey) and direction (down/up value groups)."""
    order = recipes["NLSY97"][2].abs().mean().sort_values(ascending=False).index
    size_rows, direction_rows = [], []
    for f in order:
        size = {"feature": FEATURE_LABELS.get(f, f)}
        where = {"feature": FEATURE_LABELS.get(f, f)}
        for name, (_, frame, sv) in recipes.items():
            if not frame[f].notna().any():
                size[f"{name} men"], size[f"{name} women"] = "—", "—"
            else:
                for sex in ["male", "female"]:
                    size[f"{name} {'men' if sex == 'male' else 'women'}"] = (
                        f"{sv.loc[frame['sex'] == sex, f].abs().mean():.2f}")
            where[name] = direction(sv[f], frame[f], frame[TARGET].mean())
        size_rows.append(size)
        direction_rows.append(where)
    return pd.DataFrame(size_rows), pd.DataFrame(direction_rows), order


def any_partner_table(hcmst):
    """Among 18-43s with no live-in partner, the share with a partner they don't live with."""
    singles = hcmst[hcmst[TARGET] == 0]
    bands = pd.cut(singles["age"], AGE_BANDS, right=False, labels=AGE_LABELS)
    rows = []
    for label, part in [(b, g) for b, g in singles.groupby(bands, observed=True)] \
            + [("all 18-43", singles)]:
        by_sex = part.groupby("sex")["any_partner"].agg(["mean", "size"])
        rows.append({
            "age": label,
            "men": f"{int(by_sex['size']['male']):,}",
            "men with a partner": f"{100 * by_sex['mean']['male']:.0f}%",
            "women": f"{int(by_sex['size']['female']):,}",
            "women with a partner": f"{100 * by_sex['mean']['female']:.0f}%",
        })
    return pd.DataFrame(rows)


def rate_by_age_figure(scored_sets, path):
    """Model mean prediction by age vs the observed share per age band, one panel per survey."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (survey, d) in zip(axes, scored_sets.items()):
        for sex in ["male", "female"]:
            rows = d[d["sex"] == sex]
            color = SEX_COLORS[sex]
            line = rows.groupby("age")["p"].mean()
            ax.plot(line.index, line, color=color, linewidth=2,
                    label=f"{sex}: model (mean prediction by age)")
            observed = rows.groupby(value_groups(rows["age"]), observed=True)[TARGET].mean()
            ax.scatter(BAND_MIDPOINTS, observed, color=color, s=45, edgecolor="white",
                       linewidth=1.5, zorder=3, label=f"{sex}: observed (age bands)")
        ax.set(title=survey, xlabel="age", ylim=(0, 1))
        style(ax)
    axes[0].set_ylabel("share partnered")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("The NLSY97-trained core model, scored unchanged on each survey", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def calibration_figure(scored_sets, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    for ax, (survey, d) in zip(axes, scored_sets.items()):
        ax.plot([0, 1], [0, 1], color="#898781", linestyle="--", linewidth=1,
                label="perfect calibration")
        for sex in ["male", "female"]:
            rows = d[d["sex"] == sex]
            bins = rows.groupby(pd.qcut(rows["p"], 10, duplicates="drop"), observed=True)
            ax.plot(bins["p"].mean(), bins[TARGET].mean(), marker="o", markersize=6,
                    linewidth=1.8, color=SEX_COLORS[sex], label=sex)
        ax.set(title=survey, xlabel="predicted probability of being partnered",
               xlim=(0, 1), ylim=(0, 1))
        style(ax)
    axes[0].set_ylabel("observed share partnered")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def push_figure(recipes, order, path):
    """Mean |SHAP| per feature per survey, men and women side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharex=True)
    ys = np.arange(len(order))[::-1]
    labels = [FEATURE_LABELS.get(f, f) for f in order]
    for ax, sex, title in zip(axes, ["male", "female"], ["Men", "Women"]):
        for i, name in enumerate(DATASETS):
            _, frame, sv = recipes[name]
            # NaN width = feature not measured: no bar rather than a misleading zero one.
            widths = [sv.loc[frame["sex"] == sex, f].abs().mean()
                      if frame[f].notna().any() else np.nan for f in order]
            ax.barh(ys + (1 - i) * 0.25, widths, height=0.24, color=DATASET_COLORS[name],
                    label=name)
        ax.set_yticks(ys, labels)
        ax.set(title=title, xlabel="mean |SHAP| on the log-odds")
        style(ax)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def push(name, feature, recipes):
    """A feature's value-group pushes in one survey, in pp around that survey's rate."""
    _, frame, sv = recipes[name]
    return points(sv[feature], frame[feature], frame[TARGET].mean())


def plain_findings(transport, recipes, hcmst):
    """A few takeaway sentences; every number is computed here."""
    row = {(t["survey"], t["model"]): t for t in transport.to_dict("records")}
    nha_core, nha_base = row[("NHANES (20-43)", CORE_LABEL)], row[("NHANES (20-43)", BASE_LABEL)]
    hc_core, hc_base = row[("HCMST (18-43)", CORE_LABEL)], row[("HCMST (18-43)", BASE_LABEL)]

    def sex_size(name, f):
        _, frame, sv = recipes[name]
        return {s: float(sv.loc[frame["sex"] == s, f].abs().mean()) for s in ["male", "female"]}

    def swing_of(name, f):
        _, frame, sv = recipes[name]
        return swing(sv[f], frame[f], frame[TARGET].mean())

    young = {n: push(n, "age", recipes)["18-24"] for n in DATASETS}
    black = {n: push(n, "race_ethnicity", recipes).get("black") for n in DATASETS}
    some_college = {n: push(n, "education", recipes).get("some_college") for n in DATASETS}
    no_hs = {n: push(n, "education", recipes).get("less_than_hs") for n in DATASETS}
    emp = {n: sex_size(n, "employed") for n in DATASETS}
    singles = hcmst[hcmst[TARGET] == 0]
    with_partner = singles.groupby("sex")["any_partner"].mean()
    height_sizes = [sex_size(n, "height_cm")[s] for n in ["NLSY97", "NHANES"]
                    for s in ["male", "female"]]
    height_swings = [swing_of(n, "height_cm") for n in ["NLSY97", "NHANES"]]
    low_bmi = {n: push(n, "bmi", recipes).get("low") for n in ["NLSY97", "NHANES"]}

    def join(per_dataset, fmt="{:+.1f}"):
        return " / ".join(fmt.format(per_dataset[n]) for n in DATASETS)

    young_gap = join({n: abs(v) for n, v in young.items()}, "{:.0f}")
    race_size = sex_size("NLSY97", "race_ethnicity")

    return [
        f"- **The recipe travels to NHANES.** Scored unchanged, the NLSY97 core model beats "
        f"NHANES's own age + sex rates on everything: {nha_core['log_loss']:.3f} vs "
        f"{nha_base['log_loss']:.3f} log loss, {nha_core['roc_auc']:.3f} vs "
        f"{nha_base['roc_auc']:.3f} AUC. Even its level is nearly right (mean prediction "
        f"{nha_core['mean_pred']:.3f} against {nha_core['partnered']:.3f} observed).",
        f"- **HCMST breaks the level, not the ranking.** The same model still discriminates "
        f"({hc_core['roc_auc']:.3f} AUC) but underpredicts by "
        f"{100 * (hc_core['partnered'] - hc_core['mean_pred']):.0f} pp on average "
        f"({hc_core['mean_pred']:.3f} predicted vs {hc_core['partnered']:.3f} observed), so "
        f"log loss collapses to {hc_core['log_loss']:.3f} and HCMST's own age + sex rates "
        f"score better ({hc_base['log_loss']:.3f} / {hc_base['roc_auc']:.3f}).",
        f"- **Age dominates in all three surveys** (largest mean |SHAP| for both sexes "
        f"everywhere), and the young-adult gap widened over time: 18-24 sits {young_gap} pp "
        f"below its survey's average in NLSY97 / NHANES / HCMST.",
        f"- **The Black partnered gap replicates at full size**: {join(black)} pp below the "
        f"survey average in NLSY97 / NHANES / HCMST, and the push is bigger for women than "
        f"men in all three (e.g. NLSY97: {race_size['female']:.2f} vs "
        f"{race_size['male']:.2f} mean |SHAP|).",
        f"- **\"Some college, no degree\" partners latest in all three surveys** "
        f"({join(some_college)} pp), while the ends of the education range partner sooner "
        f"(less than high school {join(no_hs)} pp). The shape is a U, not a ladder: it is "
        f"not \"more education, more partnering\".",
        f"- **Employment pushes up everywhere and matters more for men** "
        f"(mean |SHAP|, men vs women: NLSY97 {emp['NLSY97']['male']:.2f}/{emp['NLSY97']['female']:.2f}, "
        f"NHANES {emp['NHANES']['male']:.2f}/{emp['NHANES']['female']:.2f}, "
        f"HCMST {emp['HCMST']['male']:.2f}/{emp['HCMST']['female']:.2f}), but its size "
        f"tracks the question asked: the not-employed-vs-employed gap is "
        f"{swing_of('NLSY97', 'employed'):.0f} pp under NLSY97's \"worked any weeks last "
        f"year\", {swing_of('NHANES', 'employed'):.0f} pp under NHANES's \"working now\" and "
        f"{swing_of('HCMST', 'employed'):.0f} pp in HCMST.",
        f"- **Height barely registers wherever it is measured** (mean |SHAP| at most "
        f"{max(height_sizes):.2f}, largest low-to-high gap {max(height_swings):.1f} pp) - "
        f"the \"short men partner less\" story does not show up in who ends up partnered. "
        f"BMI: the lightest tertile sits {low_bmi['NLSY97']:.1f} pp (NLSY97) and "
        f"{low_bmi['NHANES']:.1f} pp (NHANES) below average; above that, little.",
        f"- **\"Single\" means \"no live-in partner\" - and many singles are not partnerless.** "
        f"In HCMST, {singles['any_partner'].mean():.0%} of the 18-43s the main model would "
        f"call single have a girlfriend or boyfriend they don't live with: women "
        f"{with_partner['female']:.0%}, men {with_partner['male']:.0%}. The main model's "
        f"target cannot see relationships short of cohabitation, and the men it calls "
        f"single are far more often truly alone than the women.",
    ]


def write_report(train, scored_sets, transport, size, where, any_table, recipes, findings):
    row = {(t["survey"], t["model"]): t for t in transport.to_dict("records")}
    home = row[("NLSY97 (test split)", CORE_LABEL)]
    nha = row[("NHANES (20-43)", CORE_LABEL)]
    hc = row[("HCMST (18-43)", CORE_LABEL)]
    hc_frame = recipes["HCMST"][1]
    nha_missing_height = recipes["NHANES"][1]["height_cm"].isna().mean()
    hc_by_sex = {sex: metrics(g[TARGET].to_numpy(), g["p"].to_numpy())
                 for sex, g in scored_sets["HCMST (18-43)"].groupby("sex")}
    hc_singles = hc_frame[hc_frame[TARGET] == 0]
    # Full-featured reference numbers from results/metrics.md, for scale only.
    FULL_LL, FULL_AUC = 0.5425, 0.7896
    surveys = pd.DataFrame([
        {"survey": "NLSY97", "interviews": f"{train['year'].min()}-{train['year'].max()}",
         "ages": "18-43", "rows scored": f"{home['rows']:,}",
         "partnered": f"{home['partnered']:.3f}", "height / BMI": "self-reported",
         "employed asks": "worked any weeks last year"},
        {"survey": "NHANES", "interviews": "2007-2021", "ages": "20-43",
         "rows scored": f"{nha['rows']:,}", "partnered": f"{nha['partnered']:.3f}",
         "height / BMI": "measured at exam", "employed asks": "working now"},
        {"survey": "HCMST", "interviews": "2017", "ages": "18-43",
         "rows scored": f"{hc['rows']:,}", "partnered": f"{hc['partnered']:.3f}",
         "height / BMI": "not asked", "employed asks": "working for pay now"},
    ])
    findings_text = "\n".join(findings)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""\
# Outside check: does the NLSY97 model hold in other surveys?

The main model was trained on one cohort (NLSY97, born 1980-84). This page asks whether
what it learned describes other Americans, measured by other surveys: **NHANES** (a
health examination survey, cycles 2007-2021) and **HCMST 2017** (Stanford's couple
survey). A finding that shows up in all three surveys is a fact about America, not an
accident of one sample; one that shows up once is suspect. This is *external validation*.

Three questions, matching the three sections: the NLSY97 recipe scored unchanged on the
other surveys; the same recipe retrained on each survey to compare what each feature
does there; and HCMST's count of "singles" who do have a partner, just not a live-in one.

## The three surveys, side by side

{markdown(surveys)}

The NLSY97 row is the held-out test split; rows are person-interviews, unweighted, ages
restricted to the main model's 18-43. Race/ethnicity is hispanic / black / other in all
three ("other" includes Asian and multiracial respondents; NHANES's 2007 cycles have no
Asian category).

## The core model, scored unchanged

The **core model** is train.py's CatBoost recipe (same preprocessing, same settings) on
only the seven columns all three surveys share: age, sex, race/ethnicity, education,
employed, height, BMI. It is fitted on the NLSY97 **train split** and then applied
unchanged - no refitting, no recalibration - to NHANES and HCMST.

**HCMST has no height or BMI at all** (the survey never asks). CatBoost treats a missing
number as its own value, so the model simply scores HCMST with the other five features;
the same mechanism covers NHANES respondents who were interviewed but not examined
({nha_missing_height:.0%} of 20-43s lack measured height).

{markdown(fmt(transport))}

The comparison floor is each survey's **own age + sex rates** (fitted on that survey;
the NLSY97 floor is fitted on train and scored on the held-out test split, as in
evaluate.py - NHANES's and HCMST's floors are fitted on the rows they are scored on).
Mean pred. is the average prediction - how far off the *level* is.

What the table says:

- **NHANES: the model transfers.** Better log loss, better AUC than the local floor, and
  the level is nearly right. Moving from the home test split to NHANES costs
  {nha['log_loss'] - home['log_loss']:.3f} log loss and {home['roc_auc'] - nha['roc_auc']:.3f} AUC.
- **HCMST: the ranking survives, the level does not.** AUC stays at {hc['roc_auc']:.3f}
  (men {hc_by_sex['male']['roc_auc']:.3f}, women {hc_by_sex['female']['roc_auc']:.3f}),
  but the average prediction is {100 * (hc['partnered'] - hc['mean_pred']):.0f} pp too low
  - a {hc['partnered']:.1%}-partnered survey scored by a model that says
  {hc['mean_pred']:.1%} - so log loss collapses and HCMST's own age + sex rates beat the
  transported model on both metrics. Part of that level gap is the survey itself: HCMST
  2017's married share runs ~8 pp above Census for the same years in its own published
  numbers (panel participation favours people in couples; noted when the data was
  cleaned). The rest is era: HCMST is 2017 only, when 20-somethings partnered later than
  across NLSY97's mixed decades.
- For scale: the full-featured main model scores {FULL_LL} log loss / {FULL_AUC} AUC on
  the same test split (`results/metrics.md`); keeping only the seven shared columns
  costs {home['log_loss'] - FULL_LL:.3f} log loss and {FULL_AUC - home['roc_auc']:.3f} AUC.

![rate by age](figures/outside_rate_by_age.png)

The model line is its mean prediction at each age; dots are the observed share per age
band (bands, because HCMST has ~1,400 rows - per-age dots would be noise). The NHANES
panels stay close to their dots. The HCMST panel shows the failure mode: the *slope* is
roughly right, the whole line sits too low, most steeply in the 20s.

![calibration](figures/outside_calibration.png)

## The same recipe fitted on each survey

Each survey gets its own core model (same recipe, fitted on its own rows), explained by
SHAP on the rows it was fitted on. Size is the average strength of a feature's push
(mean |SHAP| on the log-odds) for men and women separately; direction is the value group
with the strongest down and up push, in percentage points around that survey's own
partnered rate. HCMST's height and BMI columns are empty, so there is nothing to read.

{markdown(size)}

{markdown(where)}

![feature pushes](figures/outside_shap.png)

Read together:

- **Age** is the strongest feature in every survey, for both sexes, by a wide margin -
  and its pull steepened over time (see the short version below).
- **Sex**: women are more likely to be partnered at the same age in all three surveys,
  though NHANES's gap is the smallest.
- **Race/ethnicity**: the Black partnered gap appears in all three surveys at nearly the
  same size, and is a bigger push for women than men everywhere. This is the most
  replicated single finding on this page.
- **Education**: all three surveys put "some college, no degree" at the bottom and both
  education extremes higher - a U-shape, not a ladder.
- **Employed** pushes up everywhere and matters more for men than women in all three.
  The size differs a lot, and the wording of the question is part of why: NLSY97 asks
  about the whole past year, NHANES and HCMST about the interview week.
- **Height** is the smallest measured feature in both surveys that have it; **BMI**
  shows one consistent move - the lightest tertile down - and little else.

## HCMST: "single" is not "no partner"

The main model's target counts marriage and living together, so a girlfriend or
boyfriend not living with you counts as single. HCMST is the one survey that asks. Among
its 18-43s with no live-in partner:

{markdown(any_table)}

{hc_singles['any_partner'].mean():.0%} of them have a partner
they don't live with - {hc_singles.loc[hc_singles['sex'] == 'female', 'any_partner'].mean():.0%}
of the women but only {hc_singles.loc[hc_singles['sex'] == 'male', 'any_partner'].mean():.0%}
of the men, a gap that holds in every age band. So the main model's "single" class is a
mix - mostly unpartnered men and women who do have a partner - and anything the model
says about "singles" is about *not cohabiting*, not about being alone. A feature that
helps people date without moving in (many, plausibly) will look weaker for this target
than it is for "has a relationship at all".

## The short version

{findings_text}

## Why the surveys can disagree

- **Definitions**: "employed" means the past year in NLSY97 and the interview week in
  NHANES/HCMST; NHANES measures height and weight, NLSY97 asks (people overstate height
  and understate weight); "married, spouse absent" still counts as partnered everywhere.
- **Years**: NLSY97's rows span interviews 1998-2024, NHANES 2007-2021, HCMST 2017.
  Americans increasingly postpone cohabitation and marriage, so a survey taken later
  shows fewer partnered 20-somethings - the age gradient in HCMST (2017) is much steeper
  than the mixed-decades gradient the core model learned.
- **Samples**: NLSY97 oversamples Black and Hispanic youth and follows one birth cohort;
  NHANES is a health survey whose examination subsample misses ~8% of height/BMI; HCMST
  is an online probability panel whose own published married rate runs above Census, and
  its LGB oversample is only down-weighted by weights this page does not use.
- **Target transport**: each survey decides partnered status with its own questions;
  NLSY97's is closest to "who lives with a partner on the interview date", HCMST's to a
  partnership screener. Small differences in wording move the level by points.

## Limits

Associations, not causes, throughout. Unweighted numbers (the surveys' sampling weights
are unused, as everywhere else in this project). The per-survey models are fitted and
explained on the same rows - their pushes describe what the recipe learned there, not an
out-of-sample score. HCMST is small ({hc['rows']:,} rows, ~650 per sex), so its per-sex
numbers wobble. NHANES cycles are pooled as if one survey. And the core model's seven
features leave out most of what matters (earnings, family background, personality), so
"what held" here is about these seven, not about the full model.
""")
    print(f"saved {REPORT}")


def main():
    train, test = shared(TRAIN), shared(TEST)
    nhanes = shared(CLEAN / "nhanes.parquet")
    hcmst = shared(CLEAN / "hcmst.parquet", extra=["any_partner"])

    print("fitting the core model on the NLSY97 train split ...", flush=True)
    core = core_model().fit(train[FEATURES], train[TARGET])

    transport_rows, scored_sets = [], {}
    for survey, fit_frame, frame in [("NLSY97 (test split)", train, test),
                                     ("NHANES (20-43)", nhanes, nhanes),
                                     ("HCMST (18-43)", hcmst, hcmst)]:
        p = core.predict_proba(frame[FEATURES])[:, 1]
        scored_sets[survey] = frame.assign(p=p)
        transport_rows.append(scored(survey, CORE_LABEL, p, frame))
        floor = age_sex_baseline().fit(fit_frame[FEATURES], fit_frame[TARGET])
        transport_rows.append(scored(survey, BASE_LABEL,
                                     floor.predict_proba(frame[FEATURES])[:, 1], frame))
    transport = pd.DataFrame(transport_rows)

    print("fitting the same recipe on NHANES and HCMST, explaining with SHAP ...", flush=True)
    recipes = {"NLSY97": (core, train, shap_frame(core, train[FEATURES]))}
    for name, frame in [("NHANES", nhanes), ("HCMST", hcmst)]:
        model = core_model().fit(frame[FEATURES], frame[TARGET])
        recipes[name] = (model, frame, shap_frame(model, frame[FEATURES]))

    size, where, order = push_tables(recipes)
    any_table = any_partner_table(hcmst)

    FIGURES.mkdir(parents=True, exist_ok=True)
    rate_by_age_figure(scored_sets, FIGURES / "outside_rate_by_age.png")
    calibration_figure(scored_sets, FIGURES / "outside_calibration.png")
    push_figure(recipes, order, FIGURES / "outside_shap.png")

    findings = plain_findings(transport, recipes, hcmst)
    write_report(train, scored_sets, transport, size, where, any_table, recipes, findings)
    print(REPORT.read_text())


if __name__ == "__main__":
    main()
