"""What goes with being partnered, for men and for women, according to the final model.

Reads the train split and models/partnered.joblib (run split.py and train.py first).
Writes results/figures/{importance,curves_numeric,curves_categorical,interactions}.png and
results/explain_numbers.md, the source of every number in results/insights.md.
Train set only: the test set was opened once for scoring and stays closed.

Stability: the CatBoost pipeline is refitted on each of the 5 cross-validation training folds
(the same folds as train.py). Every effect is computed with the final model and with those 5
fold models; an effect counts as stable only when all 5 agree on its direction.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool
from scipy.special import expit
from sklearn.model_selection import GroupKFold

from evaluate import INK, MUTED, SEX_COLORS, style
from predict import MODEL, predict_proba
from train import CATEGORICAL, FEATURES, SEED, TARGET, TRAIN, make_models, markdown

OUT = Path("results/explain_numbers.md")
FIGURES = Path("results/figures")
SEXES = ["male", "female"]
SAMPLE = 2000  # rows per sex behind each "change one thing" average
MIN_EFFECT = 0.01  # below 1 percentage point an effect is reported as "small"
AGE_BANDS = [18, 25, 30, 35, 44]
AGE_LABELS = ["18-24", "25-29", "30-34", "35-43"]

# "From → to" values for each numeric feature, chosen as ordinary values a reader can picture.
# Height gets its own values per sex (men are ~13 cm taller).
CONTRASTS = {
    "age": (22, 32),
    "earnings": (20_000, 60_000),
    "weeks_worked": (0, 52),
    "hours_worked": (1_000, 2_080),
    "height_cm": {"male": (170, 185), "female": (157, 170)},
    "bmi": (22, 32),
    "general_health": (1, 4),  # excellent → fair
    "asvab_percentile": (25, 75),
    "mother_educ_grade": (12, 16),
    "father_educ_grade": (12, 16),
    "lived_with_both_parents_age12": (0, 1),
    "family_income_1997": (25_000, 75_000),
    "urban": (0, 1),
    "attendance_worship": (1, 6),  # never → about once a week
    "importance_faith": (5, 1),  # not at all important → extremely important
    "big5_extraversion": (3, 6),
    "big5_agreeableness": (3, 6),
    "big5_conscientiousness": (3, 6),
    "big5_emotional_stability": (3, 6),
    "big5_openness": (3, 6),
    "nonresident_children": (0, 1),
}
NUMERIC = list(CONTRASTS)

# Named changes (from, to) for the by-age table and the same-person examples. Having a job is
# one change of weeks, hours and pay together: moving weeks alone while pay stays put is not a
# person who exists.
CHANGES = {
    "earnings $20k → $60k": ({"earnings": 20_000}, {"earnings": 60_000}),
    "no job → full-time all year at $40k": (
        {"weeks_worked": 0, "hours_worked": 0, "earnings": 0},
        {"weeks_worked": 52, "hours_worked": 2_080, "earnings": 40_000}),
    "high school → bachelor's": ({"education": "hs"}, {"education": "bachelor_plus"}),
    "worship never → weekly": ({"attendance_worship": 1}, {"attendance_worship": 6}),
    "no → one child living elsewhere": ({"nonresident_children": 0}, {"nonresident_children": 1}),
    "height 170 → 185 cm": ({"height_cm": 170}, {"height_cm": 185}),
    "BMI 22 → 32": ({"bmi": 22}, {"bmi": 32}),
}
BY_AGE = list(CHANGES)[:5]
EXAMPLES = [  # "same person, one thing changed": (sex, age, change)
    ("male", 30, "earnings $20k → $60k"),
    ("female", 30, "earnings $20k → $60k"),
    ("male", 30, "no job → full-time all year at $40k"),
    ("female", 30, "no job → full-time all year at $40k"),
    ("male", 30, "high school → bachelor's"),
    ("female", 30, "high school → bachelor's"),
    ("male", 30, "height 170 → 185 cm"),
    ("female", 30, "BMI 22 → 32"),
    ("male", 28, "worship never → weekly"),
    ("female", 28, "worship never → weekly"),
    ("male", 30, "no → one child living elsewhere"),
    ("female", 30, "no → one child living elsewhere"),
]


def fold_models(model, X, y, groups):
    """The final CatBoost recipe refitted on each CV training fold, packed like the final model."""
    folds = GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, groups=groups)
    fitted = []
    for i, (fit_rows, val_rows) in enumerate(folds, 1):
        print(f"refitting catboost on fold {i}/5 ...", flush=True)
        pipe = make_models()["catboost"].fit(X.iloc[fit_rows], y[fit_rows])
        fitted.append(({**model, "base": {"catboost": pipe}}, val_rows))
    return fitted


def shap_push(model, rows):
    """How many percentage points each feature moves each row's probability (TreeSHAP).

    SHAP splits the model's log-odds for a row into one share per feature. Each share is
    turned into probability points: the prediction minus the prediction with that share removed.
    """
    pipe = model["base"]["catboost"]
    Xt = pipe[:-1].transform(rows[FEATURES])
    shap = pipe[-1].get_feature_importance(Pool(Xt, cat_features=CATEGORICAL), type="ShapValues")
    values, log_odds = shap[:, :-1], shap.sum(axis=1)

    def calibrated(z):
        return model["stacker"].predict_proba(expit(z).reshape(-1, 1))[:, 1]

    p = calibrated(log_odds)
    push = np.column_stack([p - calibrated(log_odds - values[:, j]) for j in range(values.shape[1])])
    return pd.DataFrame(push, columns=Xt.columns, index=rows.index)


def importance(model, fitted, train):
    push = shap_push(model, train).abs()
    table = pd.DataFrame({"all": push.mean(), **{sex: push[train["sex"] == sex].mean() for sex in SEXES}})
    # Rank of each feature in each fold model, scored on the rows that model did not train on.
    ranks = pd.DataFrame({i: shap_push(m, train.iloc[val]).abs().mean().rank(ascending=False)
                          for i, (m, val) in enumerate(fitted)})
    table["rank"] = table["all"].rank(ascending=False).astype(int)
    table["fold ranks"] = ranks.min(axis=1).astype(int).astype(str) + "-" + ranks.max(axis=1).astype(int).astype(str)
    return table.sort_values("all", ascending=False)


def sample(train, sex, features, rng, extra=None):
    """Up to SAMPLE real rows of one sex where the features are known (answered in that round)."""
    keep = (train["sex"] == sex) & train[features].notna().all(axis=1)
    if extra is not None:
        keep &= extra
    rows = train[keep]
    return rows.sample(min(SAMPLE, len(rows)), random_state=rng.integers(1 << 31))


def curve(model, rows, settings):
    """Mean probability when every row gets each setting ({feature: value}), all else unchanged."""
    stacked = pd.concat([rows.assign(**setting) for setting in settings])
    return predict_proba(model, stacked).reshape(len(settings), len(rows)).mean(axis=1)


def curves(models, rows, settings):
    """One curve per model: row 0 the final model, rows 1-5 the fold models."""
    return np.array([curve(m, rows, settings) for m in models])


def along(feature, values):
    return [{feature: v} for v in values]


def grid_for(train, feature):
    if feature in CATEGORICAL:
        return sorted(train[feature].dropna().unique())
    known = train[feature].dropna()
    known = known[known <= known.quantile(0.995)]  # e.g. no curve out to 12 children
    if known.nunique() <= 30:
        return sorted(known.unique())
    return np.unique(np.quantile(known, np.linspace(0.05, 0.95, 12)).round(1))


def verdict(final, folds):
    """Stable when every fold model agrees with the final model's direction; small below 1 point."""
    if abs(final) < MIN_EFFECT:
        return "small"
    return "stable" if (np.sign(folds) == np.sign(final)).all() else "unstable"


def effect_cell(c):
    return f"{100 * c[0]:+.1f} ({100 * c[1:].min():+.1f} to {100 * c[1:].max():+.1f}) {verdict(c[0], c[1:])}"


def main():
    train = pd.read_parquet(TRAIN)
    model = joblib.load(MODEL)
    fitted = fold_models(model, train[FEATURES], train[TARGET].to_numpy(), train["person_id"])
    models = [model] + [m for m, _ in fitted]
    rng = np.random.default_rng(SEED)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sections = []

    base_rates = train.groupby(["sex", pd.cut(train["age"], AGE_BANDS, right=False, labels=AGE_LABELS)],
                               observed=True)[TARGET].mean().unstack().round(3).reset_index()
    sections.append("## Partnered share in the train set, by sex and age\n\n" + markdown(base_rates))

    print("SHAP importance ...", flush=True)
    imp = importance(model, fitted, train)
    imp_table = imp.reset_index(names="feature")
    for col in ["all", *SEXES]:
        imp_table[col] = (100 * imp_table[col]).round(1)
    sections.append(
        "## Importance: average push on a person's probability (percentage points, TreeSHAP)\n\n"
        "How far, on average, each feature moves one person's predicted probability up or down "
        "from the average prediction. `rank` in the final model; `fold ranks` the range of its "
        "rank across the 5 fold models.\n\n" + markdown(imp_table))
    importance_plot(imp, FIGURES / "importance.png")

    print("change-one-thing curves ...", flush=True)
    effects, numeric_curves = [], {}
    for feature in NUMERIC:
        grid = grid_for(train, feature)
        row = {"feature": feature}
        for sex in SEXES:
            rows = sample(train, sex, [feature], rng)
            lo, hi = CONTRASTS[feature][sex] if isinstance(CONTRASTS[feature], dict) else CONTRASTS[feature]
            numeric_curves[feature, sex] = (grid, curves(models, rows, along(feature, grid)))
            ends = curves(models, rows, along(feature, [lo, hi]))
            row[f"{sex}: from → to"] = f"{lo:,} → {hi:,}"
            row[f"{sex}: at 'from'"] = f"{ends[0, 0]:.3f}"
            row[f"{sex}: change, points"] = effect_cell(ends[:, 1] - ends[:, 0])
        effects.append(row)
    sections.append(
        "## Change one thing, averaged over real people (percentage points)\n\n"
        f"For up to {SAMPLE:,} real people of each sex in the train set (those whose answer to "
        "this question is known), set the feature to 'from' and then to 'to', keep everything "
        "else as it is, and average the model's probability. `at 'from'`: the average probability "
        "at the 'from' value. `change`: final model, then (lowest to highest) across the 5 fold "
        "models, then stable / unstable / small (under 1 point).\n\n" + markdown(pd.DataFrame(effects)))
    curves_numeric_plot(numeric_curves, FIGURES / "curves_numeric.png")

    categorical_rows, categorical_curves = [], {}
    for feature in [c for c in CATEGORICAL if c != "sex"]:
        grid = grid_for(train, feature)
        reference = train[feature].mode()[0]
        for sex in SEXES:
            probs = curves(models, sample(train, sex, [feature], rng), along(feature, grid))
            categorical_curves[feature, sex] = (grid, probs)
            ref = probs[:, grid.index(reference)]
            for i, value in enumerate(grid):
                if value != reference:
                    categorical_rows.append({"feature": feature, "value": value, "compared with": reference,
                                             "sex": sex, "change, points": effect_cell(probs[:, i] - ref)})
    sections.append(
        "## Categories: each value against the most common one (percentage points)\n\n"
        "Same method as above.\n\n" + markdown(pd.DataFrame(categorical_rows)))
    curves_categorical_plot(categorical_curves, FIGURES / "curves_categorical.png")

    print("interactions ...", flush=True)
    bands = pd.cut(train["age"], AGE_BANDS, right=False, labels=AGE_LABELS)
    interaction_rows, interaction_cells = [], {}
    for name in BY_AGE:
        for sex in SEXES:
            for band in AGE_LABELS:
                rows = sample(train, sex, list(CHANGES[name][0]), rng, extra=bands == band)
                ends = curves(models, rows, CHANGES[name])
                change = ends[:, 1] - ends[:, 0]
                interaction_cells[name, sex, band] = change
                interaction_rows.append({"change": name, "sex": sex, "age": band,
                                         f"at 'from'": f"{ends[0, 0]:.3f}",
                                         "change, points": effect_cell(change)})
    sections.append(
        "## Interactions: the same change by age band and sex (percentage points)\n\n"
        "Same method, with the real people drawn from one age band at a time.\n\n"
        + markdown(pd.DataFrame(interaction_rows)))
    interactions_plot(interaction_cells, FIGURES / "interactions.png")

    pipe = model["base"]["catboost"]
    Xt = pipe[:-1].transform(train[FEATURES])
    pairs = pipe[-1].get_feature_importance(Pool(Xt, cat_features=CATEGORICAL), type="Interaction")[:12]
    pair_table = pd.DataFrame([{"feature 1": Xt.columns[int(a)], "feature 2": Xt.columns[int(b)],
                                "strength": round(s, 2)} for a, b, s in pairs])
    sections.append(
        "## Strongest pairwise interactions inside CatBoost\n\n"
        "CatBoost's own interaction score (how much two features' splits depend on each other; "
        "relative units, the scores sum to 100 over all pairs). A pointer for where to look, "
        "not an effect size.\n\n" + markdown(pair_table))

    print("same-person examples ...", flush=True)
    sections.append(
        "## Same person, one thing changed\n\n"
        "The typical person: for numbers, the median of train-set people of that sex aged within "
        "a year of the stated age; for categories, the most common value. Then one feature is set "
        "to 'from' and then 'to'. Probabilities from the final model; change as above.\n\n"
        + markdown(examples_table(models, train)))

    OUT.write_text("# Explanation numbers (written by src/explain.py)\n\n"
                   f"Train set: {len(train):,} rows, {train['person_id'].nunique():,} people. "
                   "Final model: CatBoost + calibration. Percentages are shares of rows "
                   "(one row per person per survey round), unweighted.\n\n"
                   + "\n\n".join(sections) + "\n")
    print(f"wrote {OUT} and figures in {FIGURES}")


def typical_person(train, sex, age):
    peers = train[(train["sex"] == sex) & train["age"].between(age - 1, age + 1)]
    person = {col: peers[col].mode()[0] if col in CATEGORICAL else peers[col].median() for col in FEATURES}
    return pd.DataFrame([{**person, "age": age}])


def examples_table(models, train):
    rows = []
    for sex, age, name in EXAMPLES:
        person = typical_person(train, sex, age)
        ends = curves(models, person, CHANGES[name])
        p = person.iloc[0]
        rows.append({
            "person": f"{sex}, {age}", "change": name,
            "probability from → to": f"{ends[0, 0]:.3f} → {ends[0, 1]:.3f}",
            "change, points": effect_cell(ends[:, 1] - ends[:, 0]),
            "the rest of this person": (f"race {p['race_ethnicity']}, {p['education']}, {p['religion']}, {p['census_region']}, "
                                        f"earns ${p['earnings']:,.0f}, {p['weeks_worked']:.0f} weeks, "
                                        f"{p['height_cm']:.0f} cm, BMI {p['bmi']:.1f}, "
                                        f"attendance {p['attendance_worship']:.0f}"),
        })
    return pd.DataFrame(rows)


def importance_plot(imp, path):
    imp = imp.iloc[::-1]
    y = np.arange(len(imp))
    fig, ax = plt.subplots(figsize=(8, 9))
    for offset, sex in [(0.2, "male"), (-0.2, "female")]:
        ax.barh(y + offset, 100 * imp[sex], height=0.36, color=SEX_COLORS[sex], label=sex)
    ax.set_yticks(y, imp.index)
    ax.set(xlabel="average push on one person's probability, percentage points (up or down)",
           title="Which facts move the prediction most, men vs women")
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def min_span(ax, points=10):
    """At least a 10-point y range, so a 1-point wiggle does not fill the panel."""
    lo, hi = ax.get_ylim()
    pad = max(0, points - (hi - lo)) / 2
    ax.set_ylim(lo - pad, hi + pad)


def curves_numeric_plot(numeric_curves, path):
    fig, axes = plt.subplots(6, 4, figsize=(14, 19))
    for ax, feature in zip(axes.flat, NUMERIC):
        for sex in SEXES:
            grid, probs = numeric_curves[feature, sex]
            ax.fill_between(grid, 100 * probs[1:].min(axis=0), 100 * probs[1:].max(axis=0),
                            color=SEX_COLORS[sex], alpha=0.18, linewidth=0)
            ax.plot(grid, 100 * probs[0], color=SEX_COLORS[sex], linewidth=2, label=sex)
        ax.set_title(feature, color=INK, fontsize=11)
        min_span(ax)
        style(ax)
    for ax in axes.flat[len(NUMERIC):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("average probability, %")
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Change one thing, keep the rest: average predicted probability of being partnered\n"
                 "(line: final model; band: range across the 5 fold models)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def curves_categorical_plot(categorical_curves, path):
    features = sorted({f for f, _ in categorical_curves})
    fig, axes = plt.subplots(len(features), 1, figsize=(8, 2.1 * len(features) + 1))
    for ax, feature in zip(axes, features):
        for offset, sex in [(-0.12, "male"), (0.12, "female")]:
            grid, probs = categorical_curves[feature, sex]
            x = np.arange(len(grid)) + offset
            # The final model saw all of train, so its dot can sit just outside the fold range.
            ax.vlines(x, 100 * probs[1:].min(axis=0), 100 * probs[1:].max(axis=0),
                      color=SEX_COLORS[sex], linewidth=2)
            ax.scatter(x, 100 * probs[0], s=64, color=SEX_COLORS[sex], edgecolor="white",
                       linewidth=1.5, zorder=3, label=sex)
        ax.set_xticks(np.arange(len(grid)), grid)
        ax.set_title(feature, color=INK, fontsize=11, loc="left")
        ax.set_ylabel("avg probability, %")
        min_span(ax)
        style(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper right")
    fig.suptitle("Categories: average predicted probability if everyone had this value\n"
                 "(dot: final model; line: range across the 5 fold models)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def interactions_plot(cells, path):
    fig, axes = plt.subplots(1, len(BY_AGE), figsize=(4 * len(BY_AGE), 4.5))
    x = np.arange(len(AGE_LABELS))
    for ax, name in zip(axes, BY_AGE):
        for offset, sex in [(-0.18, "male"), (0.18, "female")]:
            change = np.array([cells[name, sex, band] for band in AGE_LABELS])
            ax.bar(x + offset, 100 * change[:, 0], width=0.34, color=SEX_COLORS[sex], label=sex)
            ax.vlines(x + offset, 100 * change[:, 1:].min(axis=1), 100 * change[:, 1:].max(axis=1),
                      color=INK, linewidth=1.2)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(x, AGE_LABELS)
        ax.set(title=name.replace("$", r"\$"), xlabel="age")  # a bare $ starts matplotlib math
        style(ax)
    axes[0].set_ylabel("change in probability, percentage points")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper right")
    fig.suptitle("Does the same change matter more at some ages? (black line: range across fold models)")
    fig.tight_layout(rect=(0, 0, 0.95, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
