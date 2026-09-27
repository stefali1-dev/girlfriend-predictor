"""Score the final model on the test set, once.

Writes results/metrics.md and results/figures/{calibration,rate_by_age}.png.
Everything was chosen with cross-validation on train; the test set only reports.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from predict import MODEL, base_probabilities
from train import FEATURES, TARGET, TRAIN, make_models, markdown, metrics

TEST = Path("data/curated/test.parquet")
OUT = Path("results/metrics.md")
FIGURES = Path("results/figures")
AGE_BANDS = [18, 25, 30, 35, 40, 44]
AGE_LABELS = ["18-24", "25-29", "30-34", "35-39", "40-43"]

# Reference chart palette: series in fixed order, recessive ink for everything else.
SEX_COLORS = {"male": "#2a78d6", "female": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def metrics_row(label, y, p):
    return {"group": label, "rows": len(y), "partnered": f"{y.mean():.3f}",
            **{k: f"{v:.4f}" for k, v in metrics(y, p).items()}}


def style(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def calibration_plot(test, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="perfect calibration")
    for sex, rows in test.groupby("sex"):
        bins = rows.groupby(pd.qcut(rows["p"], 10, duplicates="drop"), observed=True)
        ax.plot(bins["p"].mean(), bins[TARGET].mean(), marker="o", markersize=8, linewidth=2,
                color=SEX_COLORS[sex], label=f"{sex} (10 equal-size bins)")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="predicted probability of being partnered",
           ylabel="observed share partnered",
           title="Test set: predicted vs observed, by sex")
    style(ax)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rate_by_age_plot(test, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    by_age = test.groupby(["sex", "age"])[[TARGET, "p"]].mean().reset_index()
    for sex, rows in by_age.groupby("sex"):
        color = SEX_COLORS[sex]
        ax.plot(rows["age"], rows["p"], color=color, linewidth=2, label=f"{sex}: model (mean prediction)")
        ax.scatter(rows["age"], rows[TARGET], color=color, s=40, edgecolor="white", linewidth=1.5,
                   zorder=3, label=f"{sex}: observed")
    ax.set(xlabel="age", ylabel="share partnered", ylim=(0, 1),
           title="Test set: partnered share by age and sex, model vs observed")
    style(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confident_errors(test):
    """Rows the model was sure about (below 0.1 or above 0.9) and how often it was wrong."""
    rows = []
    for label, sure, wrong_answer in [("predicted ≥ 0.9", test["p"] >= 0.9, 0),
                                      ("predicted ≤ 0.1", test["p"] <= 0.1, 1)]:
        for outcome, part in test[sure].groupby(test[TARGET] == wrong_answer):
            rows.append({
                "confidence": label, "actually": "wrong" if outcome else "right",
                "rows": len(part), "share": f"{len(part) / sure.sum():.3f}",
                "mean age": f"{part['age'].mean():.1f}", "men": f"{(part['sex'] == 'male').mean():.2f}",
                "median earnings": f"{part['earnings'].median():,.0f}",
                "mean weeks worked": f"{part['weeks_worked'].mean():.1f}",
                "has nonresident children": f"{(part['nonresident_children'] > 0).mean():.2f}",
            })
    return pd.DataFrame(rows)


def main():
    train, test = pd.read_parquet(TRAIN), pd.read_parquet(TEST)
    model = joblib.load(MODEL)
    y = test[TARGET].to_numpy()

    base = base_probabilities(model, test)
    test["p"] = model["stacker"].predict_proba(base)[:, 1]

    # The age + sex floor is refitted on train here only to put the test numbers in context.
    floor = make_models()["age_sex_baseline"].fit(train[FEATURES], train[TARGET])
    overall = [metrics_row("age + sex baseline", y, floor.predict_proba(test[FEATURES])[:, 1])]
    overall += [metrics_row(name, y, base[:, i]) for i, name in enumerate(model["base"])]
    overall += [metrics_row("final stack", y, test["p"].to_numpy())]

    by_sex = [metrics_row(sex, rows[TARGET].to_numpy(), rows["p"].to_numpy())
              for sex, rows in test.groupby("sex")]
    bands = pd.cut(test["age"], AGE_BANDS, right=False, labels=AGE_LABELS)
    by_age = [metrics_row(band, rows[TARGET].to_numpy(), rows["p"].to_numpy())
              for band, rows in test.groupby(bands, observed=True)]

    FIGURES.mkdir(parents=True, exist_ok=True)
    calibration_plot(test, FIGURES / "calibration.png")
    rate_by_age_plot(test, FIGURES / "rate_by_age.png")

    OUT.write_text(
        "# Test set results\n\n"
        f"{len(test):,} rows, {test['person_id'].nunique():,} people never seen in training. "
        "Unweighted (survey weights not used). Lower is better except ROC-AUC. "
        "calib_error: mean gap between predicted and observed rate over 10 equal-size bins.\n\n"
        "Second opening of the test set: a re-report after a data fix (hours_worked above "
        "8,760 set to missing), with no changes to the model code or settings.\n\n"
        "## All models\n\n" + markdown(pd.DataFrame(overall)) + "\n\n"
        "## Final stack by sex\n\n" + markdown(pd.DataFrame(by_sex)) + "\n\n"
        "## Final stack by age band\n\n" + markdown(pd.DataFrame(by_age)) + "\n\n"
        "## Confidently wrong\n\n"
        "Rows where the final model was sure, split into right and wrong answers.\n\n"
        + markdown(confident_errors(test)) + "\n\n"
        "## Figures\n\n"
        "![calibration](figures/calibration.png)\n\n![rate by age](figures/rate_by_age.png)\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
