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

| survey | interviews | ages | rows scored | partnered | height / BMI | employed asks |
|---|---|---|---|---|---|---|
| NLSY97 | 1998-2024 | 18-43 | 26,434 | 0.428 | self-reported | worked any weeks last year |
| NHANES | 2007-2021 | 20-43 | 16,410 | 0.554 | measured at exam | working now |
| HCMST | 2017 | 18-43 | 1,416 | 0.606 | not asked | working for pay now |

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
(8% of 20-43s lack measured height).

| survey | model | rows | partnered | mean pred. | log loss | brier | ROC-AUC | calib. error |
|---|---|---|---|---|---|---|---|---|
| NLSY97 (test split) | core model (trained on NLSY97) | 26,434 | 0.428 | 0.435 | 0.5668 | 0.1937 | 0.7661 | 0.0114 |
| NLSY97 (test split) | age + sex rates of this survey | 26,434 | 0.428 | 0.437 | 0.6007 | 0.2086 | 0.7178 | 0.0100 |
| NHANES (20-43) | core model (trained on NLSY97) | 16,410 | 0.554 | 0.537 | 0.6164 | 0.2131 | 0.7106 | 0.0281 |
| NHANES (20-43) | age + sex rates of this survey | 16,410 | 0.554 | 0.554 | 0.6304 | 0.2200 | 0.6768 | 0.0001 |
| HCMST (18-43) | core model (trained on NLSY97) | 1,416 | 0.606 | 0.385 | 0.7688 | 0.2710 | 0.6961 | 0.2206 |
| HCMST (18-43) | age + sex rates of this survey | 1,416 | 0.606 | 0.606 | 0.5553 | 0.1867 | 0.7579 | 0.0007 |

The comparison floor is each survey's **own age + sex rates** (fitted on that survey;
the NLSY97 floor is fitted on train and scored on the held-out test split, as in
evaluate.py - NHANES's and HCMST's floors are fitted on the rows they are scored on).
Mean pred. is the average prediction - how far off the *level* is.

What the table says:

- **NHANES: the model transfers.** Better log loss, better AUC than the local floor, and
  the level is nearly right. Moving from the home test split to NHANES costs
  0.050 log loss and 0.055 AUC.
- **HCMST: the ranking survives, the level does not.** AUC stays at 0.696
  (men 0.739, women 0.677),
  but the average prediction is 22 pp too low
  - a 60.6%-partnered survey scored by a model that says
  38.5% - so log loss collapses and HCMST's own age + sex rates beat the
  transported model on both metrics. Part of that level gap is the survey itself: HCMST
  2017's married share runs ~8 pp above Census for the same years in its own published
  numbers (panel participation favours people in couples; noted when the data was
  cleaned). The rest is era: HCMST is 2017 only, when 20-somethings partnered later than
  across NLSY97's mixed decades.
- For scale: the full-featured main model scores 0.5425 log loss / 0.7896 AUC on
  the same test split (`results/metrics.md`); keeping only the seven shared columns
  costs 0.024 log loss and 0.023 AUC.

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

| feature | NLSY97 men | NLSY97 women | NHANES men | NHANES women | HCMST men | HCMST women |
|---|---|---|---|---|---|---|
| age | 0.85 | 0.74 | 0.64 | 0.53 | 1.05 | 0.86 |
| race/ethnicity | 0.29 | 0.43 | 0.23 | 0.32 | 0.18 | 0.23 |
| sex | 0.27 | 0.26 | 0.11 | 0.11 | 0.20 | 0.20 |
| education | 0.21 | 0.22 | 0.14 | 0.12 | 0.16 | 0.12 |
| BMI | 0.17 | 0.15 | 0.15 | 0.12 | — | — |
| employed | 0.11 | 0.03 | 0.13 | 0.07 | 0.34 | 0.11 |
| height | 0.07 | 0.06 | 0.07 | 0.04 | — | — |

| feature | NLSY97 | NHANES | HCMST |
|---|---|---|---|
| age | 18-24 -20.7 pp ... 35-43 +21.4 pp | 18-24 -29.8 pp ... 35-43 +12.8 pp | 18-24 -43.0 pp ... 35-43 +20.3 pp |
| race/ethnicity | black -14.9 pp ... other +6.8 pp | black -17.1 pp ... hispanic +5.5 pp | black -17.1 pp ... other +3.2 pp |
| sex | male -6.4 pp ... female +6.5 pp | male -1.2 pp ... female +0.6 pp | male -4.0 pp ... female +4.0 pp |
| education | some_college -5.5 pp ... less_than_hs +6.6 pp | some_college -3.5 pp ... less_than_hs +3.8 pp | some_college -3.5 pp ... less_than_hs +6.0 pp |
| BMI | low -4.9 pp ... high +3.9 pp | low -2.4 pp ... middle +2.4 pp | not measured in this survey |
| employed | not employed -5.0 pp ... employed +1.0 pp | not employed -0.8 pp ... employed +0.6 pp | not employed -11.1 pp ... employed +3.3 pp |
| height | low -0.7 pp ... high +1.6 pp | middle -0.2 pp ... low +0.4 pp | not measured in this survey |

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

| age | men | men with a partner | women | women with a partner |
|---|---|---|---|---|
| 18-24 | 94 | 29% | 104 | 58% |
| 25-29 | 93 | 28% | 92 | 59% |
| 30-34 | 26 | 42% | 38 | 63% |
| 35-43 | 56 | 41% | 55 | 62% |
| all 18-43 | 269 | 32% | 289 | 60% |

46% of them have a partner
they don't live with - 60%
of the women but only 32%
of the men, a gap that holds in every age band. So the main model's "single" class is a
mix - mostly unpartnered men and women who do have a partner - and anything the model
says about "singles" is about *not cohabiting*, not about being alone. A feature that
helps people date without moving in (many, plausibly) will look weaker for this target
than it is for "has a relationship at all".

## The short version

- **The recipe travels to NHANES.** Scored unchanged, the NLSY97 core model beats NHANES's own age + sex rates on everything: 0.616 vs 0.630 log loss, 0.711 vs 0.677 AUC. Even its level is nearly right (mean prediction 0.537 against 0.554 observed).
- **HCMST breaks the level, not the ranking.** The same model still discriminates (0.696 AUC) but underpredicts by 22 pp on average (0.385 predicted vs 0.606 observed), so log loss collapses to 0.769 and HCMST's own age + sex rates score better (0.555 / 0.758).
- **Age dominates in all three surveys** (largest mean |SHAP| for both sexes everywhere), and the young-adult gap widened over time: 18-24 sits 21 / 30 / 43 pp below its survey's average in NLSY97 / NHANES / HCMST.
- **The Black partnered gap replicates at full size**: -14.9 / -17.1 / -17.1 pp below the survey average in NLSY97 / NHANES / HCMST, and the push is bigger for women than men in all three (e.g. NLSY97: 0.43 vs 0.29 mean |SHAP|).
- **"Some college, no degree" partners latest in all three surveys** (-5.5 / -3.5 / -3.5 pp), while the ends of the education range partner sooner (less than high school +6.6 / +3.8 / +6.0 pp). The shape is a U, not a ladder: it is not "more education, more partnering".
- **Employment pushes up everywhere and matters more for men** (mean |SHAP|, men vs women: NLSY97 0.11/0.03, NHANES 0.13/0.07, HCMST 0.34/0.11), but its size tracks the question asked: the not-employed-vs-employed gap is 6 pp under NLSY97's "worked any weeks last year", 1 pp under NHANES's "working now" and 14 pp in HCMST.
- **Height barely registers wherever it is measured** (mean |SHAP| at most 0.07, largest low-to-high gap 2.3 pp) - the "short men partner less" story does not show up in who ends up partnered. BMI: the lightest tertile sits -4.9 pp (NLSY97) and -2.4 pp (NHANES) below average; above that, little.
- **"Single" means "no live-in partner" - and many singles are not partnerless.** In HCMST, 46% of the 18-43s the main model would call single have a girlfriend or boyfriend they don't live with: women 60%, men 32%. The main model's target cannot see relationships short of cohabitation, and the men it calls single are far more often truly alone than the women.

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
out-of-sample score. HCMST is small (1,416 rows, ~650 per sex), so its per-sex
numbers wobble. NHANES cycles are pooled as if one survey. And the core model's seven
features leave out most of what matters (earnings, family background, personality), so
"what held" here is about these seven, not about the full model.
