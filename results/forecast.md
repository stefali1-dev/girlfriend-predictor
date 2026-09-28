# Finding a partner within about two years

Among NLSY97 adults who were **single** at an interview - not married, not living with a
partner - **15.7% had a partner (married or living together) at the
interview closest to 2 years later**. This page builds a model that predicts that from facts
known **before**: age, education, work, earnings, height, personality, family background.
Because every input is older than the outcome, a partner's own income or housing cannot
masquerade as "earnings help" - the classic reason "partnered right now" numbers overstate
what a person can change.

## The rows

64,872 rows, 8,209 people, single at ages 18-41,
interviews 1998-2022. Each row needs a follow-up interview
2-3 years later (51,331 rows matched at 2 years, 13,541 at 3); single interviews without one
were dropped. Across all rows, 20.4% found a partner.

## Train and test: earlier interviews vs the latest ones

- **Train:** rounds 1-17 (interviews
  1998-2016), 41,156 rows,
  5,707 people, 21.2% found a partner.
- **Test:** rounds 18-20 (interviews
  2017-2022), 1,812 rows,
  870 people, 15.7% found a partner.
- **Set aside, unused:** 21,904 of 64,872 rows (34%)
  - 4,243 latest-round rows of training people (train stays in the earlier
  rounds), 7,958 earlier-round rows of the test people, and
  9,703 rows of the 1,593 held-out people who never reach a
  test-round interview (people are held out before anyone knows who will still be single
  and interviewed in 2017-2022). This is the price of a
  train set that is not stacked toward people who partnered in the end.

Two rules, and why. **Time:** every test interview happened after every training interview,
so the test imitates the real use: applying the model to a future it has not seen. **People
separate:** keeping a person's earlier rows in train while their latest row is in test would
let the model partly "recognise" the person instead of learning what singles are like.

The order of those two rules matters, and the first attempt got it wrong. Removing "the test
people's earlier rows" from train sounds natural - but the test people **are** the long-term
singles, and deleting exactly them left train with the singles who partnered in the end:
68% of that train's 35+ singles found a partner within 2 years, against the true
17%. Every model trained on such a set overpredicts wildly on the
test years. Holding people out **at random** instead (the other 70% of people
to train, the rest never seen) keeps train representative while both rules still hold.

**Where the knobs come from** (so the test set is not quietly tuning them). TEST_ROUND is
18 because rounds 18-20 are simply the latest interviews with
a possible follow-up; HELD_SHARE is 30%, fixed before any test number was
computed, to keep the test big enough to read (it lands at 1,812 rows); RECENT_ROUND is 12, the first
round with Big Five answers, so the calibrator sees the feature world of the test era. None
of the three was adjusted after seeing test numbers. One design choice did use the test era:
the split itself - the random-people variant was adopted after the drop-variant's
overprediction was measured on test-era rows (the 68%-vs-17%
story above). It was not picked for score, but it is test-era information influencing design,
disclosed here.

What the split costs, honestly: the NLSY97 cohort was born 1980-84, so the test interviews
only contain ages 32-41 - behavior for singles in their
20s is measured only by cross-validation on earlier data - and the training rounds contain
no single person older than 36 (238 training
rows are 35+), while 54% of test rows are older. There, every model is
extrapolating past anything it has seen. And partnering rates fell over the decades, so the
calibrator is fitted on out-of-fold predictions from rounds 12+ only
(11,761 rows, interviews 2008 onward), the era closest to the test years; it
puts weight 0.85 on CatBoost's log-odds. The calibration plot shows how
well that worked.

## Model comparison (cross-validation on train)

The "new people, same era" measure: each fold's people are disjoint, but the interview
years mix, so it shows how well the model reads people in a familiar world.
Mean ± sd over 5 folds grouped by person. Lower is better except ROC-AUC.

| model | log_loss | brier | roc_auc | calib_error |
|---|---|---|---|---|
| age_sex_baseline | 0.5119 ± 0.0098 | 0.1656 ± 0.0041 | 0.5628 ± 0.0127 | 0.0163 ± 0.0046 |
| logistic | 0.4961 ± 0.0123 | 0.1603 ± 0.0048 | 0.6426 ± 0.0145 | 0.0160 ± 0.0033 |
| catboost | 0.4908 ± 0.0117 | 0.1588 ± 0.0046 | 0.6578 ± 0.0124 | 0.0163 ± 0.0020 |

## Test: the latest interviews (2017-2022)

The "future era" stress test: every interview later than every training interview, and more
than half at ages training never saw. Expect it to score below cross-validation; that gap
is the price of time.

| group | rows | partnered | log_loss | brier | roc_auc | calib_error |
|---|---|---|---|---|---|---|
| age + sex baseline | 1812 | 0.157 | 0.4473 | 0.1363 | 0.4532 | 0.0674 |
| logistic | 1812 | 0.157 | 0.4254 | 0.1304 | 0.6277 | 0.0323 |
| catboost (raw) | 1812 | 0.157 | 0.4359 | 0.1358 | 0.6350 | 0.0581 |
| catboost + calibration (final) | 1812 | 0.157 | 0.4303 | 0.1330 | 0.6350 | 0.0492 |

Two honest notes on this table. First, **logistic regression beats the final model** on
log loss, Brier and calibration (0.4254 vs 0.4303 log
loss). The gap is inside noise: the average per-row log-loss difference is 0.0049
with standard error 0.0051, and a person-level bootstrap 95% interval runs
[-0.0069, +0.0160] across zero. CatBoost stays final anyway, because the
cross-validation above - 41,156 rows, 23× the test set,
same-era - favours it on log loss and AUC, and because the SHAP explanation needs its
trees. Second, the age + sex floor scores below chance on AUC (0.4532) for the
extrapolation reason above: at the ages it never saw it predicts the same average for
everyone, which ranks the 37+ test rows too high. Logistic, which bends a curve through age
instead of memorising cells, is the honest "guess by age and sex" here.

By sex and by age band (final model):

| group | rows | partnered | log_loss | brier | roc_auc | calib_error |
|---|---|---|---|---|---|---|
| female | 975 | 0.159 | 0.4309 | 0.1339 | 0.6397 | 0.0431 |
| male | 837 | 0.155 | 0.4296 | 0.1319 | 0.6340 | 0.0636 |

| group | rows | partnered | log_loss | brier | roc_auc | calib_error |
|---|---|---|---|---|---|---|
| 30-34 | 287 | 0.199 | 0.4681 | 0.1507 | 0.6787 | 0.0382 |
| 35-41 | 1525 | 0.150 | 0.4232 | 0.1296 | 0.6279 | 0.0599 |

Read the bands as rough: 30-34 is 287 rows, and 35-41 is mostly
ages the training rounds never saw (65%
of its rows are older than 36).

For scale: the same recipe predicting "partnered now" scores 0.790 AUC on
its own held-out test set, versus 0.6350 here - the 2-year question is
genuinely harder, as expected for something with a lot of chance in it.

![calibration](figures/forecast_calibration.png)

![rate by age](figures/forecast_rate_by_age.png)

## What goes with finding a partner (SHAP)

Each row is one feature: its average push on the model (log-odds) for men and for women
separately, and the value groups with the strongest down and up push, in plain percentage
points (pp) around the training average of 21%. The pushes are
computed on the 41,156 training rows (same era, 23× the
test set) so rare feature values are not read off a thin slice; the test set above stays
for scoring only. Red dots in the figures are high feature values, blue are low
(a sample of 5,000 rows each).

| feature | men: avg push | women: avg push | values with the strongest down vs up push (shift in the chance) |
|---|---|---|---|
| race_ethnicity | 0.20 | 0.28 | black -5.4 pp ... other +3.5 pp |
| sex | 0.16 | 0.18 | male -2.5 pp ... female +3.1 pp |
| earnings | 0.16 | 0.13 | low -1.9 pp ... high +3.7 pp |
| enrolled | 0.13 | 0.12 | high_school -3.6 pp ... graduate +1.7 pp |
| hours_worked | 0.12 | 0.11 | low -2.3 pp ... high +2.6 pp |
| age | 0.12 | 0.10 | 35-41 -4.8 pp ... 25-29 +1.3 pp |
| bmi | 0.08 | 0.10 | high -2.0 pp ... middle +1.2 pp |
| family_income_1997 | 0.06 | 0.07 | high -0.2 pp ... middle +1.1 pp |
| nonresident_children | 0.07 | 0.04 | 0 -0.5 pp ... any other value +6.6 pp |
| census_region | 0.05 | 0.05 | northeast -2.1 pp ... midwest +0.8 pp |

![SHAP men](figures/forecast_shap_male.png)

![SHAP women](figures/forecast_shap_female.png)

## "Partnered now" vs "finds a partner": which features may run backwards

The "now" columns refit the partnered-now recipe and explain it the same way, restricted to
ages 32-41 so a feature's size means the same thing on both
sides; the "finds a partner" columns use the training rows. Shares are each feature's slice
of total importance. A feature big for "now" but small for "finding" mostly marks *having*
a partner, not *getting* one - often the partner is the cause (two incomes, shared
housing). These are associations, not proven causes.

| feature | 'partnered now' share | its low vs high values, now | 'finds a partner' share | its low vs high values, finding | read |
|---|---|---|---|---|---|
| age | 21% | 30-34 +12.2 pp ... 35-41 +12.2 pp | 7% | 35-41 -4.8 pp ... 25-29 +1.3 pp | matters for now, much less for finding: likely runs backwards |
| race_ethnicity | 13% | black -16.1 pp ... other +5.9 pp | 15% | black -5.4 pp ... other +3.5 pp | matters for both questions |
| earnings | 13% | low -0.7 pp ... high +14.0 pp | 9% | low -1.9 pp ... high +3.7 pp | matters for both questions |
| sex | 6% | male -3.5 pp ... female +4.0 pp | 11% | male -2.5 pp ... female +3.1 pp | minor for both |
| enrolled | 4% | high_school -8.7 pp ... not_enrolled +2.4 pp | 8% | high_school -3.6 pp ... graduate +1.7 pp | minor for both |
| hours_worked | 2% | low -0.9 pp ... high +1.5 pp | 7% | low -2.3 pp ... high +2.6 pp | minor for both |
| bmi | 3% | low -1.9 pp ... high +2.2 pp | 6% | high -2.0 pp ... middle +1.2 pp | minor for both |
| big5_extraversion | 5% | low -0.1 pp ... high +5.9 pp | 1% | low -0.7 pp ... high +2.2 pp | minor for both |
| family_income_1997 | 2% | low -0.4 pp ... middle +0.7 pp | 4% | high -0.2 pp ... middle +1.1 pp | minor for both |
| nonresident_children | 2% | any other value -5.4 pp ... 0 +0.7 pp | 4% | 0 -0.5 pp ... any other value +6.6 pp | minor for both |
| census_region | 2% | northeast -4.8 pp ... midwest +1.4 pp | 3% | northeast -2.1 pp ... midwest +0.8 pp | minor for both |
| big5_conscientiousness | 3% | low -2.6 pp ... high +2.6 pp | 1% | low +0.0 pp ... high +0.5 pp | minor for both |

## The short version

- **Race is the model's biggest push** (black -5.4 pp, hispanic +2.2 pp, other +3.5 pp around the training average; bigger for women (0.28) than men (0.20)). One caveat: the survey oversamples Black respondents - 45% of test rows - so unweighted shares flatter race.
- **Own earnings help, but modestly.** The low-to-high earnings gap is 6 pp here, against 15 pp in the partnered-now model at matched ages: money mostly marks *having* a partner more than it drives *getting* one.
- **Children living elsewhere flip sign - among otherwise similar singles.** The forecast model puts such singles +6.6 pp up, the partnered-now model 5.4 pp down. The raw training gap is +5.7 pp (men +7.3, women +5.3); in the small test slice the raw gap vanishes (-0.6 pp), so the numbers compare similar people, not raw rates.
- **Enrollment separates now, not next.** Students and non-enrolled differ by 11 pp in the partnered-now model at matched ages but only 5 pp here: it shifts *when* people partner, not who finds someone - and at the test ages students are rare (118 of 1,812 singles).
- **Outgoing people do better on both questions.** The low-to-high extraversion gap is 3 pp here and 6 pp for being partnered - one of the few personality reads that survives both questions.

## Limits

Associations, not causes - measuring features before the outcome removes the worst backwards
arrow, not all of them. One US cohort born 1980-84, and the test years cover ages
32-41 only. Unweighted (survey weights not used; the
oversample of Black respondents also enlarges race's importance share), and people who left
the survey are absent. The target counts only marriage and living together: a girlfriend or
boyfriend not living together counts as "still single".
