# Test set results

26,434 rows, 1,770 people never seen in training. Unweighted (survey weights not used). Lower is better except ROC-AUC. calib_error: mean gap between predicted and observed rate over 10 equal-size bins.

## All models

| group | rows | partnered | log_loss | brier | roc_auc | calib_error |
|---|---|---|---|---|---|---|
| age + sex baseline | 26434 | 0.428 | 0.6007 | 0.2086 | 0.7178 | 0.0100 |
| catboost | 26434 | 0.428 | 0.5431 | 0.1845 | 0.7892 | 0.0146 |
| final stack | 26434 | 0.428 | 0.5430 | 0.1845 | 0.7892 | 0.0139 |

## Final stack by sex

| group | rows | partnered | log_loss | brier | roc_auc | calib_error |
|---|---|---|---|---|---|---|
| female | 13466 | 0.460 | 0.5600 | 0.1909 | 0.7774 | 0.0200 |
| male | 12968 | 0.395 | 0.5252 | 0.1779 | 0.7973 | 0.0168 |

## Final stack by age band

| group | rows | partnered | log_loss | brier | roc_auc | calib_error |
|---|---|---|---|---|---|---|
| 18-24 | 10989 | 0.233 | 0.4452 | 0.1457 | 0.7859 | 0.0128 |
| 25-29 | 6800 | 0.502 | 0.6365 | 0.2227 | 0.6903 | 0.0208 |
| 30-34 | 3608 | 0.598 | 0.6026 | 0.2079 | 0.7173 | 0.0267 |
| 35-39 | 3554 | 0.639 | 0.5881 | 0.2016 | 0.7168 | 0.0276 |
| 40-43 | 1483 | 0.619 | 0.5851 | 0.1999 | 0.7358 | 0.0446 |

## Confidently wrong

Rows where the final model was sure, split into right and wrong answers.

| confidence | actually | rows | share | mean age | men | median earnings | mean weeks worked | has nonresident children |
|---|---|---|---|---|---|---|---|---|
| predicted ≥ 0.9 | right | 158 | 0.919 | 37.8 | 0.48 | 102,500 | 41.2 | 0.00 |
| predicted ≥ 0.9 | wrong | 14 | 0.081 | 38.6 | 0.57 | 117,000 | 41.0 | 0.00 |
| predicted ≤ 0.1 | right | 3008 | 0.963 | 19.2 | 0.64 | 3,000 | 26.0 | 0.01 |
| predicted ≤ 0.1 | wrong | 114 | 0.037 | 19.6 | 0.66 | 5,000 | 30.1 | 0.03 |

## Figures

![calibration](figures/calibration.png)

![rate by age](figures/rate_by_age.png)
