# Cross-validation on train (5 folds grouped by person)

105,145 rows, 7,080 people. Mean ± standard deviation over the 5 validation folds. Lower is better except ROC-AUC.

| model | log_loss | brier | roc_auc | calib_error |
|---|---|---|---|---|
| age_sex_baseline | 0.6067 ± 0.0054 | 0.2111 ± 0.0024 | 0.7108 ± 0.0082 | 0.0132 ± 0.0040 |
| logistic | 0.5701 ± 0.0080 | 0.1952 ± 0.0035 | 0.7645 ± 0.0086 | 0.0163 ± 0.0040 |
| catboost | 0.5462 ± 0.0085 | 0.1856 ± 0.0036 | 0.7885 ± 0.0086 | 0.0124 ± 0.0039 |
| tabpfn | 0.5544 ± 0.0100 | 0.1890 ± 0.0042 | 0.7807 ± 0.0096 | 0.0209 ± 0.0073 |
| catboost + calibration (stacker on catboost alone) | 0.5463 ± 0.0086 | 0.1856 ± 0.0037 | 0.7885 ± 0.0086 | 0.0140 ± 0.0049 |
| stack: logistic + catboost | 0.5464 ± 0.0086 | 0.1857 ± 0.0037 | 0.7884 ± 0.0086 | 0.0142 ± 0.0049 |
| stack: catboost + tabpfn | 0.5462 ± 0.0087 | 0.1856 ± 0.0037 | 0.7886 ± 0.0088 | 0.0149 ± 0.0061 |
| stack: logistic + catboost + tabpfn | 0.5463 ± 0.0087 | 0.1856 ± 0.0037 | 0.7885 ± 0.0088 | 0.0148 ± 0.0059 |

Final model: stack of catboost; stacker weights on each model's log-odds: {'catboost': 1.022}.
