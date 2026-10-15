# ATP evaluation report

Chronological split: train 70,947 matches, validation 5,099, test 5,355. Data cutoff: 2024-12-16.
Rows are completed tour-level main-draw matches (retirements and walkovers excluded).

## Test set

| Model | Log-loss ↓ | Brier ↓ | Accuracy ↑ | ROC-AUC ↑ | ECE ↓ |
|---|---:|---:|---:|---:|---:|
| Rank baseline | 0.6282 | 0.2196 | 63.8% | 0.698 | 0.015 |
| Elo baseline | 0.6045 | 0.2098 | 65.6% | 0.729 | 0.032 |
| Logistic regression | **0.5915** | 0.2044 | 66.6% | 0.743 | 0.026 |
| XGBoost (calibrated) | 0.5949 | 0.2058 | 65.8% | 0.740 | 0.031 |

## Validation set

Used for early stopping, model selection and calibration.

| Model | Log-loss ↓ | Brier ↓ | Accuracy ↑ | ROC-AUC ↑ | ECE ↓ |
|---|---:|---:|---:|---:|---:|
| Rank baseline | 0.6269 | 0.2190 | 63.9% | 0.700 | 0.012 |
| Elo baseline | 0.6017 | 0.2086 | 66.3% | 0.732 | 0.021 |
| Logistic regression | 0.5906 | 0.2037 | 67.6% | 0.746 | 0.013 |
| XGBoost (calibrated) | **0.5900** | 0.2034 | 67.5% | 0.747 | 0.013 |

## Paired comparisons (test set, bootstrap 95% CI)

Mean per-match difference, first model minus second: negative values favour the first model. An interval containing 0 means no significant difference.

| Comparison | Metric | Difference | 95% CI |
|---|---|---:|---|
| Logistic regression vs Elo baseline | log_loss | -0.0130 | [-0.0174, -0.0091] |
| Logistic regression vs Elo baseline | brier | -0.0054 | [-0.0072, -0.0036] |
| XGBoost (calibrated) vs Elo baseline | log_loss | -0.0096 | [-0.0136, -0.0057] |
| XGBoost (calibrated) vs Elo baseline | brier | -0.0040 | [-0.0058, -0.0023] |
| XGBoost (calibrated) vs Logistic regression | log_loss | +0.0034 | [+0.0016, +0.0054] |
| XGBoost (calibrated) vs Logistic regression | brier | +0.0014 | [+0.0006, +0.0022] |

## Test set by surface

| surface | n | Elo log-loss | Elo accuracy | Logistic log-loss | Logistic accuracy | XGBoost log-loss | XGBoost accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clay | 1,726 | 0.6146 | 64.2% | 0.5954 | 67.1% | 0.5992 | 65.5% |
| Grass | 634 | 0.6092 | 64.4% | 0.5912 | 66.6% | 0.5960 | 65.1% |
| Hard | 2,995 | 0.5976 | 66.6% | 0.5892 | 66.4% | 0.5922 | 66.2% |

## Test set by level

| tourney_level | n | Elo log-loss | Elo accuracy | Logistic log-loss | Logistic accuracy | XGBoost log-loss | XGBoost accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 2,936 | 0.6259 | 63.5% | 0.6130 | 64.7% | 0.6167 | 63.6% |
| F | 29 | 0.5565 | 65.5% | 0.5893 | 69.0% | 0.5812 | 65.5% |
| G | 983 | 0.5267 | 72.4% | 0.5053 | 73.3% | 0.5086 | 73.1% |
| M | 1,344 | 0.6218 | 64.7% | 0.6134 | 65.5% | 0.6168 | 65.0% |
| O | 63 | 0.4705 | 74.6% | 0.4660 | 77.8% | 0.4659 | 76.2% |

## Test set by best of

| best_of | n | Elo log-loss | Elo accuracy | Logistic log-loss | Logistic accuracy | XGBoost log-loss | XGBoost accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 4,372 | 0.6220 | 64.0% | 0.6109 | 65.1% | 0.6143 | 64.2% |
| 5 | 983 | 0.5267 | 72.4% | 0.5053 | 73.3% | 0.5086 | 73.1% |

## Selected model

```json
{
  "best_params": {
    "learning_rate": 0.03,
    "max_depth": 5.0,
    "min_child_weight": 50.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0
  },
  "best_iteration": 466,
  "calibration_slope": 1.0055,
  "search": [
    {
      "params": {
        "learning_rate": 0.03,
        "max_depth": 3.0,
        "min_child_weight": 10.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0
      },
      "best_iteration": 840,
      "val_log_loss": 0.5902899175107563
    },
    {
      "params": {
        "learning_rate": 0.03,
        "max_depth": 4.0,
        "min_child_weight": 20.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0
      },
      "best_iteration": 480,
      "val_log_loss": 0.5901617365522414
    },
    {
      "params": {
        "learning_rate": 0.03,
        "max_depth": 5.0,
        "min_child_weight": 50.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0
      },
      "best_iteration": 466,
      "val_log_loss": 0.5899891316065746
    }
  ]
}
```
