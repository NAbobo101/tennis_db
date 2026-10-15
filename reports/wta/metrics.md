# WTA evaluation report

Chronological split: train 27,456 matches, validation 4,675, test 4,916. Data cutoff: 2024-12-16.
Rows are completed tour-level main-draw matches (retirements and walkovers excluded).

## Test set

| Model | Log-loss ↓ | Brier ↓ | Accuracy ↑ | ROC-AUC ↑ | ECE ↓ |
|---|---:|---:|---:|---:|---:|
| Rank baseline | 0.6312 | 0.2209 | 63.8% | 0.695 | 0.017 |
| Elo baseline | 0.5991 | 0.2072 | 66.3% | 0.736 | 0.018 |
| Logistic regression | 0.5918 | 0.2040 | 67.7% | 0.746 | 0.014 |
| XGBoost (calibrated) | **0.5916** | 0.2040 | 67.4% | 0.745 | 0.011 |

## Validation set

Used for early stopping, model selection and calibration.

| Model | Log-loss ↓ | Brier ↓ | Accuracy ↑ | ROC-AUC ↑ | ECE ↓ |
|---|---:|---:|---:|---:|---:|
| Rank baseline | 0.6330 | 0.2211 | 64.6% | 0.697 | 0.017 |
| Elo baseline | 0.5977 | 0.2059 | 67.9% | 0.741 | 0.018 |
| Logistic regression | 0.5920 | 0.2036 | 67.7% | 0.748 | 0.014 |
| XGBoost (calibrated) | **0.5920** | 0.2035 | 68.0% | 0.748 | 0.013 |

## Paired comparisons (test set, bootstrap 95% CI)

Mean per-match difference, first model minus second: negative values favour the first model. An interval containing 0 means no significant difference.

| Comparison | Metric | Difference | 95% CI |
|---|---|---:|---|
| Logistic regression vs Elo baseline | log_loss | -0.0073 | [-0.0114, -0.0033] |
| Logistic regression vs Elo baseline | brier | -0.0032 | [-0.0051, -0.0014] |
| XGBoost (calibrated) vs Elo baseline | log_loss | -0.0074 | [-0.0109, -0.0039] |
| XGBoost (calibrated) vs Elo baseline | brier | -0.0032 | [-0.0047, -0.0017] |
| XGBoost (calibrated) vs Logistic regression | log_loss | -0.0001 | [-0.0023, +0.0020] |
| XGBoost (calibrated) vs Logistic regression | brier | +0.0001 | [-0.0008, +0.0010] |

## Test set by surface

| surface | n | Elo log-loss | Elo accuracy | Logistic log-loss | Logistic accuracy | XGBoost log-loss | XGBoost accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clay | 1,270 | 0.5940 | 66.7% | 0.5849 | 68.3% | 0.5871 | 67.4% |
| Grass | 586 | 0.6115 | 65.4% | 0.5921 | 67.7% | 0.5910 | 67.9% |
| Hard | 3,060 | 0.5988 | 66.2% | 0.5946 | 67.4% | 0.5936 | 67.3% |

## Test set by level

| tourney_level | n | Elo log-loss | Elo accuracy | Logistic log-loss | Logistic accuracy | XGBoost log-loss | XGBoost accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| F | 45 | 0.6477 | 57.8% | 0.6277 | 60.0% | 0.6143 | 60.0% |
| G | 992 | 0.5616 | 69.7% | 0.5551 | 70.7% | 0.5537 | 70.7% |
| I | 1,671 | 0.6153 | 63.9% | 0.6068 | 66.0% | 0.6074 | 65.6% |
| O | 62 | 0.6063 | 66.1% | 0.6200 | 64.5% | 0.6098 | 61.3% |
| P | 1,106 | 0.6087 | 65.8% | 0.6007 | 67.2% | 0.5987 | 67.7% |
| PM | 1,040 | 0.5960 | 67.6% | 0.5898 | 68.5% | 0.5929 | 67.4% |

## Test set by best of

| best_of | n | Elo log-loss | Elo accuracy | Logistic log-loss | Logistic accuracy | XGBoost log-loss | XGBoost accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 4,916 | 0.5991 | 66.3% | 0.5918 | 67.7% | 0.5916 | 67.4% |

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
  "best_iteration": 123,
  "calibration_slope": 1.1037,
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
      "best_iteration": 191,
      "val_log_loss": 0.5932451418003686
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
      "best_iteration": 191,
      "val_log_loss": 0.5930599277561319
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
      "best_iteration": 123,
      "val_log_loss": 0.5927040283233618
    }
  ]
}
```
