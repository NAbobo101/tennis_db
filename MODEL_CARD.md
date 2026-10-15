# Model card: tennis-predictor 1.0

Format inspired by [Mitchell et al. (2019), *Model Cards for Model Reporting*](https://arxiv.org/abs/1810.03993).

## Model details

| | |
|---|---|
| Task | Probability that player A beats player B in a professional singles match, before it starts |
| Models | XGBoost (served), plus a logistic regression and rank/Elo baselines reported for comparison |
| Inputs | 37 pre-match features: Elo, ranking, serve/return rates, form, head-to-head, activity, biography, context ([feature table](docs/methodology.md#5-features)) |
| Output | A calibrated, symmetric probability; the Elo-only probability; the five largest TreeSHAP contributions |
| Versions | One model per tour (ATP, WTA), version 1.0.0, feature schema 1 |
| Training data | Jeff Sackmann's tennis_atp / tennis_wta, 1968-2024, cut off at 2024-12-16 |
| License | Code MIT; the data and every model and report derived from it are CC BY-NC-SA 4.0 |

## Intended use

- Exploring and explaining matchups ("how does surface or format change this matchup?").
- Teaching and demonstrating leakage-free sports forecasting: chronological splits, calibration,
  baselines.
- A starting point for research, e.g. adding odds, new ratings or new features.

## Out-of-scope use

- **Betting or any commercial use.** The data license is non-commercial. On comparable samples
  bookmaker odds are 2-4 accuracy points better than this model.
- **Decisions about people,** such as selection, contracts or anything affecting a player's
  career.
- **Live or in-play prediction.** Only pre-match information is modelled.
- **Matches after the data cutoff without retraining.** Ratings, form and ranks are frozen at
  2024-12-16. The CLI warns when the prediction date or a player's last match is far from it.

## Training and evaluation data

| | ATP | WTA |
|---|---|---|
| Rating history | 916,628 matches, all tiers, 1968-2024 | 740,537 matches, all tiers, 1968-2024 |
| Training rows (tour-level main draws) | 70,947 (1995-2020) | 27,456 (2009-2020) |
| Validation (early stopping, selection, calibration) | 5,099 (2021-2022) | 4,675 (2021-2022) |
| Test (used once) | 5,355 (2023-2024) | 4,916 (2023-2024) |

Rows are completed matches. Retirements and walkovers update the ratings but are never targets.

## Metrics (test set)

| Model | ATP log-loss | ATP Brier | ATP accuracy | ATP ECE | WTA log-loss | WTA Brier | WTA accuracy | WTA ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Elo baseline | 0.6045 | 0.2098 | 65.6% | 0.032 | 0.5991 | 0.2072 | 66.3% | 0.018 |
| Logistic regression | 0.5915 | 0.2044 | 66.6% | 0.026 | 0.5918 | 0.2040 | 67.7% | 0.014 |
| **XGBoost (served)** | 0.5949 | 0.2058 | 65.8% | 0.031 | 0.5916 | 0.2040 | 67.4% | 0.011 |

- The served model was chosen on validation, where XGBoost and logistic regression were tied.
- On the ATP test years the logistic regression is significantly better: log-loss +0.0034
  [+0.0016, +0.0054] for XGBoost.
- Full breakdowns are in [`reports/`](reports).

## Limitations and biases

- **Data coverage is uneven.** Serve statistics exist for ~99% of recent ATP tour matches but
  are sparse for the WTA before ~2009 and absent for most ITF/Futures matches. Players who come
  mostly from the lower tiers rely on Elo and shrunk priors.
- **Dates.** Matches carry the tournament Monday, not the match day. Rest and fatigue are
  approximations.
- **Unobserved factors.** Injuries, illness, motivation, weather and travel are absent. Expect
  the largest errors around comebacks and withdrawals.
- **Players with few matches** (qualifiers, young players) have uncertain ratings. The
  `elo_matches_log_*` features expose this to the model, but individual predictions remain noisy.
- **Calibration** was fitted on 2021-2022 and may drift as the game evolves. Retrain with
  `make reproduce` after updating the data.

## Ethical considerations

The model describes public sports results. It should not be used to profile individuals or to
facilitate gambling. The predictions are probabilities: a 70% favourite loses three times out of
ten.
