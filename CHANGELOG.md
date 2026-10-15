# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-15

Complete rewrite of the prototype into the `tennis_predictor` package.

### Added

- Typed loader for all singles tiers of the Sackmann data, covering:
  - outcome parsing (retirements, walkovers, defaults, unmarked retirements);
  - statistic plausibility checks;
  - level-code normalization;
  - chronological ordering and a parquet cache.
- Tennis Elo with surface blending and best-of-five conversion.
- Leakage-safe sequential `FeatureTracker` with 37 registered features. Same-round matches are
  treated as simultaneous.
- Chronological train/validation/test split.
- Models: rank and Elo baselines, a symmetric logistic regression, and XGBoost with early
  stopping, a small grid, swap augmentation and temperature scaling.
- Evaluation reports: log-loss, Brier, accuracy, ROC-AUC, ECE, paired bootstrap intervals,
  breakdowns, and reliability and importance figures.
- Serving with official ranks at the data cutoff, tour-specific event levels, TreeSHAP
  explanations and inactivity warnings.
- `tennis` CLI: `train`, `evaluate`, `predict`, `players search`, `ratings`, `data download`.
- Tooling: uv, ruff, mypy `--strict`, pytest with 200+ tests on synthetic data, pre-commit,
  GitHub Actions, Makefile.
- Documentation: README (English and Portuguese), methodology, model card, contributing guide,
  citation file.

### Changed

- Raw data moved from `database/` to `data/raw/<upstream repository>/` with attribution.

### Removed

- `super_ia.py`: its reported 66.7% / 67.5% accuracy came from target leakage and a shuffled
  split. See [the prototype audit](docs/methodology.md#the-prototype-audit).

## [0.1.0] - 2025-07-16

### Added

- Interactive XGBoost prototype (`super_ia.py`) and the Sackmann ATP, WTA and Match Charting
  Project data.

[1.0.0]: https://github.com/NAbobo101/tennis_db/compare/ed98aed...HEAD
[0.1.0]: https://github.com/NAbobo101/tennis_db/commit/ed98aed
