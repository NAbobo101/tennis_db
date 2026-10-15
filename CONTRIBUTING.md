# Contributing

Thanks for taking the time to improve this project!

## Development setup

You need [uv](https://docs.astral.sh/uv/) (it installs the right Python version for you).

```bash
git clone https://github.com/NAbobo101/tennis_db.git
cd tennis_db
make install        # uv sync --locked + pre-commit hooks
make check          # ruff, mypy --strict and the fast test suite
```

Useful targets (`make help` lists them all):

| Command | What it does |
|---|---|
| `make format` | Auto-fix lint findings and format the code |
| `make test` | Fast tests on synthetic data, with coverage |
| `make test-slow` | Integration tests on the full vendored dataset (~1.5 min) |
| `make reproduce` | Retrain both tours and regenerate `reports/` (~3 min) |

## Guidelines

- **No leakage, ever.** Any new feature must be computed inside
  `FeatureTracker.snapshot` from state built by earlier matches only, and must be registered in
  `features/specs.py` with its swap behaviour. `tests/test_leakage.py` and
  `tests/test_tracker.py::test_declared_symmetry_holds` must keep passing.
- **Measure, then claim.** Model changes are judged on validation log-loss; the test years
  (2023-2024) are only for the final report. Commit the regenerated `reports/` with the change.
- Keep `mypy --strict` and `ruff` clean, and add tests for new behaviour (CI enforces 85% coverage).
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages,
  for example `feat(features): add fatigue from match minutes`.

## Data

The CSV files in `data/raw/` are third-party data (CC BY-NC-SA 4.0, Jeff Sackmann). Please
report data errors upstream in the [original repositories](data/raw/README.md) instead of editing
the files here.
