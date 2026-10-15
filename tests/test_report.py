from __future__ import annotations

import json
import re

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from tennis_predictor.config import Settings
from tennis_predictor.evaluation.report import MODELS
from tennis_predictor.modeling.artifacts import load_predictions


def _table(markdown: str, heading: str) -> dict[str, list[str]]:
    section = markdown.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| ") and "---" not in line]
    return {
        cells[0]: cells[1:]
        for cells in ([c.strip() for c in row.strip("|").split("|")] for row in rows[1:])
    }


def test_report_files(trained_settings: Settings) -> None:
    out = trained_settings.report_dir
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(payload["metrics"]) == {"val", "test"}
    assert set(payload["metrics"]["test"]) == set(MODELS.values())
    assert payload["row_counts"]["test"] == payload["metrics"]["test"]["Elo baseline"]["n"]
    assert set(payload["breakdowns"]) == {"Surface", "Level", "Best of"}
    assert payload["feature_importance_gain"]
    for name in ("reliability", "feature_importance"):
        for theme in ("light", "dark"):
            figure = out / "figures" / f"{name}_{theme}.png"
            assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_markdown_numbers_match_the_predictions(trained_settings: Settings) -> None:
    out = trained_settings.report_dir
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    markdown = (out / "metrics.md").read_text(encoding="utf-8")
    predictions = load_predictions(trained_settings.model_dir)
    test = predictions.loc[predictions["split"] == "test"]

    for heading, split in (("Test set", "test"), ("Validation set", "val")):
        table = _table(markdown, heading)
        losses = {name: values["log_loss"] for name, values in payload["metrics"][split].items()}
        for name, loss in losses.items():
            cell = table[name][0]
            assert cell.strip("*") == f"{loss:.4f}"
            assert cell.startswith("**") == (loss == min(losses.values()))

    def point_loss(p: pd.Series) -> np.ndarray:
        y = test["y"].to_numpy(dtype=float)
        q = p.to_numpy(dtype=float)
        return np.asarray(-(y * np.log(q) + (1 - y) * np.log(1 - q)))

    expected = float(np.mean(point_loss(test["p_xgboost"]) - point_loss(test["p_elo"])))
    reported = payload["bootstrap"]["XGBoost (calibrated) vs Elo baseline"][0]
    assert reported["metric"] == "log_loss"
    assert reported["difference"] == pytest.approx(expected)
    assert "XGBoost (calibrated) vs Logistic regression" in payload["bootstrap"]

    for breakdown in payload["breakdowns"].values():
        assert sum(row["n"] for row in breakdown) == len(test)
    assert re.search(r"Test set by level", markdown)


def test_importing_the_report_keeps_the_matplotlib_backend() -> None:
    import importlib

    import tennis_predictor.evaluation.report as report

    backend = mpl.get_backend()
    importlib.reload(report)
    assert mpl.get_backend() == backend
