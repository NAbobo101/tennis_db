from __future__ import annotations

import json

from tennis_predictor.config import Settings
from tennis_predictor.evaluation.report import MODELS


def test_report_files(trained_settings: Settings) -> None:
    out = trained_settings.report_dir
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(payload["metrics"]) == {"val", "test"}
    assert set(payload["metrics"]["test"]) == set(MODELS.values())
    assert payload["row_counts"]["test"] == payload["metrics"]["test"]["Elo baseline"]["n"]
    assert set(payload["breakdowns"]) == {"Surface", "Level", "Best of"}
    assert payload["feature_importance_gain"]

    markdown = (out / "metrics.md").read_text(encoding="utf-8")
    assert markdown.startswith("# ATP evaluation report")
    assert "| XGBoost (calibrated) |" in markdown
    assert "paired bootstrap" in markdown
    for name in ("reliability", "feature_importance"):
        for theme in ("light", "dark"):
            figure = out / "figures" / f"{name}_{theme}.png"
            assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
