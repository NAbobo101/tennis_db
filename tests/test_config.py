from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tennis_predictor.config import Settings, SplitConfig, XGBoostConfig, load_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("tour", ["atp", "wta"])
def test_shipped_configs_are_valid(tour: str) -> None:
    settings = load_settings(ROOT / "configs" / f"{tour}.yaml")
    assert settings.tour == tour
    assert "main" in settings.data.history_tiers
    assert settings.model_dir == Path("models") / tour
    assert settings.report_dir == Path("reports") / tour


def test_split_years_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="ordered"):
        SplitConfig(
            train_start=2010, train_end=2022, val_years=(2021, 2022), test_years=(2023, 2024)
        )


def test_split_years_must_not_overlap() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        SplitConfig(
            train_start=2010, train_end=2021, val_years=(2021, 2022), test_years=(2023, 2024)
        )


def test_split_of_assigns_years() -> None:
    split = SplitConfig(train_start=2010)
    assert [split.split_of(year) for year in (2009, 2010, 2020, 2021, 2022, 2023, 2024, 2025)] == [
        None, "train", "train", "val", "val", "test", "test", None,
    ]  # fmt: skip


def test_tiers_must_exist_for_the_tour() -> None:
    with pytest.raises(ValidationError, match="do not exist"):
        Settings.model_validate(
            {
                "tour": "wta",
                "levels": ["G"],
                "data": {"history_tiers": ["main", "futures"]},
                "split": {"train_start": 2010},
            }
        )


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {"tour": "atp", "levels": ["G"], "split": {"train_start": 2010}, "typo": 1}
        )


def test_grid_candidates() -> None:
    cfg = XGBoostConfig(grid=({"max_depth": 5}, {"learning_rate": 0.1}))
    candidates = cfg.candidates()
    assert [c["max_depth"] for c in candidates] == [5, 3]
    assert [c["learning_rate"] for c in candidates] == [0.03, 0.1]
    assert XGBoostConfig().candidates() == [XGBoostConfig().candidates()[0]]


def test_grid_rejects_unknown_parameters() -> None:
    with pytest.raises(ValueError, match="unknown xgboost grid parameters"):
        XGBoostConfig(grid=({"depth": 3},)).candidates()


def test_load_settings_requires_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_settings(path)
