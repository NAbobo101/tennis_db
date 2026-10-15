"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

Tour = Literal["atp", "wta"]
Tier = Literal["main", "qual_chall", "futures", "qual_itf"]

TIERS_BY_TOUR: dict[str, frozenset[str]] = {
    "atp": frozenset({"main", "qual_chall", "futures"}),
    "wta": frozenset({"main", "qual_itf"}),
}


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DataConfig(_Section):
    """Where the raw data lives and which files feed the rating history."""

    raw_dir: Path = Path("data/raw")
    cache_dir: Path = Path("data/interim")
    history_tiers: tuple[Tier, ...] = ("main",)
    first_year: int = 1968
    last_year: int = 2024
    exclude_tourneys: str = Field(
        default=r"(?i)laver cup|next ?gen",
        description="Regex on tourney_name for exhibition-like events removed everywhere.",
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.first_year > self.last_year:
            raise ValueError("data.first_year must be <= data.last_year")
        if "main" not in self.history_tiers:
            raise ValueError("data.history_tiers must include 'main'")
        return self


class EloConfig(_Section):
    """FiveThirtyEight-style tennis Elo (Kovalchik, 2016)."""

    initial: float = 1500.0
    k_numerator: PositiveFloat = 250.0
    k_offset: PositiveFloat = 5.0
    k_exponent: PositiveFloat = 0.4
    slam_multiplier: PositiveFloat = 1.1
    retirement_multiplier: float = Field(default=0.5, ge=0.0, le=1.0)
    surface_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class FeatureConfig(_Section):
    """Hyper-parameters of the pre-match feature tracker."""

    stats_half_life_days: PositiveFloat = 365.0
    form_half_life_days: PositiveFloat = 120.0
    stats_prior_matches: PositiveFloat = 2.0
    form_prior_matches: PositiveFloat = 3.0
    h2h_prior: PositiveFloat = 2.0
    unranked_rank: PositiveFloat = 2000.0
    rest_cap_days: PositiveInt = 365
    recent_window_days: PositiveInt = 28


class SplitConfig(_Section):
    """Chronological train / validation / test split, by calendar year."""

    train_start: int
    train_end: int = 2020
    val_years: tuple[int, int] = (2021, 2022)
    test_years: tuple[int, int] = (2023, 2024)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        bounds = [
            self.train_start,
            self.train_end,
            self.val_years[0],
            self.val_years[1],
            self.test_years[0],
            self.test_years[1],
        ]
        if bounds != sorted(bounds):
            raise ValueError("split years must be ordered: train <= val <= test")
        if not (self.train_end < self.val_years[0] and self.val_years[1] < self.test_years[0]):
            raise ValueError("train, validation and test years must not overlap")
        return self

    def split_of(self, year: int) -> str | None:
        """Return ``"train"``, ``"val"``, ``"test"`` or ``None`` for a calendar year."""
        if self.train_start <= year <= self.train_end:
            return "train"
        if self.val_years[0] <= year <= self.val_years[1]:
            return "val"
        if self.test_years[0] <= year <= self.test_years[1]:
            return "test"
        return None


class XGBoostConfig(_Section):
    """Gradient boosting hyper-parameters plus an optional small search grid."""

    n_estimators: PositiveInt = 2000
    learning_rate: PositiveFloat = 0.03
    max_depth: PositiveInt = 3
    min_child_weight: PositiveFloat = 10.0
    subsample: float = Field(default=0.8, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=0.8, gt=0.0, le=1.0)
    reg_lambda: float = Field(default=1.0, ge=0.0)
    early_stopping_rounds: PositiveInt = 100
    n_jobs: PositiveInt = 4
    augment_swapped: bool = Field(
        default=True,
        description="Also train on the player-swapped copy of every training row.",
    )
    grid: tuple[dict[str, float], ...] = Field(
        default=(),
        description="Overrides of the base parameters; the best one on validation log-loss wins.",
    )

    def candidates(self) -> list[dict[str, float]]:
        """Return the parameter sets to try (the base configuration when the grid is empty)."""
        base: dict[str, float] = {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
        }
        if not self.grid:
            return [base]
        unknown = {key for override in self.grid for key in override} - base.keys()
        if unknown:
            raise ValueError(f"unknown xgboost grid parameters: {sorted(unknown)}")
        return [{**base, **override} for override in self.grid]


class Settings(_Section):
    """Complete configuration for one tour."""

    tour: Tour
    seed: int = 42
    levels: frozenset[str]
    data: DataConfig = DataConfig()
    elo: EloConfig = EloConfig()
    features: FeatureConfig = FeatureConfig()
    split: SplitConfig
    xgboost: XGBoostConfig = XGBoostConfig()
    logistic_c: PositiveFloat = 1.0
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")

    @model_validator(mode="after")
    def _check_tiers(self) -> Self:
        invalid = set(self.data.history_tiers) - TIERS_BY_TOUR[self.tour]
        if invalid:
            raise ValueError(f"tiers {sorted(invalid)} do not exist for tour {self.tour!r}")
        if self.split.train_start < self.data.first_year:
            raise ValueError("split.train_start must not precede data.first_year")
        if self.split.test_years[1] > self.data.last_year:
            raise ValueError("split.test_years must not exceed data.last_year")
        return self

    @property
    def model_dir(self) -> Path:
        """Directory holding the trained artifacts of this tour."""
        return self.models_dir / self.tour

    @property
    def report_dir(self) -> Path:
        """Directory holding the evaluation report of this tour."""
        return self.reports_dir / self.tour


def load_settings(path: Path) -> Settings:
    """Load and validate a YAML configuration file."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return Settings.model_validate(raw)
