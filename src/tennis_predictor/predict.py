"""Serve pre-match predictions from a trained model."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit

from tennis_predictor.data.schema import LEVEL_ORDINAL, ROUND_ORDER, date_to_day, day_to_date
from tennis_predictor.features.specs import FEATURE_NAMES, swap_players
from tennis_predictor.features.tracker import MatchContext
from tennis_predictor.modeling.artifacts import LoadedArtifacts, load_artifacts
from tennis_predictor.modeling.train import positive_proba, symmetric_proba
from tennis_predictor.players import (
    ACTIVE_DAYS,
    PlayerSummary,
    player_summaries,
    resolve_player,
)


class Event(StrEnum):
    """Kind of tournament."""

    GRAND_SLAM = "grand-slam"
    FINALS = "finals"
    MASTERS = "masters"
    OLYMPICS = "olympics"
    TOUR_500 = "tour-500"
    TOUR = "tour"


#: Upstream level code that each event corresponds to, per tour. ATP 250 and 500 events
#: share the ``A`` code; for the WTA, ``tour`` is a 250 (``I``) and ``tour-500`` is ``P``.
EVENT_LEVEL_CODES: dict[str, dict[Event, str]] = {
    "atp": {
        Event.GRAND_SLAM: "G",
        Event.FINALS: "F",
        Event.MASTERS: "M",
        Event.OLYMPICS: "O",
        Event.TOUR_500: "A",
        Event.TOUR: "A",
    },
    "wta": {
        Event.GRAND_SLAM: "G",
        Event.FINALS: "F",
        Event.MASTERS: "PM",
        Event.OLYMPICS: "O",
        Event.TOUR_500: "P",
        Event.TOUR: "I",
    },
}


@dataclass(frozen=True, slots=True)
class MatchupRequest:
    """A hypothetical match to predict."""

    player_a: str
    player_b: str
    surface: str = "Hard"
    best_of: int | None = None
    event: Event = Event.TOUR
    round: str = "R32"
    as_of: dt.date | None = None


@dataclass(frozen=True, slots=True)
class Prediction:
    """Calibrated win probability of player A plus context for the user."""

    a: PlayerSummary
    b: PlayerSummary
    probability_a: float
    elo_probability_a: float
    as_of: dt.date
    surface: str
    best_of: int
    top_factors: list[tuple[str, float]]
    warnings: list[str]

    @property
    def favourite(self) -> PlayerSummary:
        """The player with the higher win probability."""
        return self.a if self.probability_a >= 0.5 else self.b


class Predictor:
    """Load a trained model once and answer many matchup questions."""

    def __init__(self, artifacts: LoadedArtifacts) -> None:
        self.artifacts = artifacts

    @classmethod
    def from_directory(cls, model_dir: Path) -> Predictor:
        """Load the artifacts written by ``tennis train``."""
        return cls(load_artifacts(model_dir))

    @property
    def data_cutoff(self) -> dt.date:
        """Date of the last tournament included in the model state."""
        return day_to_date(self.artifacts.tracker.last_day)

    @cached_property
    def players(self) -> list[PlayerSummary]:
        """Every player known to the model."""
        return player_summaries(self.artifacts.tracker)

    def _features(
        self, request: MatchupRequest, a: PlayerSummary, b: PlayerSummary
    ) -> tuple[pd.DataFrame, MatchContext, dt.date]:
        tracker = self.artifacts.tracker
        as_of = request.as_of or self.data_cutoff
        if as_of < self.data_cutoff:
            raise ValueError(
                f"as_of {as_of} is before the data cutoff {self.data_cutoff}: the model state "
                "already contains later matches"
            )
        surface = request.surface.strip().capitalize()
        if surface not in {"Hard", "Clay", "Grass"}:
            raise ValueError(f"unknown surface {request.surface!r}; use hard, clay or grass")
        round_code = request.round.strip().upper()
        if round_code not in ROUND_ORDER:
            raise ValueError(
                f"unknown round {request.round!r}; use one of {', '.join(ROUND_ORDER)}"
            )
        is_slam = request.event is Event.GRAND_SLAM
        best_of = request.best_of or (5 if is_slam and tracker.tour == "atp" else 3)
        if best_of not in {3, 5}:
            raise ValueError("best_of must be 3 or 5")
        day = date_to_day(as_of)
        context = MatchContext(
            day=day,
            tourney_id="__prediction__",
            surface=surface,
            best_of=best_of,
            level_ordinal=LEVEL_ORDINAL[tracker.tour][
                EVENT_LEVEL_CODES[tracker.tour][request.event]
            ],
            round_order=ROUND_ORDER[round_code],
            is_slam=is_slam,
        )
        row = tracker.snapshot(
            tracker.player_info(a.player_id, as_of_day=day),
            tracker.player_info(b.player_id, as_of_day=day),
            context,
        )
        return pd.DataFrame([row], columns=list(FEATURE_NAMES), dtype="float64"), context, as_of

    def _top_factors(self, X: pd.DataFrame, count: int = 5) -> list[tuple[str, float]]:
        """Largest per-feature contributions (TreeSHAP) to A's calibrated log-odds.

        Contributions are computed over the trees kept by early stopping, averaged over
        both player orders and multiplied by the calibration slope.
        """
        model = self.artifacts.model
        booster = model.get_booster()
        trees = (0, model.best_iteration + 1)
        direct = booster.predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=trees)
        swapped = booster.predict(
            xgb.DMatrix(swap_players(X)), pred_contribs=True, iteration_range=trees
        )
        contributions = 0.5 * (direct[0, :-1] - swapped[0, :-1]) * self.artifacts.calibrator.slope
        order = np.argsort(-np.abs(contributions))[:count]
        return [(FEATURE_NAMES[index], float(contributions[index])) for index in order]

    def predict(self, request: MatchupRequest) -> Prediction:
        """Win probability of ``request.player_a`` against ``request.player_b``."""
        a = resolve_player(self.players, request.player_a, reference_date=self.data_cutoff)
        b = resolve_player(self.players, request.player_b, reference_date=self.data_cutoff)
        if a.player_id == b.player_id:
            raise ValueError(f"both names resolve to the same player: {a.label()}")
        X, context, as_of = self._features(request, a, b)
        raw = symmetric_proba(positive_proba(self.artifacts.model), X)
        probability = float(self.artifacts.calibrator.transform(raw)[0])
        warnings = [
            f"{player.name} has not played since {player.last_played} (form and ratings are stale)"
            for player in (a, b)
            if player.last_played is None or (as_of - player.last_played).days > ACTIVE_DAYS
        ]
        return Prediction(
            a=a,
            b=b,
            probability_a=probability,
            elo_probability_a=float(expit(X["elo_blend_logit"].iloc[0])),
            as_of=as_of,
            surface=context.surface,
            best_of=context.best_of,
            top_factors=self._top_factors(X),
            warnings=warnings,
        )
