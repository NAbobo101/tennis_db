"""The served prediction must equal what the training pipeline computes for the same match."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from tennis_predictor.config import Settings
from tennis_predictor.data.schema import LEVEL_ORDINAL, day_to_date
from tennis_predictor.features.dataset import Dataset, build_dataset
from tennis_predictor.features.specs import FEATURE_NAMES, swap_players
from tennis_predictor.modeling.artifacts import LoadedArtifacts, load_artifacts
from tennis_predictor.modeling.train import positive_proba, symmetric_proba
from tennis_predictor.predict import EVENT_LEVEL_CODES, Event, MatchupRequest, Predictor
from tests.synthetic import RETIRED_PLAYER

EVENTS = {"G": Event.GRAND_SLAM, "M": Event.MASTERS, "A": Event.TOUR, "F": Event.FINALS}


@pytest.fixture(scope="module")
def artifacts(trained_settings: Settings) -> LoadedArtifacts:
    return load_artifacts(trained_settings.model_dir)


def _final_rows(dataset: Dataset) -> list[int]:
    finals = dataset.meta.index[(dataset.meta["round"] == "F") & (dataset.meta["split"] == "test")]
    return [int(i) for i in finals[:6]]


def test_served_features_and_probability_match_training(
    matches: pd.DataFrame, settings: Settings, dataset: Dataset, artifacts: LoadedArtifacts
) -> None:
    """Serving a past final from the state just before it reproduces its training row."""
    checked = 0
    for row in _final_rows(dataset):
        meta = dataset.meta.iloc[row]
        position = int(
            matches.index[
                (matches["tourney_id"] == meta["tourney_id"]) & (matches["round"] == "F")
            ][0]
        )
        _, tracker = build_dataset(matches.iloc[:position], settings)
        served = Predictor(
            LoadedArtifacts(artifacts.model, artifacts.calibrator, tracker, artifacts.metadata)
        )
        request = MatchupRequest(
            str(meta["a_id"]),
            str(meta["b_id"]),
            surface=str(meta["surface"]),
            best_of=int(meta["best_of"]),
            event=EVENTS[str(meta["tourney_level"])],
            round="F",
            as_of=day_to_date(int(meta["day"])),
        )
        a = next(p for p in served.players if p.player_id == meta["a_id"])
        b = next(p for p in served.players if p.player_id == meta["b_id"])
        X, _, _ = served._features(request, a, b)
        expected = dataset.X.iloc[[row]].reset_index(drop=True)
        # The predictor knows nothing about the event in progress (games already played).
        columns = [
            name
            for name in FEATURE_NAMES
            if name not in {"tourney_games_diff", "qualifier_diff", "wildcard_diff"}
        ]
        pd.testing.assert_frame_equal(X[columns], expected[columns])

        result = served.predict(request)
        manual = artifacts.calibrator.transform(
            symmetric_proba(positive_proba(artifacts.model), X)
        )[0]
        assert result.probability_a == pytest.approx(manual)
        checked += 1
    assert checked >= 3


def test_explanations_add_up_to_the_calibrated_log_odds(artifacts: LoadedArtifacts) -> None:
    predictor = Predictor(artifacts)
    a, b = predictor.players[0], predictor.players[1]
    X, _, _ = predictor._features(MatchupRequest(str(a.player_id), str(b.player_id)), a, b)
    model = artifacts.model
    trees = (0, model.best_iteration + 1)
    booster = model.get_booster()
    margin = booster.predict(xgb.DMatrix(X), output_margin=True, iteration_range=trees)[0]
    swapped = booster.predict(
        xgb.DMatrix(swap_players(X)), output_margin=True, iteration_range=trees
    )[0]
    contributions = booster.predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=trees)[0]
    contributions_swapped = booster.predict(
        xgb.DMatrix(swap_players(X)), pred_contribs=True, iteration_range=trees
    )[0]
    total = 0.5 * (contributions[:-1].sum() - contributions_swapped[:-1].sum())
    assert total == pytest.approx(0.5 * (margin - swapped), abs=1e-4)
    factors = dict(predictor._top_factors(X, count=len(FEATURE_NAMES)))
    assert sum(factors.values()) == pytest.approx(total * artifacts.calibrator.slope, abs=1e-4)


def test_as_of_moves_the_state_forward(artifacts: LoadedArtifacts) -> None:
    predictor = Predictor(artifacts)
    a, b = predictor.players[0], predictor.players[1]
    now = predictor.predict(MatchupRequest(str(a.player_id), str(b.player_id)))
    later_date = predictor.data_cutoff + dt.timedelta(days=200)
    later = predictor.predict(MatchupRequest(str(a.player_id), str(b.player_id), as_of=later_date))
    assert now.as_of == predictor.data_cutoff
    assert later.as_of == later_date
    assert later.probability_a != pytest.approx(now.probability_a, abs=1e-9)


def test_inactive_players_are_flagged(artifacts: LoadedArtifacts) -> None:
    predictor = Predictor(artifacts)
    active = max(predictor.players, key=lambda p: p.matches)
    result = predictor.predict(MatchupRequest(str(RETIRED_PLAYER), str(active.player_id)))
    assert len(result.warnings) == 1
    assert "has not played since" in result.warnings[0]


def test_served_ranks_come_from_the_official_list(artifacts: LoadedArtifacts) -> None:
    from tests.synthetic import OFF_LIST_PLAYER

    tracker = artifacts.tracker
    assert np.isnan(tracker.player_info(OFF_LIST_PLAYER, as_of_day=tracker.last_day).rank)
    ranked = [p for p in tracker.players.values() if p.rank == 1.0]
    assert len(ranked) == 1


@pytest.mark.parametrize("tour", ["atp", "wta"])
def test_events_map_to_training_levels(tour: str) -> None:
    for event in Event:
        assert EVENT_LEVEL_CODES[tour][event] in LEVEL_ORDINAL[tour]
    assert LEVEL_ORDINAL[tour][EVENT_LEVEL_CODES[tour][Event.GRAND_SLAM]] == 4


def test_wta_grand_slams_are_best_of_three(wta_settings: Settings) -> None:
    predictor = Predictor.from_directory(wta_settings.model_dir)
    a, b = predictor.players[0], predictor.players[1]
    slam = predictor.predict(
        MatchupRequest(str(a.player_id), str(b.player_id), event=Event.GRAND_SLAM)
    )
    assert slam.best_of == 3
    X, context, _ = predictor._features(
        MatchupRequest(str(a.player_id), str(b.player_id), event=Event.TOUR), a, b
    )
    assert context.level_ordinal == LEVEL_ORDINAL["wta"]["I"] == 0
    assert X["level_ordinal"].iloc[0] == 0
