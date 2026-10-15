from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from tennis_predictor.config import EloConfig, FeatureConfig
from tennis_predictor.data.schema import Outcome
from tennis_predictor.features.dataset import Dataset
from tennis_predictor.features.specs import FEATURE_NAMES, FEATURES, Symmetry, swap_players
from tennis_predictor.features.tracker import FeatureTracker, MatchContext, MatchResult, PlayerInfo

STATS_W = (5.0, 2.0, 70.0, 42.0, 32.0, 15.0, 10.0, 3.0, 4.0)
STATS_L = (3.0, 4.0, 72.0, 40.0, 26.0, 12.0, 10.0, 5.0, 9.0)


def ctx(
    day: int, tourney: str = "t1", surface: str = "Hard", best_of: int = 3, round_order: int = 9
) -> MatchContext:
    return MatchContext(
        day,
        tourney,
        surface,
        best_of,
        level_ordinal=1,
        round_order=round_order,
        is_slam=best_of == 5,
    )


def info(player_id: int, **kwargs: object) -> PlayerInfo:
    defaults: dict[str, object] = {
        "name": f"P{player_id}",
        "rank": float(player_id),
        "rank_points": 1000.0,
        "age": 25.0,
        "height": 185.0,
    }
    defaults.update(kwargs)
    return PlayerInfo(player_id, **defaults)  # type: ignore[arg-type]


def completed(games: tuple[int, int] = (12, 7)) -> MatchResult:
    return MatchResult(Outcome.COMPLETED, STATS_W, STATS_L, *games)


@pytest.fixture
def tracker() -> FeatureTracker:
    return FeatureTracker(EloConfig(), FeatureConfig(), "atp")


def feature(row: list[float], name: str) -> float:
    return row[FEATURE_NAMES.index(name)]


def test_snapshot_matches_registry(tracker: FeatureTracker) -> None:
    row = tracker.snapshot(info(1), info(2), ctx(1000))
    assert len(row) == len(FEATURE_NAMES) == len(FEATURES)


def test_unknown_players_are_neutral(tracker: FeatureTracker) -> None:
    row = tracker.snapshot(info(1, rank=5.0), info(2, rank=5.0), ctx(1000))
    for name in (
        "elo_diff",
        "surface_elo_diff",
        "elo_blend_logit",
        "form_diff",
        "spw_diff",
        "h2h_score",
    ):
        assert feature(row, name) == pytest.approx(0.0)


def test_update_moves_ratings_form_and_h2h(tracker: FeatureTracker) -> None:
    tracker.update(info(1), info(2), ctx(1000), completed())
    row = tracker.snapshot(info(1), info(2), ctx(1007))
    assert feature(row, "elo_diff") > 0
    assert feature(row, "surface_elo_diff") > 0
    assert feature(row, "form_diff") > 0
    assert feature(row, "h2h_score") == pytest.approx(1 / 3)
    assert feature(row, "h2h_matches_log") == pytest.approx(math.log(2))
    assert feature(row, "spw_diff") > 0
    assert tracker.last_day == 1000


def test_surface_ratings_are_separate(tracker: FeatureTracker) -> None:
    tracker.update(info(1), info(2), ctx(1000, surface="Clay"), completed())
    on_grass = tracker.snapshot(info(1), info(2), ctx(1007, surface="Grass"))
    assert feature(on_grass, "surface_elo_diff") == pytest.approx(0.0)
    assert feature(on_grass, "elo_diff") > 0


def test_walkovers_only_refresh_facts(tracker: FeatureTracker) -> None:
    tracker.update(info(1, rank=3.0), info(2), ctx(1000), MatchResult(Outcome.WALKOVER))
    row = tracker.snapshot(info(1), info(2), ctx(1007))
    assert feature(row, "elo_diff") == 0.0
    assert feature(row, "recent_matches_diff") == 0.0
    assert tracker.players[1].rank == 3.0
    assert tracker.players[1].elo.matches == 0


def test_retirement_counts_with_reduced_k_and_no_stats(tracker: FeatureTracker) -> None:
    full = FeatureTracker(EloConfig(), FeatureConfig(), "atp")
    full.update(info(1), info(2), ctx(1000), completed())
    tracker.update(info(1), info(2), ctx(1000), MatchResult(Outcome.RETIRED, None, None, 8, 5))
    assert tracker.players[1].elo.value - 1500 == pytest.approx(
        (full.players[1].elo.value - 1500) / 2
    )
    assert tracker.players[2].retired_last
    assert not tracker.players[1].retired_last
    assert tracker.global_count == 0
    row = tracker.snapshot(info(1), info(2), ctx(1007))
    assert feature(row, "after_retirement_diff") == -1.0


def test_activity_features(tracker: FeatureTracker) -> None:
    tracker.update(info(1), info(3), ctx(1000, tourney="a", round_order=9), completed((12, 7)))
    same_event = tracker.snapshot(info(1), info(2), ctx(1000, tourney="a", round_order=11))
    assert feature(same_event, "tourney_games_diff") == 19.0
    assert feature(same_event, "rest_days_log_diff") == pytest.approx(
        math.log1p(0) - math.log1p(365)
    )
    assert feature(same_event, "recent_matches_diff") == 1.0
    later = tracker.snapshot(info(1), info(2), ctx(1100, tourney="b"))
    assert feature(later, "tourney_games_diff") == 0.0
    assert feature(later, "recent_matches_diff") == 0.0
    assert feature(later, "rest_days_log_diff") == pytest.approx(math.log1p(100) - math.log1p(365))


def test_rank_features(tracker: FeatureTracker) -> None:
    row = tracker.snapshot(
        info(1, rank=10.0, rank_points=3000.0), info(2, rank=math.nan), ctx(1000)
    )
    assert feature(row, "log_rank_diff") == pytest.approx(math.log(10) - math.log(2000))
    assert feature(row, "unranked_diff") == -1.0
    assert feature(row, "log_points_diff") == pytest.approx(math.log1p(3000))
    missing_points = tracker.snapshot(info(1, rank_points=math.nan), info(2), ctx(1000))
    assert math.isnan(feature(missing_points, "log_points_diff"))


def test_best_of_five_increases_elo_edge(tracker: FeatureTracker) -> None:
    for day in range(1000, 1100, 7):
        tracker.update(info(1), info(2), ctx(day), completed())
    bo3 = feature(tracker.snapshot(info(1), info(2), ctx(1200)), "elo_blend_logit")
    bo5 = feature(tracker.snapshot(info(1), info(2), ctx(1200, best_of=5)), "elo_blend_logit")
    assert bo5 > bo3 > 0


def test_declared_symmetry_holds(built: tuple[Dataset, FeatureTracker]) -> None:
    _dataset, tracker = built
    rng = np.random.default_rng(0)
    ids = list(tracker.players)
    surfaces = ["Hard", "Clay", "Grass", "Unknown"]
    for _ in range(25):
        a, b = (int(x) for x in rng.choice(ids, size=2, replace=False))
        context = ctx(
            tracker.last_day + int(rng.integers(0, 50)),
            surface=surfaces[int(rng.integers(0, 4))],
            best_of=int(rng.choice([3, 5])),
        )
        info_a = tracker.player_info(a, as_of_day=context.day)
        info_b = tracker.player_info(b, as_of_day=context.day)
        forward = tracker.snapshot(info_a, info_b, context)
        backward = tracker.snapshot(info_b, info_a, context)
        for spec, f, g in zip(FEATURES, forward, backward, strict=True):
            if math.isnan(f):
                assert math.isnan(g), spec.name
            elif spec.symmetry is Symmetry.ANTI:
                assert f == pytest.approx(-g, abs=1e-9), spec.name
            else:
                assert f == pytest.approx(g, abs=1e-12), spec.name


def test_swap_players_negates_only_antisymmetric(dataset: Dataset) -> None:
    swapped = swap_players(dataset.X)
    for spec in FEATURES:
        if spec.symmetry is Symmetry.ANTI:
            assert np.allclose(swapped[spec.name], -dataset.X[spec.name], equal_nan=True)
        else:
            assert swapped[spec.name].equals(dataset.X[spec.name])


def test_save_and_load_round_trip(built: tuple[Dataset, FeatureTracker], tmp_path: Path) -> None:
    _, tracker = built
    tracker.save(tmp_path)
    restored = FeatureTracker.load(tmp_path)
    assert restored.last_day == tracker.last_day
    assert restored.h2h == tracker.h2h
    ids = sorted(tracker.players)[:6]
    context = ctx(tracker.last_day + 3, tourney="x")
    for a, b in itertools.pairwise(ids):
        original = tracker.snapshot(
            tracker.player_info(a, as_of_day=context.day),
            tracker.player_info(b, as_of_day=context.day),
            context,
        )
        loaded = restored.snapshot(
            restored.player_info(a, as_of_day=context.day),
            restored.player_info(b, as_of_day=context.day),
            context,
        )
        assert np.allclose(original, loaded, equal_nan=True)


def test_player_info_ages_players(tracker: FeatureTracker) -> None:
    tracker.update(info(1, age=20.0), info(2), ctx(1000), completed())
    assert tracker.player_info(1, as_of_day=1000 + 365).age == pytest.approx(20.0 + 365 / 365.25)
    assert tracker.has_player(1)
    assert not tracker.has_player(99)
