from __future__ import annotations

import pytest

from tennis_predictor.config import EloConfig
from tennis_predictor.features.elo import (
    Rating,
    best_of_five_from_best_of_three,
    k_factor,
    match_win_probability,
    set_probability,
    update_ratings,
    win_probability,
)

CFG = EloConfig()


def test_win_probability() -> None:
    assert win_probability(0.0) == pytest.approx(0.5)
    assert win_probability(400.0) == pytest.approx(10 / 11)
    assert win_probability(150.0) + win_probability(-150.0) == pytest.approx(1.0)


def test_best_of_five_matches_tennis_abstract_example() -> None:
    # Nadal vs Shapovalov (Sackmann, 2019): 81.4% in best-of-three -> 86.7% in best-of-five.
    assert best_of_five_from_best_of_three(0.814) == pytest.approx(0.867, abs=1e-3)


@pytest.mark.parametrize("p3", [0.0, 0.1, 0.35, 0.5, 0.77, 0.99, 1.0])
def test_best_of_five_properties(p3: float) -> None:
    p5 = best_of_five_from_best_of_three(p3)
    assert 0.0 <= p5 <= 1.0
    assert p5 + best_of_five_from_best_of_three(1 - p3) == pytest.approx(1.0)
    if p3 > 0.5:
        assert p5 >= p3  # the better player benefits from longer matches


@pytest.mark.parametrize("p", [0.2, 0.5, 0.63, 0.9])
def test_set_probability_inverts_best_of_three(p: float) -> None:
    p3 = p * p * (3 - 2 * p)
    assert set_probability(p3) == pytest.approx(p)


def test_match_win_probability_uses_format() -> None:
    assert match_win_probability(100.0, 3) == pytest.approx(win_probability(100.0))
    assert match_win_probability(100.0, 5) > match_win_probability(100.0, 3)


def test_k_factor_decreases_with_experience() -> None:
    assert k_factor(0, CFG) == pytest.approx(250 / 5**0.4)
    assert k_factor(100, CFG) < k_factor(10, CFG) < k_factor(0, CFG)


def test_update_is_zero_sum_for_equal_experience() -> None:
    winner, loser = Rating(1600.0, 20), Rating(1500.0, 20)
    update_ratings(winner, loser, best_of=3, k_multiplier=1.0, cfg=CFG)
    assert winner.value + loser.value == pytest.approx(3100.0)
    assert (winner.matches, loser.matches) == (21, 21)


def test_upsets_move_ratings_more() -> None:
    favourite, underdog = Rating(1800.0, 50), Rating(1500.0, 50)
    expected_win = Rating(1800.0, 50), Rating(1500.0, 50)
    update_ratings(underdog, favourite, best_of=3, k_multiplier=1.0, cfg=CFG)
    update_ratings(expected_win[0], expected_win[1], best_of=3, k_multiplier=1.0, cfg=CFG)
    assert underdog.value - 1500.0 > expected_win[0].value - 1800.0 > 0


def test_multiplier_scales_update() -> None:
    a, b = Rating(1500.0), Rating(1500.0)
    c, d = Rating(1500.0), Rating(1500.0)
    update_ratings(a, b, best_of=3, k_multiplier=0.5, cfg=CFG)
    update_ratings(c, d, best_of=3, k_multiplier=1.0, cfg=CFG)
    assert a.value - 1500.0 == pytest.approx((c.value - 1500.0) / 2)
