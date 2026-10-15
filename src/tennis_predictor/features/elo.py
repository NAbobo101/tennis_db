"""Tennis Elo ratings (FiveThirtyEight / Kovalchik 2016 formulation).

The rating difference is interpreted as a best-of-three match probability; best-of-five
matches use the probability implied by the same per-set strength.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tennis_predictor.config import EloConfig


def win_probability(rating_diff: float) -> float:
    """Logistic Elo expectation for a rating difference (best-of-three)."""
    return 1.0 / (1.0 + math.pow(10.0, -rating_diff / 400.0))


def set_probability(match_probability: float) -> float:
    """Invert ``P3 = p^2 (3 - 2p)``: the per-set win probability behind a best-of-three probability.

    Writing ``p = 1/2 + x`` turns the cubic into ``cos(3t) = 1 - 2 P3`` with ``x = cos(t)``,
    which has the closed-form root below on ``x in [-1/2, 1/2]``.
    """
    clipped = min(max(match_probability, 0.0), 1.0)
    return 0.5 + math.cos((math.acos(1.0 - 2.0 * clipped) - 2.0 * math.pi) / 3.0)


def best_of_five_from_best_of_three(p3: float) -> float:
    """Convert a best-of-three win probability into the best-of-five equivalent."""
    p = set_probability(p3)
    return p * p * p * (10.0 - 15.0 * p + 6.0 * p * p)


def match_win_probability(rating_diff: float, best_of: int) -> float:
    """Win probability for a rating difference and match format."""
    p3 = win_probability(rating_diff)
    return best_of_five_from_best_of_three(p3) if best_of == 5 else p3


def k_factor(matches_played: int, cfg: EloConfig) -> float:
    """Dynamic K-factor ``K = 250 / (n + 5) ** 0.4`` that shrinks with experience."""
    return cfg.k_numerator / math.pow(matches_played + cfg.k_offset, cfg.k_exponent)


@dataclass(slots=True)
class Rating:
    """A mutable Elo rating with the number of matches it is based on."""

    value: float
    matches: int = 0


def update_ratings(
    winner: Rating, loser: Rating, *, best_of: int, k_multiplier: float, cfg: EloConfig
) -> None:
    """Apply one match result to both ratings in place."""
    surprise = 1.0 - match_win_probability(winner.value - loser.value, best_of)
    winner.value += k_factor(winner.matches, cfg) * k_multiplier * surprise
    loser.value -= k_factor(loser.matches, cfg) * k_multiplier * surprise
    winner.matches += 1
    loser.matches += 1
