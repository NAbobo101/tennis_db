"""Registry of model features and their behaviour when the two players are swapped."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import pandas as pd


class Symmetry(StrEnum):
    """How a feature reacts to swapping player A and player B."""

    ANTI = "antisymmetric"  # f(B, A) == -f(A, B)
    SYM = "symmetric"  # f(B, A) == f(A, B)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Name, group, swap behaviour and description of one feature."""

    name: str
    group: str
    symmetry: Symmetry
    description: str


_A, _S = Symmetry.ANTI, Symmetry.SYM

FEATURES: Final[tuple[FeatureSpec, ...]] = (
    FeatureSpec("elo_diff", "elo", _A, "Overall Elo of A minus B"),
    FeatureSpec("surface_elo_diff", "elo", _A, "Surface-specific Elo of A minus B"),
    FeatureSpec(
        "elo_blend_logit",
        "elo",
        _A,
        "Logit of the Elo win probability from the 50/50 overall/surface blend, "
        "adjusted for best-of-5",
    ),
    FeatureSpec("elo_matches_log_diff", "elo", _A, "log1p(rated matches) of A minus B"),
    FeatureSpec(
        "surface_matches_log_diff", "elo", _A, "log1p(rated matches on this surface) of A minus B"
    ),
    FeatureSpec(
        "elo_matches_log_min", "elo", _S, "log1p(rated matches) of the less experienced player"
    ),
    FeatureSpec(
        "log_rank_diff", "ranking", _A, "ln(rank A) - ln(rank B); unranked players get a large rank"
    ),
    FeatureSpec("log_points_diff", "ranking", _A, "log1p(ranking points) of A minus B"),
    FeatureSpec("unranked_diff", "ranking", _A, "1 if only A is unranked, -1 if only B is"),
    FeatureSpec("spw_diff", "serve_return", _A, "Serve points won rate, A minus B"),
    FeatureSpec("rpw_diff", "serve_return", _A, "Return points won rate, A minus B"),
    FeatureSpec("ace_rate_diff", "serve_return", _A, "Aces per serve point, A minus B"),
    FeatureSpec("df_rate_diff", "serve_return", _A, "Double faults per serve point, A minus B"),
    FeatureSpec("first_in_diff", "serve_return", _A, "First serves in per serve point, A minus B"),
    FeatureSpec("first_won_diff", "serve_return", _A, "First-serve points won rate, A minus B"),
    FeatureSpec("second_won_diff", "serve_return", _A, "Second-serve points won rate, A minus B"),
    FeatureSpec("bp_saved_diff", "serve_return", _A, "Break points saved / faced, A minus B"),
    FeatureSpec(
        "bp_converted_diff", "serve_return", _A, "Break points converted / opportunities, A minus B"
    ),
    FeatureSpec(
        "spw_plus_rpw_diff",
        "serve_return",
        _A,
        "(spw + rpw) of A minus B: overall point-winning edge",
    ),
    FeatureSpec(
        "stats_mass_log_min",
        "serve_return",
        _S,
        "log1p(decayed serve points observed), minimum of A and B",
    ),
    FeatureSpec("form_diff", "form", _A, "Time-decayed, shrunk win rate of A minus B"),
    FeatureSpec(
        "h2h_score", "head_to_head", _A, "(wins A over B - wins B over A) / (meetings + prior)"
    ),
    FeatureSpec("h2h_matches_log", "head_to_head", _S, "log1p(previous meetings)"),
    FeatureSpec(
        "rest_days_log_diff",
        "activity",
        _A,
        "log1p(days since previous match, capped) of A minus B",
    ),
    FeatureSpec("recent_matches_diff", "activity", _A, "Matches in the trailing window, A minus B"),
    FeatureSpec(
        "tourney_games_diff", "activity", _A, "Games already played in this event, A minus B"
    ),
    FeatureSpec(
        "after_retirement_diff",
        "activity",
        _A,
        "1 if A retired in their previous match (minus same for B)",
    ),
    FeatureSpec("age_diff", "player", _A, "Age in years, A minus B"),
    FeatureSpec("height_diff", "player", _A, "Height in cm, A minus B"),
    FeatureSpec("left_handed_diff", "player", _A, "Left-handedness indicator, A minus B"),
    FeatureSpec("qualifier_diff", "player", _A, "Qualifier / lucky loser indicator, A minus B"),
    FeatureSpec("wildcard_diff", "player", _A, "Wild card indicator, A minus B"),
    FeatureSpec("best_of_5", "context", _S, "1 for best-of-five matches"),
    FeatureSpec("surface_clay", "context", _S, "1 on clay"),
    FeatureSpec("surface_grass", "context", _S, "1 on grass"),
    FeatureSpec("level_ordinal", "context", _S, "Tournament importance (Grand Slam highest)"),
    FeatureSpec(
        "round_ordinal", "context", _S, "Round of the event (qualifying lowest, final highest)"
    ),
)

FEATURE_NAMES: Final[tuple[str, ...]] = tuple(spec.name for spec in FEATURES)
ANTISYMMETRIC_FEATURES: Final[tuple[str, ...]] = tuple(
    spec.name for spec in FEATURES if spec.symmetry is Symmetry.ANTI
)


def swap_players(X: pd.DataFrame) -> pd.DataFrame:
    """Return the feature matrix seen from the other player's side."""
    swapped = X.copy()
    columns = [name for name in ANTISYMMETRIC_FEATURES if name in swapped.columns]
    swapped[columns] = -swapped[columns]
    return swapped


def features_markdown_table() -> str:
    """Render the feature registry as a Markdown table."""
    lines = ["| Feature | Group | Swap | Description |", "|---|---|---|---|"]
    lines += [
        f"| `{spec.name}` | {spec.group} | {spec.symmetry.value} | {spec.description} |"
        for spec in FEATURES
    ]
    return "\n".join(lines)
