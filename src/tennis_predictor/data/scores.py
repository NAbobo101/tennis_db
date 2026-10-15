"""Parsing of the free-text ``score`` column."""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from tennis_predictor.data.schema import Outcome

_NOISE = re.compile(r"&nbsp;|\+H\d+")
_SPACES = re.compile(r"\s+")
_SET = re.compile(r"^(\d{1,2})-(\d{1,2})(?:\(\d+\))?$")
_MATCH_TIEBREAK = re.compile(r"^\[(\d{1,2})-(\d{1,2})\]$")

_WALKOVER = re.compile(r"\bW[/.]?[O0]\b|WALKOVER")
_DEFAULT = re.compile(r"\bDEF")
_RETIRED = re.compile(r"\bRET|\bRTD")
_ABANDONED = re.compile(r"\bAB(?:D|N|ANDONED)?\b|UNFINISHED")
_UNKNOWN = re.compile(r"\bUNK")
_STATUS_TOKENS = frozenset(
    {"RET", "RET.", "RTD", "RTD.", "DEF", "DEF.", "DEFAULT", "AB", "ABD", "ABN"}
)
_DIGIT = re.compile(r"\d")


def normalize_score(raw: object) -> str:
    """Strip HTML entities, injury-timeout markers and redundant whitespace."""
    if not isinstance(raw, str):
        return ""
    return _SPACES.sub(" ", _NOISE.sub(" ", raw)).strip()


def count_games(score: str) -> tuple[int, int] | None:
    """Count games won by the winner and the loser.

    A deciding match tie-break such as ``[10-7]`` counts as one game. Returns ``None``
    when any token cannot be parsed (for example Excel-mangled scores like ``7-Oct``).
    """
    winner = loser = 0
    for token in score.split():
        if token.upper() in _STATUS_TOKENS:
            continue
        if match := _SET.match(token):
            winner += int(match.group(1))
            loser += int(match.group(2))
        elif match := _MATCH_TIEBREAK.match(token):
            first, second = int(match.group(1)), int(match.group(2))
            winner += first > second
            loser += second > first
        else:
            return None
    return winner, loser


@lru_cache(maxsize=65536)
def classify_outcome(score: str) -> Outcome:
    """Classify a normalized score into an :class:`Outcome`."""
    upper = score.upper()
    if not upper:
        return Outcome.UNKNOWN
    if _WALKOVER.search(upper):
        return Outcome.WALKOVER
    if _DEFAULT.search(upper):
        return Outcome.DEFAULT
    if _ABANDONED.search(upper):
        return Outcome.ABANDONED
    if _UNKNOWN.search(upper):
        return Outcome.UNKNOWN
    if _RETIRED.search(upper):
        return Outcome.RETIRED
    # Scores with typos ("6-1-6-2", "7-Oct") still describe a played match; their game
    # counts are simply unavailable. Scores without any digit ("BYE", ">") are not matches.
    return Outcome.COMPLETED if _DIGIT.search(upper) else Outcome.UNKNOWN


def parse_scores(scores: pd.Series) -> pd.DataFrame:
    """Vectorized wrapper returning ``score``, ``outcome``, ``w_games`` and ``l_games``."""
    unique = pd.unique(scores.to_numpy(dtype=object))
    parsed: dict[object, tuple[str, str, int, int]] = {}
    for raw in unique:
        score = normalize_score(raw)
        games = count_games(score) or (0, 0)
        parsed[raw] = (score, classify_outcome(score).value, games[0], games[1])
    rows = [parsed[raw] for raw in scores.to_numpy(dtype=object)]
    frame = pd.DataFrame(
        rows, columns=["score", "outcome", "w_games", "l_games"], index=scores.index
    )
    return frame.astype({"w_games": "int32", "l_games": "int32"})
