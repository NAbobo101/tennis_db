"""Parsing of the free-text ``score`` column."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from tennis_predictor.data.schema import Outcome

_NOISE = re.compile(r"&nbsp;|\+H\d+")
_SPACES = re.compile(r"\s+")
_SET = re.compile(r"^(\d{1,2})-(\d{1,2})(?:\(\d+\))?$")
_MATCH_TIEBREAK = re.compile(r"^\[(\d{1,2})-(\d{1,2})\]$")

_WALKOVER = re.compile(r"\bW[/.]?[O0]\b|WALKOVER")
_DEFAULT = re.compile(r"\bDEF")
_RETIRED = re.compile(r"\bRET|\bRTD|\bRE\b")
_ABANDONED = re.compile(r"\bAB(?:D|N|ANDONED)?\b|UNFINISHED")
_UNKNOWN = re.compile(r"\bUNK")
_STATUS_TOKENS = frozenset(
    {"RE", "RET", "RET.", "RET'D", "RTD", "RTD.", "DEF", "DEF.", "DEFAULT", "AB", "ABD", "ABN"}
)
_DIGIT = re.compile(r"\d")


def normalize_score(raw: object) -> str:
    """Strip HTML entities, injury-timeout markers and redundant whitespace."""
    if not isinstance(raw, str):
        return ""
    return _SPACES.sub(" ", _NOISE.sub(" ", raw)).strip()


@dataclass(frozen=True, slots=True)
class SetScore:
    """Games of one set from the match winner's point of view."""

    winner: int
    loser: int
    match_tiebreak: bool = False

    @property
    def finished(self) -> bool:
        """Whether the set reached a valid conclusion (6-4, 7-5, 7-6, 70-68, [10-8]...)."""
        high, low = max(self.winner, self.loser), min(self.winner, self.loser)
        if self.match_tiebreak:
            return high >= 7 and high - low >= 2
        return (high >= 6 and high - low >= 2) or (high == 7 and low == 6)


def parse_sets(score: str) -> list[SetScore] | None:
    """Parse the sets of a score, ignoring status tokens such as ``RET``.

    Returns ``None`` when any other token cannot be parsed (for example Excel-mangled
    scores like ``7-Oct`` or typos like ``6-1-6-2``).
    """
    sets: list[SetScore] = []
    for token in score.split():
        if token.upper() in _STATUS_TOKENS:
            continue
        if match := _SET.match(token):
            sets.append(SetScore(int(match.group(1)), int(match.group(2))))
        elif match := _MATCH_TIEBREAK.match(token):
            sets.append(SetScore(int(match.group(1)), int(match.group(2)), match_tiebreak=True))
        else:
            return None
    return sets


def count_games(score: str) -> tuple[int, int] | None:
    """Count games won by the winner and the loser.

    A deciding match tie-break such as ``[10-7]`` counts as one game. Returns ``None``
    when the score cannot be parsed.
    """
    sets = parse_sets(score)
    if sets is None:
        return None
    winner = sum(
        1 if s.match_tiebreak and s.winner > s.loser else 0 if s.match_tiebreak else s.winner
        for s in sets
    )
    loser = sum(
        1 if s.match_tiebreak and s.loser > s.winner else 0 if s.match_tiebreak else s.loser
        for s in sets
    )
    return winner, loser


def _looks_unfinished(sets: list[SetScore]) -> bool:
    """A score without a status marker that cannot be a completed match.

    Either its last set is unfinished (``5-7 5-4``) or the winner did not win more sets
    than the loser (``2-6 3-0``): such matches ended with a retirement.
    """
    if not sets:
        return True
    won = sum(s.finished and s.winner > s.loser for s in sets)
    lost = sum(s.finished and s.loser > s.winner for s in sets)
    return not sets[-1].finished or won <= lost


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
    sets = parse_sets(score)
    if _RETIRED.search(upper):
        # Retiring before a single game is played is a walkover in all but name.
        if sets is not None and all(s.winner == 0 and s.loser == 0 for s in sets):
            return Outcome.WALKOVER
        return Outcome.RETIRED
    if sets is None:
        # Scores with typos ("6-1-6-2", "7-Oct") still describe a played match whose
        # game counts are unavailable. Scores without any digit ("BYE") are not matches.
        return Outcome.COMPLETED if _DIGIT.search(upper) else Outcome.UNKNOWN
    return Outcome.RETIRED if _looks_unfinished(sets) else Outcome.COMPLETED


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
