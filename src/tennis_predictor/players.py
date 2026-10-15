"""Find players by (fuzzy) name and summarize their current state."""

from __future__ import annotations

import datetime as dt
import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from tennis_predictor.data.schema import day_to_date
from tennis_predictor.features.tracker import FeatureTracker

#: Minimum fuzzy score for an unambiguous match, and the margin it needs over the runner-up.
ACCEPT_SCORE = 90.0
ACCEPT_MARGIN = 5.0
#: Among equally good matches, a player active this recently wins over retired namesakes.
ACTIVE_DAYS = 365


@dataclass(frozen=True, slots=True)
class PlayerSummary:
    """Identity and current strength of a player."""

    player_id: int
    name: str
    ioc: str
    elo: float
    matches: int
    rank: float | None
    last_played: dt.date | None

    def label(self) -> str:
        """Short human-readable description used in messages."""
        country = f", {self.ioc}" if self.ioc else ""
        last = f", last match {self.last_played:%Y-%m-%d}" if self.last_played else ""
        return f"{self.name} (id {self.player_id}{country}{last})"


class PlayerNotFoundError(LookupError):
    """No player matches the query."""


class AmbiguousPlayerError(LookupError):
    """Several players match the query; the candidates are attached."""

    def __init__(self, query: str, candidates: Sequence[PlayerSummary]) -> None:
        self.query = query
        self.candidates = list(candidates)
        options = "; ".join(candidate.label() for candidate in self.candidates)
        super().__init__(f"{query!r} is ambiguous: {options}. Use a full name or a player id.")


def normalize_name(name: str) -> str:
    """Lowercase, strip accents and punctuation: ``"Gaël Monfils"`` -> ``"gael monfils"``."""
    decomposed = unicodedata.normalize("NFKD", name)
    letters = "".join(
        char if char.isalpha() else " " for char in decomposed if not unicodedata.combining(char)
    )
    return " ".join(letters.casefold().split())


def player_summaries(tracker: FeatureTracker) -> list[PlayerSummary]:
    """Summaries of every player with at least one rated match."""
    summaries = []
    for player_id, state in tracker.players.items():
        if state.elo.matches == 0:
            continue
        summaries.append(
            PlayerSummary(
                player_id=player_id,
                name=state.name,
                ioc=state.ioc,
                elo=state.elo.value,
                matches=state.elo.matches,
                rank=state.rank if math.isfinite(state.rank) else None,
                last_played=day_to_date(state.last_day) if state.last_day >= 0 else None,
            )
        )
    return summaries


def search_players(
    players: Sequence[PlayerSummary], query: str, *, limit: int = 5, score_cutoff: float = 60.0
) -> list[tuple[PlayerSummary, float]]:
    """Best fuzzy matches for ``query``; equal scores favour the most recently active player."""
    choices = {index: normalize_name(player.name) for index, player in enumerate(players)}
    matches = process.extract(
        normalize_name(query),
        choices,
        scorer=fuzz.WRatio,
        limit=max(limit * 4, limit),
        score_cutoff=score_cutoff,
    )
    ranked = sorted(
        ((players[key], float(score)) for _, score, key in matches),
        key=lambda item: (-round(item[1], 6), -(item[0].last_played or dt.date.min).toordinal()),
    )
    return ranked[:limit]


def resolve_player(
    players: Sequence[PlayerSummary], query: str, *, reference_date: dt.date | None = None
) -> PlayerSummary:
    """Resolve a name or numeric id to exactly one player, refusing to guess.

    A query resolves when it matches one full name exactly, or when its best fuzzy match
    scores at least :data:`ACCEPT_SCORE` and either beats every other candidate by
    :data:`ACCEPT_MARGIN` or is the only such candidate active in the :data:`ACTIVE_DAYS`
    before ``reference_date`` (so "Sinner" means Jannik, not a namesake from 2009).
    """
    text = query.strip()
    if text.isdigit():
        for player in players:
            if player.player_id == int(text):
                return player
        raise PlayerNotFoundError(f"no player with id {text}")

    wanted = normalize_name(text)
    exact = [player for player in players if normalize_name(player.name) == wanted]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise AmbiguousPlayerError(query, exact)

    candidates = search_players(players, text)
    if not candidates:
        raise PlayerNotFoundError(f"no player matches {query!r}")
    best_score = candidates[0][1]
    if best_score >= ACCEPT_SCORE:
        contenders = [player for player, score in candidates if best_score - score < ACCEPT_MARGIN]
        if len(contenders) == 1:
            return contenders[0]
        if reference_date is not None:
            horizon = reference_date - dt.timedelta(days=ACTIVE_DAYS)
            active = [
                p for p in contenders if p.last_played is not None and p.last_played >= horizon
            ]
            if len(active) == 1:
                return active[0]
    raise AmbiguousPlayerError(query, [player for player, _ in candidates])
