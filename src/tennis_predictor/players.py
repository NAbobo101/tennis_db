"""Find players by name and summarize their current state."""

from __future__ import annotations

import datetime as dt
import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from tennis_predictor.data.schema import day_to_date
from tennis_predictor.features.tracker import FeatureTracker

#: Among equally good matches, players active this recently beat retired namesakes.
ACTIVE_DAYS = 365
#: A candidate wins outright when it has this many times more rated matches than the next.
DOMINANCE = 5.0
#: Minimum similarity for a query token to match a name token despite a typo.
TOKEN_SIMILARITY = 85.0

# Letters that Unicode decomposition does not map to ASCII.
_TRANSLITERATION = str.maketrans(
    {
        "đ": "dj",
        "Đ": "Dj",
        "ł": "l",
        "Ł": "L",
        "ø": "o",
        "Ø": "O",
        "æ": "ae",
        "Æ": "Ae",
        "œ": "oe",
        "ß": "ss",
        "þ": "th",
        "\u0131": "i",  # dotless i
    }
)


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
    """Lowercase ASCII letters only: ``"Novak Đoković"`` -> ``"novak djokovic"``."""
    decomposed = unicodedata.normalize("NFKD", name.translate(_TRANSLITERATION))
    letters = "".join(
        char if char.isalpha() else " " for char in decomposed if not unicodedata.combining(char)
    )
    return " ".join(letters.casefold().encode("ascii", "ignore").decode().split())


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


def _prominence(player: PlayerSummary) -> tuple[int, int]:
    return player.matches, (player.last_played or dt.date.min).toordinal()


def search_players(
    players: Sequence[PlayerSummary], query: str, *, limit: int = 5, score_cutoff: float = 60.0
) -> list[tuple[PlayerSummary, float]]:
    """Best fuzzy matches for ``query``; equal scores favour the most prominent player."""
    choices = {index: normalize_name(player.name) for index, player in enumerate(players)}
    matches = process.extract(
        normalize_name(query), choices, scorer=fuzz.WRatio, limit=None, score_cutoff=score_cutoff
    )
    ranked = sorted(
        ((players[key], float(score)) for _, score, key in matches),
        key=lambda item: (-round(item[1], 6), *(-value for value in _prominence(item[0]))),
    )
    return ranked[:limit]


def _token_level(query: str, name_tokens: Sequence[str]) -> int:
    """0 = equal to a name token, 1 = prefix of one, 2 = close typo, 3 = no match."""
    if query in name_tokens:
        return 0
    if len(query) >= 4 and any(token.startswith(query) for token in name_tokens):
        return 1
    if any(fuzz.ratio(query, token) >= TOKEN_SIMILARITY for token in name_tokens):
        return 2
    return 3


def _token_candidates(players: Sequence[PlayerSummary], query: str) -> list[PlayerSummary]:
    """Players whose name contains every query token, at the strictest matching level.

    Exact token matches beat prefixes, which beat typos ("Vilas" is Guillermo Vilas, not
    "Vila"); for one-token queries, surname matches beat first-name matches.
    """
    query_tokens = normalize_name(query).split()
    if not query_tokens:
        return []
    scored: list[tuple[int, int, PlayerSummary]] = []
    for player in players:
        tokens = normalize_name(player.name).split()
        if not tokens:
            continue
        level = max(_token_level(token, tokens) for token in query_tokens)
        if level < 3:
            surname = len(query_tokens) == 1 and _token_level(query_tokens[0], tokens[-1:]) == level
            scored.append((level, 0 if surname else 1, player))
    if not scored:
        return []
    best = min((level, surname) for level, surname, _ in scored)
    return [player for level, surname, player in scored if (level, surname) == best]


def resolve_player(
    players: Sequence[PlayerSummary], query: str, *, reference_date: dt.date | None = None
) -> PlayerSummary:
    """Resolve a name or numeric id to exactly one player, refusing to guess.

    A query resolves when it is an id, a unique full name, or matches the tokens of
    several names among which one stands out: it is the only one active in the
    :data:`ACTIVE_DAYS` before ``reference_date``, or it has :data:`DOMINANCE` times more
    rated matches than any other candidate. "Sinner" therefore means Jannik (Martin last
    played in 2009) and "Nadal" means Rafael (Joan Nadal Vives has a dozen matches), while
    two comparable namesakes raise :class:`AmbiguousPlayerError`.
    """
    text = query.strip()
    if text.isdigit():
        for player in players:
            if player.player_id == int(text):
                return player
        raise PlayerNotFoundError(f"no player with id {text}")

    wanted = normalize_name(text)
    exact = [player for player in players if normalize_name(player.name) == wanted]
    if len(exact) > 1:
        raise AmbiguousPlayerError(query, sorted(exact, key=_prominence, reverse=True))
    if len(exact) == 1 and len(wanted.split()) > 1:
        return exact[0]

    candidates = _token_candidates(players, text) or exact
    if not candidates:
        suggestions = [player for player, _ in search_players(players, text)]
        if suggestions:
            raise AmbiguousPlayerError(query, suggestions)
        raise PlayerNotFoundError(f"no player matches {query!r}")

    pool = candidates
    if reference_date is not None and len(pool) > 1:
        horizon = reference_date - dt.timedelta(days=ACTIVE_DAYS)
        active = [p for p in pool if p.last_played is not None and p.last_played >= horizon]
        pool = active or pool
    pool = sorted(pool, key=_prominence, reverse=True)
    if len(pool) == 1 or pool[0].matches >= DOMINANCE * max(pool[1].matches, 1):
        return pool[0]
    raise AmbiguousPlayerError(query, pool[:8])
