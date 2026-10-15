from __future__ import annotations

import datetime as dt

import pytest

from tennis_predictor.players import (
    AmbiguousPlayerError,
    PlayerNotFoundError,
    PlayerSummary,
    normalize_name,
    resolve_player,
    search_players,
)

TODAY = dt.date(2024, 12, 16)


def player(player_id: int, name: str, last: dt.date | None = TODAY) -> PlayerSummary:
    return PlayerSummary(player_id, name, "ITA", 2000.0, 100, 10.0, last)


PLAYERS = [
    player(1, "Jannik Sinner"),
    player(2, "Martin Sinner", dt.date(2009, 7, 13)),
    player(3, "Alexander Zverev"),
    player(4, "Mischa Zverev", dt.date(2023, 2, 20)),
    player(5, "Gaël Monfils"),
    player(6, "Ana Twin"),
    player(7, "Bea Twin"),
    player(8, "Same Name"),
    player(9, "Same Name"),
]


def test_normalize_name() -> None:
    assert normalize_name("  Gaël  MONFILS ") == "gael monfils"
    assert normalize_name("Roger-Vasselin") == "roger vasselin"


def test_search_ranks_by_score_then_recency() -> None:
    results = search_players(PLAYERS, "sinner")
    assert [p.player_id for p, _ in results[:2]] == [1, 2]
    assert results[0][1] >= results[1][1]


def test_resolve_by_id_and_exact_name() -> None:
    assert resolve_player(PLAYERS, "3").name == "Alexander Zverev"
    assert resolve_player(PLAYERS, "gael monfils").player_id == 5
    assert resolve_player(PLAYERS, "Monfils").player_id == 5


def test_active_player_wins_over_retired_namesake() -> None:
    assert resolve_player(PLAYERS, "Sinner", reference_date=TODAY).player_id == 1
    assert resolve_player(PLAYERS, "Zverev", reference_date=TODAY).player_id == 3


def test_ambiguity_is_reported() -> None:
    with pytest.raises(AmbiguousPlayerError) as error:
        resolve_player(PLAYERS, "Sinner")  # no reference date: refuse to guess
    assert {p.player_id for p in error.value.candidates} >= {1, 2}
    with pytest.raises(AmbiguousPlayerError):
        resolve_player(PLAYERS, "Twin", reference_date=TODAY)  # two active namesakes
    with pytest.raises(AmbiguousPlayerError, match="id 8"):
        resolve_player(PLAYERS, "Same Name")


def test_unknown_players() -> None:
    with pytest.raises(PlayerNotFoundError):
        resolve_player(PLAYERS, "999")
    with pytest.raises(PlayerNotFoundError):
        resolve_player(PLAYERS, "Qwxz Plkj")


def test_label() -> None:
    assert PLAYERS[0].label() == "Jannik Sinner (id 1, ITA, last match 2024-12-16)"
