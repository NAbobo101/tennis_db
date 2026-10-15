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


def player_with(
    player_id: int, name: str, matches: int, last: dt.date | None = TODAY
) -> PlayerSummary:
    return PlayerSummary(player_id, name, "USA", 1800.0, matches, None, last)


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
    player(10, "Novak Djokovic"),
]

# Real collisions found in the Sackmann data (names, match counts and last matches).
COLLISIONS = [
    player_with(20, "Collins", 1, dt.date(1973, 5, 5)),
    player_with(21, "Danielle Collins", 406),
    player_with(22, "Kylie Collins", 72),
    player_with(23, "G Mertens", 2, dt.date(1987, 1, 1)),
    player_with(24, "Elise Mertens", 694),
    player_with(25, "Au", 1, dt.date(1972, 1, 1)),
    player_with(26, "Coco Gauff", 320),
    player_with(27, "Guillermo Vilas", 1270, dt.date(1992, 11, 23)),
    player_with(28, "Kenneth Rabinad Vila", 5),
    player_with(29, "Rafael Nadal", 1409),
    player_with(30, "Joan Nadal Vives", 12),
    player_with(31, "Holger Rune", 370),
    player_with(32, "Niklas Grunewald", 40),
    player_with(33, "Tommy Paul", 618),
    player_with(34, "Paul Jubb", 300),  # first-name match only: never preferred over a surname
    player_with(35, "Jakub Paul", 331),
    *(player_with(100 + i, f"Player{i} Wang", 300 - i) for i in range(30)),
]


def test_normalize_name() -> None:
    assert normalize_name("  Gaël  MONFILS ") == "gael monfils"
    assert normalize_name("Roger-Vasselin") == "roger vasselin"
    assert normalize_name("Novak Đoković") == "novak djokovic"
    assert normalize_name("Iga Świątek") == "iga swiatek"


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


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Collins", "Danielle Collins"),  # not the single-token 1973 entry
        ("Mertens", "Elise Mertens"),  # not the initials-only "G Mertens"
        ("Gauff", "Coco Gauff"),  # not "Au", a substring of the name
        ("Vilas", "Guillermo Vilas"),  # an exact surname beats a typo-close one
        ("Nadal", "Rafael Nadal"),  # dominant career beats an active namesake
        ("Rune", "Holger Rune"),  # not "Grunewald"
        ("Tommy Paul", "Tommy Paul"),
        ("Novak Đoković", "Novak Djokovic"),  # transliteration
        ("Djoko", "Novak Djokovic"),  # prefix
        ("Djokovich", "Novak Djokovic"),  # typo
    ],
)
def test_real_world_collisions(query: str, expected: str) -> None:
    assert resolve_player([*PLAYERS, *COLLISIONS], query, reference_date=TODAY).name == expected


def test_comparable_namesakes_stay_ambiguous() -> None:
    with pytest.raises(AmbiguousPlayerError) as error:
        resolve_player(COLLISIONS, "Wang", reference_date=TODAY)  # 30 active namesakes
    assert len(error.value.candidates) == 8
    assert error.value.candidates[0].name == "Player0 Wang"  # most prominent first
    with pytest.raises(AmbiguousPlayerError) as error:
        resolve_player(COLLISIONS, "Paul", reference_date=TODAY)  # Tommy and Jakub Paul
    assert {c.name for c in error.value.candidates} == {"Tommy Paul", "Jakub Paul"}


def test_unmatched_tokens_suggest_instead_of_guessing() -> None:
    with pytest.raises(AmbiguousPlayerError) as error:
        resolve_player(COLLISIONS, "Cori Gauff", reference_date=TODAY)
    assert "Coco Gauff" in {candidate.name for candidate in error.value.candidates}
