"""Synthetic match files in the exact Sackmann CSV format.

Players have a latent skill and winners are drawn from a Bradley-Terry model, so a sane
pipeline must learn something. The files reproduce the upstream quirks the pipeline handles:

- events stored final-first, round-robin stages numbered by player (not chronologically),
  qualifying rounds sharing the main draw's ``tourney_id``;
- retirements, walkovers with fake zero statistics, an exact duplicate row;
- Carpet courts, team events (Davis Cup / BJK Cup, level ``D``), an excluded exhibition;
- Challenger/ITF matches without statistics, unranked players, missing or impossible
  heights, qualifier and wild-card entries;
- a burn-in season before the training window and an official ranking list.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "tourney_date",
    "match_num", "winner_id", "winner_seed", "winner_entry", "winner_name", "winner_hand",
    "winner_ht", "winner_ioc", "winner_age", "loser_id", "loser_seed", "loser_entry",
    "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age", "score", "best_of",
    "round", "minutes", "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon",
    "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced", "winner_rank", "winner_rank_points",
    "loser_rank", "loser_rank_points",
]  # fmt: skip
STAT_NAMES = ("ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced")

SURNAMES = [
    "Archer", "Baxter", "Carver", "Dalton", "Ellison", "Fletcher", "Garner", "Hollis",
    "Irving", "Jarrett", "Keller", "Lawson", "Mercer", "Norris", "Oakley", "Porter",
    "Quinlan", "Rowe", "Sutton", "Turner", "Upton", "Vance", "Walsh", "Yates",
]  # fmt: skip
FIRST_NAMES = ["Adam", "Ben", "Carl", "Dan", "Eli", "Finn", "Gus", "Hugo"]
SURFACE_CYCLE = ["Hard", "Clay", "Grass", "Hard", "Carpet", "Clay"]
KNOCKOUT = ["R16", "QF", "SF", "F"]

#: Player ids with deliberate quirks.
RETIRED_PLAYER = 100004  # "Carl Solo" stops playing after 2016
UNRANKED_PLAYERS = (100006, 100007)  # never ranked
NO_HEIGHT_PLAYER = 100008
BAD_HEIGHT_PLAYER = 100009
OFF_LIST_PLAYER = 100010  # absent from the final official ranking list

TOUR_LAYOUT: dict[str, dict[str, str]] = {
    "atp": {
        "repo": "tennis_atp",
        "main": "atp_matches_{year}.csv",
        "minor": "atp_matches_qual_chall_{year}.csv",
        "rankings": "atp_rankings_current.csv",
        "masters": "M",
        "tour": "A",
        "minor_level": "C",
        "team": "Davis Cup WG",
        "finals": "Tour Finals",
    },
    "wta": {
        "repo": "tennis_wta",
        "main": "wta_matches_{year}.csv",
        "minor": "wta_matches_qual_itf_{year}.csv",
        "rankings": "wta_rankings_current.csv",
        "masters": "PM",
        "tour": "I",
        "minor_level": "25",
        "team": "BJK Cup",
        "finals": "WTA Finals",
    },
}


def _players(n_players: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for index in range(n_players):
        surname = SURNAMES[index % len(SURNAMES)]
        first = FIRST_NAMES[index // len(SURNAMES) % len(FIRST_NAMES)]
        player_id = 100001 + index
        height: object = float(170 + index % 25)
        if player_id == NO_HEIGHT_PLAYER:
            height = ""
        elif player_id == BAD_HEIGHT_PLAYER:
            height = 3.0  # an upstream typo: must become NaN
        rows.append(
            {
                "id": player_id,
                "name": f"{first} {surname}",
                "skill": rng.normal(0.0, 1.0),
                "clay": rng.normal(0.0, 0.3),
                "hand": "L" if index % 7 == 0 else "R",
                "ht": height,
                "ioc": ["USA", "ESP", "FRA", "ITA"][index % 4],
                "born": 1985.0 + index % 15,
            }
        )
    # Namesakes for the name-resolution tests: two active "Twin"s, one retired "Solo".
    rows[1]["name"] = "Ana Twin"
    rows[2]["name"] = "Bea Twin"
    rows[3]["name"] = "Carl Solo"
    rows[4]["name"] = "Dan Solo"
    return pd.DataFrame(rows)


def _serve_stats(rng: np.random.Generator, edge: float, service_games: int) -> list[float]:
    svpt = int(6 * service_games + rng.integers(-3, 4))
    first_in = round(svpt * rng.uniform(0.55, 0.68))
    double_faults = int(rng.integers(0, 5))
    first_won = round(first_in * min(0.9, max(0.5, 0.70 + 0.04 * edge)))
    second_won = round((svpt - first_in - double_faults) * min(0.8, max(0.3, 0.52 + 0.04 * edge)))
    aces = int(rng.integers(0, max(1, first_won // 4)))
    bp_faced = int(rng.integers(0, 10))
    bp_saved = int(rng.integers(0, bp_faced + 1))
    return [
        aces,
        double_faults,
        svpt,
        first_in,
        first_won,
        second_won,
        service_games,
        bp_saved,
        bp_faced,
    ]


def _score(rng: np.random.Generator, best_of: int) -> tuple[str, int]:
    sets_needed = best_of // 2 + 1
    sets: list[str] = []
    lost = 0
    for _ in range(sets_needed):
        if lost < sets_needed - 1 and rng.random() < 0.3:
            sets.append(f"{rng.integers(2, 5)}-6")
            lost += 1
        sets.append(str(rng.choice(["6-3", "6-4", "7-5", "6-2", "7-6(4)"])))
    games = sum(int(a) + int(b.split("(")[0]) for a, b in (s.split("-") for s in sets))
    return " ".join(sets), games


class _Writer:
    """Generates rows for one tour; keeps the shared random state and lookups."""

    def __init__(self, players: pd.DataFrame, rng: np.random.Generator) -> None:
        self.players = players
        self.rng = rng
        self.by_id = {int(row["id"]): row for _, row in players.iterrows()}
        order = players.sort_values("skill", ascending=False)["id"].tolist()
        self.ranks = {int(player_id): position + 1 for position, player_id in enumerate(order)}

    def row(
        self,
        tourney: Mapping[str, object],
        match_num: int,
        round_code: str,
        winner: int,
        loser: int,
        *,
        entries: Mapping[int, str] | None = None,
        with_stats: bool = True,
    ) -> dict[str, object]:
        best_of = int(str(tourney["best_of"]))
        score, games = _score(self.rng, best_of)
        year = int(str(tourney["tourney_date"])[:4])
        edge = float(self.by_id[winner]["skill"] - self.by_id[loser]["skill"])
        row: dict[str, object] = {
            **{
                key: tourney[key]
                for key in (
                    "tourney_id",
                    "tourney_name",
                    "surface",
                    "tourney_level",
                    "tourney_date",
                )
            },
            "draw_size": 16,
            "match_num": match_num,
            "score": score,
            "best_of": best_of,
            "round": round_code,
            "minutes": 30 + 5 * games,
        }
        for side, player_id in (("winner", winner), ("loser", loser)):
            player = self.by_id[player_id]
            ranked = player_id not in UNRANKED_PLAYERS
            row[f"{side}_id"] = player_id
            row[f"{side}_name"] = player["name"]
            row[f"{side}_hand"] = player["hand"]
            row[f"{side}_ht"] = player["ht"]
            row[f"{side}_ioc"] = player["ioc"]
            row[f"{side}_age"] = round(year - float(player["born"]), 1)
            row[f"{side}_rank"] = self.ranks[player_id] if ranked else ""
            row[f"{side}_rank_points"] = (
                max(10, 5000 - 90 * self.ranks[player_id]) if ranked else ""
            )
            row[f"{side}_seed"] = ""
            row[f"{side}_entry"] = (entries or {}).get(player_id, "")
        service_games = max(1, games // 2)
        for prefix, sign in (("w", 1.0), ("l", -1.0)):
            stats = _serve_stats(self.rng, sign * edge, service_games)
            for name, value in zip(STAT_NAMES, stats, strict=True):
                row[f"{prefix}_{name}"] = value if with_stats else ""
        return row

    def play(self, a: int, b: int, surface: str) -> tuple[int, int]:
        def strength(player_id: int) -> float:
            player = self.by_id[player_id]
            return float(player["skill"]) + (float(player["clay"]) if surface == "Clay" else 0.0)

        p_a = 1.0 / (1.0 + math.exp(-(strength(a) - strength(b))))
        return (a, b) if self.rng.random() < p_a else (b, a)

    def knockout_event(
        self, tourney: Mapping[str, object], pool: list[int]
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """A 16-player draw with four qualifiers; returns (main rows, qualifying rows)."""
        drawn = [int(x) for x in self.rng.choice(pool, size=20, replace=False)]
        surface = str(tourney["surface"])
        qualifying, qualifiers = [], []
        for q in range(4):
            winner, loser = self.play(drawn[16 + q], drawn[q], surface)
            qualifying.append(self.row(tourney, 900 + q, "Q1", winner, loser))
            qualifiers.append(winner)
        entries = dict.fromkeys(qualifiers, "Q") | {drawn[4]: "WC"}
        current = qualifiers + drawn[4:16]
        rows, match_num = [], 1
        for round_code in KNOCKOUT:
            next_round = []
            for pair in range(0, len(current), 2):
                winner, loser = self.play(current[pair], current[pair + 1], surface)
                rows.append(
                    self.row(tourney, match_num, round_code, winner, loser, entries=entries)
                )
                next_round.append(winner)
                match_num += 1
            current = next_round
        return rows, qualifying

    def finals_event(
        self, tourney: Mapping[str, object], pool: list[int]
    ) -> list[dict[str, object]]:
        """Eight players, two round-robin groups, then semi-finals and final.

        As upstream, round-robin matches are numbered player by player, which is not the
        order in which they were played.
        """
        best = sorted(pool, key=lambda player: -float(self.by_id[player]["skill"]))[:8]
        groups = [best[0::2], best[1::2]]
        rows, wins, match_num = [], dict.fromkeys(best, 0), 1
        surface = str(tourney["surface"])
        for group in groups:
            for a, b in combinations(group, 2):
                winner, loser = self.play(a, b, surface)
                wins[winner] += 1
                rows.append(self.row(tourney, match_num, "RR", winner, loser))
                match_num += 1
        top = [sorted(group, key=lambda player: -wins[player])[:2] for group in groups]
        finalists = []
        for a, b in ((top[0][0], top[1][1]), (top[1][0], top[0][1])):
            winner, loser = self.play(a, b, surface)
            rows.append(self.row(tourney, match_num, "SF", winner, loser))
            finalists.append(winner)
            match_num += 1
        winner, loser = self.play(finalists[0], finalists[1], surface)
        rows.append(self.row(tourney, match_num, "F", winner, loser))
        return rows


def write_synthetic_tour(
    raw_dir: Path,
    *,
    tour: str = "atp",
    years: range = range(2014, 2025),
    n_players: int = 40,
    events_per_year: int = 12,
    seed: int = 0,
) -> pd.DataFrame:
    """Write yearly main and lower-tier match files plus an official ranking list.

    Returns all written main-draw rows (before any cleaning) for assertions.
    """
    layout = TOUR_LAYOUT[tour]
    rng = np.random.default_rng(seed)
    writer = _Writer(_players(n_players, rng), rng)
    folder = raw_dir / layout["repo"]
    folder.mkdir(parents=True, exist_ok=True)
    everything = []
    last_monday = dt.date(years[0], 1, 1)
    for year in years:
        main_rows: list[dict[str, object]] = []
        minor_rows: list[dict[str, object]] = []
        first_monday = dt.date(year, 1, 1) + dt.timedelta(
            days=(7 - dt.date(year, 1, 1).weekday()) % 7
        )
        pool = [int(p) for p in writer.players["id"] if year <= 2016 or int(p) != RETIRED_PLAYER]
        for event in range(events_per_year):
            date = first_monday + dt.timedelta(weeks=3 * event)
            if event in (0, 6):
                level, best_of = "G", 5 if tour == "atp" else 3
            else:
                level, best_of = (layout["masters"] if event % 3 == 1 else layout["tour"]), 3
            excluded = tour == "atp" and event == events_per_year - 1
            tourney = {
                "tourney_id": f"{year}-{event:03d}",
                "tourney_name": "Next Gen Finals" if excluded else f"Open {event:02d}",
                "surface": SURFACE_CYCLE[event % len(SURFACE_CYCLE)],
                "tourney_level": "F" if excluded else level,
                "tourney_date": date.strftime("%Y%m%d"),
                "best_of": best_of,
            }
            rows, qualifying = writer.knockout_event(tourney, pool)
            if event % 2:  # some events are stored final-first, as in recent upstream files
                rows.reverse()
            main_rows.extend(rows)
            minor_rows.extend(qualifying)

            # A lower-tier event in the same week, without statistics, only feeds ratings.
            minor = {
                **tourney,
                "tourney_id": f"{year}-C{event:02d}",
                "tourney_name": f"Challenger {event}",
                "tourney_level": layout["minor_level"],
                "best_of": 3,
            }
            a, b = (int(x) for x in rng.choice(pool, size=2, replace=False))
            winner, loser = writer.play(a, b, str(tourney["surface"]))
            minor_rows.append(writer.row(minor, 1, "F", winner, loser, with_stats=False))

        # Team cup tie (never a training row) and the season finals with a round robin.
        team_date = first_monday + dt.timedelta(weeks=3 * events_per_year - 2)
        team = {
            "tourney_id": f"{year}-D01",
            "tourney_name": layout["team"],
            "surface": "Hard",
            "tourney_level": "D",
            "tourney_date": team_date.strftime("%Y%m%d"),
            "best_of": 3,
        }
        for match_num in range(1, 5):
            a, b = (int(x) for x in rng.choice(pool, size=2, replace=False))
            winner, loser = writer.play(a, b, "Hard")
            main_rows.append(writer.row(team, match_num, "RR", winner, loser))
        finals_date = first_monday + dt.timedelta(weeks=3 * events_per_year - 1)
        finals = {
            "tourney_id": f"{year}-F01",
            "tourney_name": layout["finals"],
            "surface": "Hard",
            "tourney_level": "F",
            "tourney_date": finals_date.strftime("%Y%m%d"),
            "best_of": 3,
        }
        main_rows.extend(writer.finals_event(finals, pool))
        last_monday = finals_date

        # Quirks: a retirement, a walkover with fake zero stats, and an exact duplicate.
        main_rows[3]["score"] = "6-4 2-1 RET"
        main_rows[5]["score"] = "W/O"
        for key in list(main_rows[5]):
            if key.startswith(("w_", "l_")):
                main_rows[5][key] = 0
        main_rows.append(dict(main_rows[7]))
        frame = pd.DataFrame(main_rows, columns=COLUMNS)
        frame.to_csv(folder / layout["main"].format(year=year), index=False)
        pd.DataFrame(minor_rows, columns=COLUMNS).to_csv(
            folder / layout["minor"].format(year=year), index=False
        )
        everything.append(frame)

    ranked = sorted(
        (player for player in writer.ranks if player not in (*UNRANKED_PLAYERS, OFF_LIST_PLAYER)),
        key=lambda player: writer.ranks[player],
    )
    rankings = pd.DataFrame(
        {
            "ranking_date": int(last_monday.strftime("%Y%m%d")),
            "rank": range(1, len(ranked) + 1),
            "player": ranked,
            "points": [max(10, 6000 - 100 * position) for position in range(len(ranked))],
        }
    )
    rankings.to_csv(folder / layout["rankings"], index=False)
    return pd.concat(everything, ignore_index=True)


def write_settings(path: Path, root: Path, *, tour: str = "atp", **overrides: object) -> Path:
    """Write a fast configuration pointing at a synthetic data folder."""
    config: dict[str, object] = {
        "tour": tour,
        "seed": 7,
        "levels": ["G", "M", "A", "F"] if tour == "atp" else ["G", "PM", "P", "I", "F"],
        "data": {
            "raw_dir": str(root / "raw"),
            "cache_dir": str(root / "interim"),
            "history_tiers": ["main", "qual_chall" if tour == "atp" else "qual_itf"],
            "first_year": 2014,
            "last_year": 2024,
        },
        "split": {
            "train_start": 2015,
            "train_end": 2020,
            "val_years": [2021, 2022],
            "test_years": [2023, 2024],
        },
        "xgboost": {
            "n_estimators": 150,
            "early_stopping_rounds": 20,
            "n_jobs": 1,
            "learning_rate": 0.1,
        },
        "models_dir": str(root / "models"),
        "reports_dir": str(root / "reports"),
    }
    config.update(overrides)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path
