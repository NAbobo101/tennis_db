"""Synthetic match files in the exact Sackmann CSV format.

Players have a latent skill and winners are drawn from a Bradley-Terry model, so a sane
pipeline must learn something, while every quirk the loader handles is present: unsorted
rounds, retirements, walkovers, duplicates, Carpet, qualifying rounds, Challenger events
and an excluded exhibition.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

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

SURNAMES = [
    "Archer", "Baxter", "Carver", "Dalton", "Ellison", "Fletcher", "Garner", "Hollis",
    "Irving", "Jarrett", "Keller", "Lawson", "Mercer", "Norris", "Oakley", "Porter",
    "Quinlan", "Rowe", "Sutton", "Turner", "Upton", "Vance", "Walsh", "Yates",
]  # fmt: skip
FIRST_NAMES = ["Adam", "Ben", "Carl", "Dan", "Eli", "Finn", "Gus", "Hugo"]
SURFACE_CYCLE = ["Hard", "Clay", "Grass", "Hard", "Carpet", "Clay"]
KNOCKOUT = ["R16", "QF", "SF", "F"]


def _players(n_players: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for index in range(n_players):
        surname = SURNAMES[index % len(SURNAMES)]
        first = FIRST_NAMES[index // len(SURNAMES) % len(FIRST_NAMES)]
        rows.append(
            {
                "id": 100001 + index,
                "name": f"{first} {surname}",
                "skill": rng.normal(0.0, 1.0),
                "clay": rng.normal(0.0, 0.3),
                "hand": "L" if index % 7 == 0 else "R",
                "ht": float(170 + index % 25),
                "ioc": ["USA", "ESP", "FRA", "ITA"][index % 4],
                "born": 1985.0 + index % 15,
            }
        )
    # Two active namesakes and one retired namesake for the name-resolution tests.
    rows[1]["name"] = "Ana Twin"
    rows[2]["name"] = "Bea Twin"
    rows[3]["name"] = "Carl Solo"
    rows[4]["name"] = "Dan Solo"
    return pd.DataFrame(rows)


def _serve_stats(rng: np.random.Generator, edge: float, games: int) -> list[float]:
    svpt = int(6 * games + rng.integers(-4, 5))
    first_in = round(svpt * rng.uniform(0.55, 0.68))
    double_faults = int(rng.integers(0, 5))
    first_won = round(first_in * min(0.9, max(0.5, 0.70 + 0.04 * edge)))
    second_won = round((svpt - first_in - double_faults) * min(0.8, max(0.3, 0.52 + 0.04 * edge)))
    aces = int(rng.integers(0, max(1, first_won // 4)))
    bp_faced = int(rng.integers(0, 10))
    bp_saved = int(rng.integers(0, bp_faced + 1))
    serve_games = max(1, games // 2)
    return [
        aces,
        double_faults,
        svpt,
        first_in,
        first_won,
        second_won,
        serve_games,
        bp_saved,
        bp_faced,
    ]


def _score(rng: np.random.Generator, best_of: int) -> tuple[str, int]:
    sets_needed = best_of // 2 + 1
    sets = []
    lost = 0
    for _ in range(sets_needed):
        if lost < sets_needed - 1 and rng.random() < 0.3:
            sets.append(f"{rng.integers(2, 5)}-6")
            lost += 1
        sets.append(str(rng.choice(["6-3", "6-4", "7-5", "6-2", "7-6(4)"])))
    games = sum(int(a) + int(b.split("(")[0]) for a, b in (s.split("-") for s in sets))
    return " ".join(sets), games


def _match_row(
    tourney: Mapping[str, object],
    match_num: int,
    round_code: str,
    winner: pd.Series,
    loser: pd.Series,
    ranks: dict[int, int],
    rng: np.random.Generator,
) -> dict[str, object]:
    best_of = int(tourney["best_of"])  # type: ignore[call-overload]
    score, games = _score(rng, best_of)
    year = int(str(tourney["tourney_date"])[:4])
    edge = float(winner["skill"] - loser["skill"])
    row: dict[str, object] = {
        **{
            key: tourney[key]
            for key in ("tourney_id", "tourney_name", "surface", "tourney_level", "tourney_date")
        },
        "draw_size": 16,
        "match_num": match_num,
        "score": score,
        "best_of": best_of,
        "round": round_code,
        "minutes": 30 + 5 * games,
    }
    for side, player in (("winner", winner), ("loser", loser)):
        row[f"{side}_id"] = int(player["id"])
        row[f"{side}_name"] = player["name"]
        row[f"{side}_hand"] = player["hand"]
        row[f"{side}_ht"] = player["ht"]
        row[f"{side}_ioc"] = player["ioc"]
        row[f"{side}_age"] = round(year - float(player["born"]), 1)
        row[f"{side}_rank"] = ranks[int(player["id"])]
        row[f"{side}_rank_points"] = max(10, 5000 - 90 * ranks[int(player["id"])])
        row[f"{side}_seed"] = ""
        row[f"{side}_entry"] = ""
    for prefix, sign in (("w", 1.0), ("l", -1.0)):
        stats = _serve_stats(rng, sign * edge, games)
        for name, value in zip(
            ("ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"),
            stats,
            strict=True,
        ):
            row[f"{prefix}_{name}"] = value
    return row


def _play(
    a: pd.Series, b: pd.Series, surface: str, rng: np.random.Generator
) -> tuple[pd.Series, pd.Series]:
    strength_a = float(a["skill"]) + (float(a["clay"]) if surface == "Clay" else 0.0)
    strength_b = float(b["skill"]) + (float(b["clay"]) if surface == "Clay" else 0.0)
    p_a = 1.0 / (1.0 + math.exp(-(strength_a - strength_b)))
    return (a, b) if rng.random() < p_a else (b, a)


def write_synthetic_tour(
    raw_dir: Path,
    *,
    years: range = range(2015, 2025),
    n_players: int = 40,
    events_per_year: int = 12,
    seed: int = 0,
) -> pd.DataFrame:
    """Write ``atp_matches_YYYY.csv`` and ``atp_matches_qual_chall_YYYY.csv`` files.

    Returns all written main-draw rows (before any cleaning) for assertions.
    """
    rng = np.random.default_rng(seed)
    players = _players(n_players, rng)
    by_rank = players.sort_values("skill", ascending=False)["id"].tolist()
    ranks = {int(player_id): position + 1 for position, player_id in enumerate(by_rank)}
    folder = raw_dir / "tennis_atp"
    folder.mkdir(parents=True, exist_ok=True)
    everything = []
    for year in years:
        main_rows: list[dict[str, object]] = []
        qual_rows: list[dict[str, object]] = []
        monday = dt.date(year, 1, 1) + dt.timedelta(days=(7 - dt.date(year, 1, 1).weekday()) % 7)
        for event in range(events_per_year):
            date = monday + dt.timedelta(weeks=3 * event)
            level = "G" if event in (0, 6) else ("M" if event % 3 == 1 else "A")
            name = "Next Gen Finals" if event == events_per_year - 1 else f"Open {event:02d}"
            tourney = {
                "tourney_id": f"{year}-{event:03d}",
                "tourney_name": name,
                "surface": SURFACE_CYCLE[event % len(SURFACE_CYCLE)],
                "tourney_level": level,
                "tourney_date": date.strftime("%Y%m%d"),
                "best_of": 5 if level == "G" else 3,
            }
            # "Carl Solo" retires after 2016, so "Solo" resolves to the active "Dan Solo".
            pool_ids = [pid for pid in players["id"].tolist() if year <= 2016 or pid != 100004]
            drawn = rng.choice(pool_ids, size=20, replace=False)
            draw = [players.loc[players["id"] == pid].iloc[0] for pid in drawn]

            # Qualifying: four Q1 matches whose winners enter the main draw.
            qualifiers = []
            for q in range(4):
                winner, loser = _play(draw[16 + q], draw[q], str(tourney["surface"]), rng)
                qual_rows.append(_match_row(tourney, 900 + q, "Q1", winner, loser, ranks, rng))
                qualifiers.append(winner)
            current = qualifiers + draw[4:16]
            rows_for_event = []
            match_num = 1
            for round_code in KNOCKOUT:
                next_round = []
                for pair in range(0, len(current), 2):
                    winner, loser = _play(
                        current[pair], current[pair + 1], str(tourney["surface"]), rng
                    )
                    rows_for_event.append(
                        _match_row(tourney, match_num, round_code, winner, loser, ranks, rng)
                    )
                    next_round.append(winner)
                    match_num += 1
                current = next_round
            if event % 2:  # some events are stored final-first, as in recent upstream files
                rows_for_event.reverse()
            main_rows.extend(rows_for_event)

            # A Challenger event in the same week only feeds the rating history.
            challenger = {
                **tourney,
                "tourney_id": f"{year}-C{event:02d}",
                "tourney_name": f"Challenger {event}",
                "tourney_level": "C",
                "best_of": 3,
            }
            winner, loser = _play(draw[16], draw[17], str(tourney["surface"]), rng)
            qual_rows.append(_match_row(challenger, 1, "F", winner, loser, ranks, rng))

        # Quirks: a retirement, a walkover with fake zero stats, and an exact duplicate.
        main_rows[3]["score"] = "6-4 2-1 RET"
        main_rows[5]["score"] = "W/O"
        for key in list(main_rows[5]):
            if key.startswith(("w_", "l_")):
                main_rows[5][key] = 0
        main_rows.append(dict(main_rows[7]))
        frame = pd.DataFrame(main_rows, columns=COLUMNS)
        frame.to_csv(folder / f"atp_matches_{year}.csv", index=False)
        pd.DataFrame(qual_rows, columns=COLUMNS).to_csv(
            folder / f"atp_matches_qual_chall_{year}.csv", index=False
        )
        everything.append(frame)
    return pd.concat(everything, ignore_index=True)


def write_settings(path: Path, root: Path, **overrides: object) -> Path:
    """Write a fast configuration pointing at a synthetic data folder."""
    import yaml

    config: dict[str, object] = {
        "tour": "atp",
        "seed": 7,
        "levels": ["G", "M", "A", "F"],
        "data": {
            "raw_dir": str(root / "raw"),
            "cache_dir": str(root / "interim"),
            "history_tiers": ["main", "qual_chall"],
            "first_year": 2015,
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
