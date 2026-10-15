from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tennis_predictor.config import DataConfig, Settings
from tennis_predictor.data.loader import SORT_KEY, load_matches, match_files, normalize_levels
from tennis_predictor.data.schema import ROUND_ORDER, date_to_day
from tests.synthetic import BAD_HEIGHT_PLAYER, NO_HEIGHT_PLAYER


def test_match_files_lists_existing_tiers(settings: Settings) -> None:
    files = match_files(
        settings.data.raw_dir, "atp", ("main", "qual_chall", "futures"), range(2013, 2026)
    )
    assert {tier for tier, _ in files} == {"main", "qual_chall"}  # no Futures files exist
    assert len(files) == 2 * 11  # 2014-2024, two tiers
    assert [path.name for _, path in files[:2]] == [
        "atp_matches_2014.csv",
        "atp_matches_qual_chall_2014.csv",
    ]


def test_rows_are_chronological(matches: pd.DataFrame) -> None:
    assert matches[SORT_KEY].equals(matches[SORT_KEY].sort_values(SORT_KEY, kind="stable"))
    knockout = matches.loc[
        (matches["tier"] == "main") & matches["round"].isin(["R16", "QF", "SF", "F"])
    ]
    for _, event in knockout.groupby("tourney_id"):
        # Inside each event the final comes last, even for events stored final-first.
        assert event["round"].iloc[-1] == "F"
        assert event["round_order"].is_monotonic_increasing


def test_qualifying_sorts_before_main_draw(matches: pd.DataFrame) -> None:
    event = matches.loc[matches["tourney_id"] == "2016-002"]
    assert event["round"].iloc[0] == "Q1"
    assert event["tier"].iloc[0] == "qual_chall"
    assert ROUND_ORDER["Q1"] < ROUND_ORDER["R16"] < ROUND_ORDER["RR"] < ROUND_ORDER["QF"]


def test_cleaning(matches: pd.DataFrame) -> None:
    years = matches["year"].nunique()
    assert "Carpet" not in set(matches["surface"])
    assert not matches.duplicated(["tourney_id", "winner_id", "loser_id", "round", "score"]).any()
    walkovers = matches.loc[matches["outcome"] == "walkover"]
    retirements = matches.loc[matches["outcome"] == "retired"]
    assert len(walkovers) == years
    assert len(retirements) == years
    # Walkovers carry fake zero stats upstream; they must never count as statistics.
    assert not walkovers["stats_ok"].any()
    assert not retirements["stats_ok"].any()
    main_completed = matches.loc[(matches["tier"] == "main") & (matches["outcome"] == "completed")]
    assert main_completed["stats_ok"].mean() > 0.95
    challengers = matches.loc[matches["tourney_level"] == "C"]
    assert len(challengers) > 0
    assert not challengers["stats_ok"].any()  # completed, but no statistics upstream
    assert matches["excluded"].sum() == (15 + 4) * years  # main draw + qualifying
    assert set(matches.loc[matches["excluded"], "tourney_name"]) == {"Next Gen Finals"}
    assert matches["day"].iloc[0] == date_to_day(pd.Timestamp("2014-01-06").date())


def test_heights_and_ranks_are_cleaned(matches: pd.DataFrame) -> None:
    winners = matches.set_index("winner_id")
    assert winners.loc[[BAD_HEIGHT_PLAYER], "winner_ht"].isna().all()
    assert winners.loc[[NO_HEIGHT_PLAYER], "winner_ht"].isna().all()
    assert matches["winner_rank"].isna().any()


def test_invalid_stats_are_flagged(tmp_path: Path, settings: Settings) -> None:
    raw = tmp_path / "raw" / "tennis_atp"
    raw.mkdir(parents=True)
    frame = pd.read_csv(settings.data.raw_dir / "tennis_atp" / "atp_matches_2015.csv", dtype=str)
    frame.loc[0, "w_1stIn"] = "999"  # more first serves in than serve points
    frame.loc[1, "l_bpSaved"] = "50"  # more break points saved than faced
    frame.loc[2, "w_svpt"] = "2000"  # far more serve points than the games allow
    frame.loc[2, "w_1stIn"] = "1000"
    frame.loc[3, "l_2ndWon"] = frame.loc[3, "l_svpt"]  # second-serve points won incl. double faults
    frame.to_csv(raw / "atp_matches_2015.csv", index=False)
    cfg = DataConfig(
        raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache", first_year=2015, last_year=2015
    )
    loaded = load_matches(cfg, "atp", use_cache=False)
    edited = loaded.loc[
        (loaded["tourney_id"] == frame.loc[0, "tourney_id"])
        & loaded["match_num"].isin([int(str(frame.loc[i, "match_num"])) for i in (0, 1, 2, 4)])
    ].set_index("match_num")
    assert not edited.loc[
        [int(str(frame.loc[i, "match_num"])) for i in (0, 1, 2)], "stats_ok"
    ].any()
    assert edited.loc[int(str(frame.loc[4, "match_num"])), "stats_ok"]


@pytest.mark.parametrize(
    ("tour", "name", "level", "year", "expected"),
    [
        ("atp", "Tour Finals", "A", 2023, "F"),
        ("atp", "Masters Cup", "F", 2005, "F"),
        ("atp", "Tokyo Olympics", "A", 2021, "O"),
        ("atp", "Beijing", "A", 2024, "A"),
        ("wta", "WTA Finals", "W", 2015, "F"),
        ("wta", "Indian Wells", "P", 2016, "PM"),
        ("wta", "Rome", "P", 2016, "P"),
        ("wta", "Buenos Aires 125", "W", 2024, "C"),
        ("wta", "Eastbourne", "W", 2021, "I"),
        ("wta", "Wimbledon", "W", 1995, "W"),
        ("wta", "Olympics", "O", 2016, "O"),
    ],
)
def test_normalize_levels(tour: str, name: str, level: str, year: int, expected: str) -> None:
    assert normalize_levels(tour, [name], [level], [year]) == [expected]  # type: ignore[arg-type]


def test_cache_is_reused_and_invalidated(tmp_path: Path, settings: Settings) -> None:
    raw = tmp_path / "raw" / "tennis_atp"
    raw.mkdir(parents=True)
    source = settings.data.raw_dir / "tennis_atp" / "atp_matches_2016.csv"
    target = raw / "atp_matches_2016.csv"
    target.write_bytes(source.read_bytes())
    cfg = DataConfig(
        raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache", first_year=2016, last_year=2016
    )
    first = load_matches(cfg, "atp")
    caches = list((tmp_path / "cache").glob("atp_matches_*.parquet"))
    assert len(caches) == 1
    pd.testing.assert_frame_equal(load_matches(cfg, "atp"), first)

    target.write_bytes(source.read_bytes() + b"\n")
    os.utime(target, ns=(1, 1))
    load_matches(cfg, "atp")
    new_caches = list((tmp_path / "cache").glob("atp_matches_*.parquet"))
    assert len(new_caches) == 1
    assert new_caches != caches


def test_missing_data_raises(tmp_path: Path) -> None:
    cfg = DataConfig(raw_dir=tmp_path, cache_dir=tmp_path / "cache")
    with pytest.raises(FileNotFoundError, match="tennis data download"):
        load_matches(cfg, "wta")


def test_wta_files_load(wta_settings: Settings) -> None:
    loaded = load_matches(wta_settings.data, "wta", use_cache=False)
    assert set(loaded["tier"]) == {"main", "qual_itf"}
    assert {"G", "PM", "I", "F", "D"} <= set(loaded["tourney_level"])
    assert np.all(loaded.loc[loaded["tourney_level"] == "G", "best_of"] == 3)
