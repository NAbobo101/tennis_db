from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from tennis_predictor.config import DataConfig, Settings
from tennis_predictor.data.loader import SORT_KEY, load_matches, match_files
from tennis_predictor.data.schema import ROUND_ORDER, date_to_day


def test_match_files_lists_existing_tiers(settings: Settings) -> None:
    files = match_files(
        settings.data.raw_dir, "atp", ("main", "qual_chall", "futures"), range(2014, 2026)
    )
    tiers = {tier for tier, _ in files}
    assert tiers == {"main", "qual_chall"}  # no futures files and no 2014/2025 files exist
    assert len(files) == 20


def test_rows_are_chronological(matches: pd.DataFrame) -> None:
    assert matches[SORT_KEY].equals(matches[SORT_KEY].sort_values(SORT_KEY, kind="stable"))
    # Inside each tournament the final comes last, even for events stored final-first.
    for _, event in matches.loc[matches["tier"] == "main"].groupby("tourney_id"):
        assert event["round"].iloc[-1] == "F"
        assert event["round_order"].is_monotonic_increasing


def test_qualifying_sorts_before_main_draw(matches: pd.DataFrame) -> None:
    event = matches.loc[matches["tourney_id"] == "2016-002"]
    assert event["round"].iloc[0] == "Q1"
    assert ROUND_ORDER["Q1"] < ROUND_ORDER["R16"] < ROUND_ORDER["QF"]


def test_cleaning(matches: pd.DataFrame) -> None:
    assert "Carpet" not in set(matches["surface"])
    assert not matches.duplicated(["tourney_id", "winner_id", "loser_id", "round", "score"]).any()
    walkovers = matches.loc[matches["outcome"] == "walkover"]
    retirements = matches.loc[matches["outcome"] == "retired"]
    assert len(walkovers) == 10
    assert len(retirements) == 10
    # Walkovers carry fake zero stats upstream; they must never count as statistics.
    assert not walkovers["stats_ok"].any()
    assert not retirements["stats_ok"].any()
    assert matches.loc[matches["outcome"] == "completed", "stats_ok"].mean() > 0.95
    assert matches.loc[matches["excluded"], "tourney_name"].eq("Next Gen Finals").all()
    assert matches["day"].iloc[0] == date_to_day(pd.Timestamp("2015-01-05").date())


def test_invalid_stats_are_flagged(tmp_path: Path, settings: Settings) -> None:
    raw = tmp_path / "raw" / "tennis_atp"
    raw.mkdir(parents=True)
    frame = pd.read_csv(settings.data.raw_dir / "tennis_atp" / "atp_matches_2015.csv", dtype=str)
    frame.loc[0, "w_1stIn"] = "999"  # more first serves in than serve points
    frame.loc[1, "l_bpSaved"] = "50"  # more break points saved than faced
    frame.to_csv(raw / "atp_matches_2015.csv", index=False)
    cfg = DataConfig(
        raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache", first_year=2015, last_year=2015
    )
    loaded = load_matches(cfg, "atp", use_cache=False)
    edited = loaded.loc[
        (loaded["tourney_id"] == frame.loc[0, "tourney_id"])
        & loaded["match_num"].isin([int(str(frame.loc[i, "match_num"])) for i in (0, 1)])
    ]
    assert len(edited) == 2
    assert not edited["stats_ok"].any()


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
