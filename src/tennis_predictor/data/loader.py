"""Read, clean and chronologically order the Sackmann match files."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tennis_predictor.config import DataConfig, Tier, Tour
from tennis_predictor.data.schema import (
    EPOCH,
    FILE_FAMILIES,
    LOSER_STATS,
    MATCH_COLUMNS,
    NUMERIC_COLUMNS,
    ROUND_ORDER,
    WINNER_STATS,
    Outcome,
    canonical_surface,
)
from tennis_predictor.data.scores import parse_scores

logger = logging.getLogger(__name__)

#: Bump when the cleaning logic changes so stale parquet caches are rebuilt.
CACHE_VERSION = 1

SORT_KEY = ["day", "tourney_id", "round_order", "match_num"]
_EPOCH = np.datetime64(EPOCH.isoformat(), "D")


def match_files(
    raw_dir: Path, tour: Tour, tiers: Sequence[Tier], years: range
) -> list[tuple[Tier, Path]]:
    """List the existing yearly match files of the requested tiers, oldest first."""
    found: list[tuple[Tier, Path]] = []
    for year in years:
        for tier in tiers:
            family = FILE_FAMILIES[(tour, tier)]
            path = raw_dir / family.repo / family.pattern.format(year=year)
            if path.is_file():
                found.append((tier, path))
    return found


def read_match_file(path: Path, tier: Tier) -> pd.DataFrame:
    """Read one match file as text and coerce the numeric columns."""
    frame = pd.read_csv(path, dtype=str, usecols=lambda column: column in MATCH_COLUMNS)
    frame = frame.reindex(columns=list(MATCH_COLUMNS))
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["tier"] = tier
    return frame


def _valid_stats(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    ace, df, svpt, first_in, first_won, second_won, _, bp_saved, bp_faced = (
        frame[column] for column in columns
    )
    return (
        frame[list(columns)].drop(columns=[columns[6]]).notna().all(axis=1)
        & (svpt > 0)
        & (ace >= 0)
        & (df >= 0)
        & (first_in <= svpt)
        & (first_won <= first_in)
        & (second_won <= svpt - first_in)
        & (ace + df <= svpt)
        & (bp_saved <= bp_faced)
    )


def clean_matches(frame: pd.DataFrame, exclude_tourneys: str) -> pd.DataFrame:
    """Normalize types and values, flag usable rows and drop broken ones.

    Adds the columns ``day`` (days since :data:`~tennis_predictor.data.schema.EPOCH` of the
    tournament Monday), ``year``,
    ``round_order``, ``outcome``, ``w_games``, ``l_games``, ``excluded`` and ``stats_ok``.
    """
    df = frame.copy()
    dates = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    ids = df[["winner_id", "loser_id"]].apply(pd.to_numeric, errors="coerce")
    keep = dates.notna() & ids.notna().all(axis=1)
    df, dates, ids = df.loc[keep], dates.loc[keep], ids.loc[keep]
    df["winner_id"] = ids["winner_id"].astype("int64")
    df["loser_id"] = ids["loser_id"].astype("int64")
    df = df.loc[df["winner_id"] != df["loser_id"]]
    dates = dates.loc[df.index]

    day = (dates.to_numpy(dtype="datetime64[D]") - _EPOCH).astype("int64")
    df["day"] = day
    df["year"] = dates.dt.year.astype("int32")

    round_code = df["round"].fillna("").str.strip().str.upper()
    df["round"] = round_code
    df["round_order"] = round_code.map(ROUND_ORDER).fillna(ROUND_ORDER["RR"]).astype("int8")
    df["match_num"] = df["match_num"].fillna(0).astype("int64")
    best_of = df["best_of"].where(df["best_of"].isin([3, 5]), 3)
    df["best_of"] = best_of.astype("int8")
    df["surface"] = [canonical_surface(value) for value in df["surface"].tolist()]
    df["tourney_level"] = df["tourney_level"].fillna("").str.strip()
    df["tourney_id"] = df["tourney_id"].fillna("").str.strip()
    df["tourney_name"] = df["tourney_name"].fillna("")
    for side in ("winner", "loser"):
        df[f"{side}_name"] = df[f"{side}_name"].fillna("").str.strip()
        df[f"{side}_hand"] = df[f"{side}_hand"].fillna("U").str.strip().str.upper()
        df[f"{side}_entry"] = df[f"{side}_entry"].fillna("").str.strip().str.upper()
        df[f"{side}_ioc"] = df[f"{side}_ioc"].fillna("")
        height = df[f"{side}_ht"]
        df[f"{side}_ht"] = height.where(height >= 140)

    scores = parse_scores(df["score"])
    df["score"] = scores["score"]
    df["outcome"] = scores["outcome"]
    df["w_games"] = scores["w_games"]
    df["l_games"] = scores["l_games"]

    pattern = re.compile(exclude_tourneys)
    df["excluded"] = [bool(pattern.search(name)) for name in df["tourney_name"].tolist()]

    for column in ("w_SvGms", "l_SvGms"):
        df[column] = df[column].where(df[column] > 0)
    df["stats_ok"] = (
        (df["outcome"] == Outcome.COMPLETED.value)
        & _valid_stats(df, WINNER_STATS)
        & _valid_stats(df, LOSER_STATS)
    )

    df = df.drop_duplicates(subset=["tourney_id", "winner_id", "loser_id", "round", "score"])
    return sort_chronologically(df)


def sort_chronologically(frame: pd.DataFrame) -> pd.DataFrame:
    """Stable sort by tournament week, tournament, round and match number."""
    return frame.sort_values(SORT_KEY, kind="stable").reset_index(drop=True)


def source_fingerprint(paths: Sequence[Path], *extra: object) -> str:
    """Hash file names, sizes and modification times together with extra cache keys."""
    digest = hashlib.sha256(f"v{CACHE_VERSION}".encode())
    for item in extra:
        digest.update(repr(item).encode())
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def load_matches(cfg: DataConfig, tour: Tour, *, use_cache: bool = True) -> pd.DataFrame:
    """Load every configured match file of a tour as one clean, chronologically sorted frame.

    The result is cached as parquet in ``cfg.cache_dir``; the cache key covers the source
    files, the tiers, the year range, the exclusion pattern and :data:`CACHE_VERSION`.
    """
    years = range(cfg.first_year, cfg.last_year + 1)
    files = match_files(cfg.raw_dir, tour, cfg.history_tiers, years)
    if not files:
        raise FileNotFoundError(
            f"no {tour.upper()} match files found under {cfg.raw_dir}; "
            "run `tennis data download` first"
        )
    key = source_fingerprint(
        [path for _, path in files], tour, cfg.history_tiers, years, cfg.exclude_tourneys
    )
    cache = cfg.cache_dir / f"{tour}_matches_{key}.parquet"
    if use_cache and cache.is_file():
        logger.info("Loading cached matches from %s", cache)
        return pd.read_parquet(cache)

    logger.info("Reading %d %s match files", len(files), tour.upper())
    frames = [read_match_file(path, tier) for tier, path in files]
    matches = clean_matches(pd.concat(frames, ignore_index=True), cfg.exclude_tourneys)
    logger.info("Loaded %d matches", len(matches))
    if use_cache:
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        for stale in cfg.cache_dir.glob(f"{tour}_matches_*.parquet"):
            stale.unlink()
        matches.to_parquet(cache, index=False)
    return matches
