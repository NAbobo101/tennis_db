"""Official weekly rankings, used to refresh player facts at the data cutoff."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from tennis_predictor.config import Tour
from tennis_predictor.data.schema import FILE_FAMILIES

logger = logging.getLogger(__name__)

#: Ranking files covering the most recent seasons (older decades are not needed).
RANKING_FILES: dict[str, tuple[str, ...]] = {
    "atp": ("atp_rankings_20s.csv", "atp_rankings_current.csv"),
    "wta": ("wta_rankings_20s.csv", "wta_rankings_current.csv"),
}


def latest_rankings(raw_dir: Path, tour: Tour, as_of: dt.date) -> pd.DataFrame | None:
    """The last official ranking list published on or before ``as_of``.

    Returns a frame with ``player_id``, ``rank`` and ``points`` (one row per ranked player,
    ``attrs["ranking_date"]`` holds the list's date), or ``None`` when no ranking file is
    available.
    """
    folder = raw_dir / FILE_FAMILIES[(tour, "main")].repo
    paths = [folder / name for name in RANKING_FILES[tour] if (folder / name).is_file()]
    if not paths:
        logger.warning(
            "No %s ranking files in %s; serving uses last-match ranks", tour.upper(), folder
        )
        return None
    frame = pd.concat(
        [pd.read_csv(path, usecols=["ranking_date", "rank", "player", "points"]) for path in paths],
        ignore_index=True,
    )
    frame = frame.loc[frame["ranking_date"] <= int(as_of.strftime("%Y%m%d"))]
    if frame.empty:
        return None
    date = int(frame["ranking_date"].max())
    latest = (
        frame.loc[frame["ranking_date"] == date]
        .sort_values("rank")
        .drop_duplicates("player")
        .rename(columns={"player": "player_id"})
    )
    result = (
        latest[["player_id", "rank", "points"]]
        .astype({"player_id": "int64", "rank": "float64", "points": "float64"})
        .reset_index(drop=True)
    )
    result.attrs["ranking_date"] = dt.datetime.strptime(str(date), "%Y%m%d").date()
    return result
