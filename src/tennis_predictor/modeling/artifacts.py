"""Persist and restore everything needed to serve predictions (no pickle)."""

from __future__ import annotations

import datetime as dt
import os
import platform
import shutil
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel
from xgboost import XGBClassifier

from tennis_predictor import __version__
from tennis_predictor.config import Tour
from tennis_predictor.features.specs import FEATURE_NAMES
from tennis_predictor.features.tracker import FeatureTracker
from tennis_predictor.modeling.calibration import TemperatureScaler

MODEL_FILE = "model.ubj"
METADATA_FILE = "metadata.json"
PREDICTIONS_FILE = "predictions.parquet"
STATE_DIR = "state"
#: Bump when the meaning of a feature changes, so stale models are refused at load time.
SCHEMA_VERSION = 1


class ArtifactMetadata(BaseModel):
    """Provenance and evaluation context written next to the model."""

    package_version: str = __version__
    schema_version: int = SCHEMA_VERSION
    created_at: dt.datetime
    tour: Tour
    data_cutoff: dt.date
    rankings_date: dt.date | None = None
    feature_names: list[str]
    settings: dict[str, Any]
    calibration_slope: float
    best_params: dict[str, float]
    best_iteration: int
    search: list[dict[str, Any]]
    row_counts: dict[str, int]
    logistic_coefficients: dict[str, float]
    library_versions: dict[str, str]


def library_versions() -> dict[str, str]:
    """Versions of the interpreter and the libraries that influence results."""
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "scipy"):
        versions[package] = version(package)
    import xgboost

    versions["xgboost"] = xgboost.__version__
    return versions


@dataclass(frozen=True)
class LoadedArtifacts:
    """A trained model ready to serve."""

    model: XGBClassifier
    calibrator: TemperatureScaler
    tracker: FeatureTracker
    metadata: ArtifactMetadata


def save_artifacts(
    directory: Path,
    *,
    model: XGBClassifier,
    tracker: FeatureTracker,
    metadata: ArtifactMetadata,
    predictions: pd.DataFrame,
) -> None:
    """Write the model, the tracker state, the metadata and the held-out predictions.

    Everything is written to a temporary sibling directory that replaces ``directory`` only
    once complete, so an interrupted run never mixes files from two trainings.
    """
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(f".{directory.name}.tmp-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    model.save_model(staging / MODEL_FILE)
    tracker.save(staging / STATE_DIR)
    predictions.to_parquet(staging / PREDICTIONS_FILE, index=False)
    (staging / METADATA_FILE).write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    previous = directory.with_name(f".{directory.name}.old-{os.getpid()}")
    if directory.exists():
        directory.rename(previous)
    staging.rename(directory)
    shutil.rmtree(previous, ignore_errors=True)


def load_metadata(directory: Path) -> ArtifactMetadata:
    """Read the metadata of a trained model."""
    path = directory / METADATA_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no trained model in {directory}; run `tennis train` first")
    return ArtifactMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def load_artifacts(directory: Path) -> LoadedArtifacts:
    """Restore a model saved by :func:`save_artifacts`.

    Raises:
        FileNotFoundError: A required file is missing.
        ValueError: The model was trained with an incompatible feature schema.
    """
    metadata = load_metadata(directory)
    if metadata.schema_version != SCHEMA_VERSION or metadata.feature_names != list(FEATURE_NAMES):
        raise ValueError(
            f"the model in {directory} was trained with an incompatible feature schema; "
            "run `tennis train` again"
        )
    state = directory / STATE_DIR
    for path in (
        directory / MODEL_FILE,
        state / "tracker.json",
        state / "players.parquet",
        state / "h2h.parquet",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; run `tennis train` again")
    model = XGBClassifier()
    model.load_model(directory / MODEL_FILE)
    return LoadedArtifacts(
        model=model,
        calibrator=TemperatureScaler(metadata.calibration_slope),
        tracker=FeatureTracker.load(directory / STATE_DIR),
        metadata=metadata,
    )


def load_predictions(directory: Path) -> pd.DataFrame:
    """Validation and test predictions written at training time."""
    path = directory / PREDICTIONS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no predictions in {directory}; run `tennis train` first")
    return pd.read_parquet(path)
