"""Persist and restore everything needed to serve predictions (no pickle)."""

from __future__ import annotations

import datetime as dt
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel
from xgboost import XGBClassifier

from tennis_predictor import __version__
from tennis_predictor.config import Tour
from tennis_predictor.features.tracker import FeatureTracker
from tennis_predictor.modeling.calibration import TemperatureScaler

MODEL_FILE = "model.ubj"
METADATA_FILE = "metadata.json"
PREDICTIONS_FILE = "predictions.parquet"
STATE_DIR = "state"


class ArtifactMetadata(BaseModel):
    """Provenance and evaluation context written next to the model."""

    package_version: str = __version__
    created_at: dt.datetime
    tour: Tour
    data_cutoff: dt.date
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
    """Write the model, the tracker state, the metadata and the held-out predictions."""
    directory.mkdir(parents=True, exist_ok=True)
    model.save_model(directory / MODEL_FILE)
    tracker.save(directory / STATE_DIR)
    predictions.to_parquet(directory / PREDICTIONS_FILE, index=False)
    (directory / METADATA_FILE).write_text(metadata.model_dump_json(indent=2), encoding="utf-8")


def load_metadata(directory: Path) -> ArtifactMetadata:
    """Read the metadata of a trained model."""
    path = directory / METADATA_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no trained model in {directory}; run `tennis train` first")
    return ArtifactMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def load_artifacts(directory: Path) -> LoadedArtifacts:
    """Restore a model saved by :func:`save_artifacts`."""
    metadata = load_metadata(directory)
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
