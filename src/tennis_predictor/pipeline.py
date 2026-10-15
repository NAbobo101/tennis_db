"""End-to-end steps behind the ``tennis train`` and ``tennis evaluate`` commands."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import asdict
from pathlib import Path

from xgboost import XGBClassifier

from tennis_predictor.config import Settings
from tennis_predictor.data.loader import load_matches
from tennis_predictor.data.schema import day_to_date
from tennis_predictor.evaluation.report import Report, build_report, write_report
from tennis_predictor.features.dataset import build_dataset
from tennis_predictor.features.specs import FEATURE_NAMES
from tennis_predictor.modeling.artifacts import (
    MODEL_FILE,
    ArtifactMetadata,
    library_versions,
    load_metadata,
    load_predictions,
    save_artifacts,
)
from tennis_predictor.modeling.train import logistic_coefficients, train_models

logger = logging.getLogger(__name__)


def run_training(settings: Settings, *, use_cache: bool = True) -> ArtifactMetadata:
    """Load data, build features, fit every model and save the artifacts."""
    matches = load_matches(settings.data, settings.tour, use_cache=use_cache)
    dataset, tracker = build_dataset(matches, settings)
    result = train_models(dataset, settings)
    best = min(result.search, key=lambda candidate: candidate.val_log_loss)
    metadata = ArtifactMetadata(
        created_at=dt.datetime.now(dt.UTC).replace(microsecond=0),
        tour=settings.tour,
        data_cutoff=day_to_date(tracker.last_day),
        feature_names=list(FEATURE_NAMES),
        settings=settings.model_dump(mode="json"),
        calibration_slope=result.calibrator.slope,
        best_params=best.params,
        best_iteration=best.best_iteration,
        search=[asdict(candidate) for candidate in result.search],
        row_counts=result.row_counts,
        logistic_coefficients=logistic_coefficients(result.logistic),
        library_versions=library_versions(),
    )
    save_artifacts(
        settings.model_dir,
        model=result.xgboost,
        tracker=tracker,
        metadata=metadata,
        predictions=result.predictions,
    )
    logger.info("Saved %s artifacts to %s", settings.tour.upper(), settings.model_dir)
    return metadata


def run_evaluation(settings: Settings, *, n_resamples: int = 2000) -> tuple[Report, list[Path]]:
    """Score the saved validation/test predictions and write the report."""
    metadata = load_metadata(settings.model_dir)
    predictions = load_predictions(settings.model_dir)
    model = XGBClassifier()
    model.load_model(settings.model_dir / MODEL_FILE)
    scores = model.get_booster().get_score(importance_type="total_gain")
    importance = {
        name: float(sum(value) if isinstance(value, list) else value)
        for name, value in scores.items()
    }
    report = build_report(
        predictions,
        tour=metadata.tour,
        data_cutoff=metadata.data_cutoff.isoformat(),
        row_counts=metadata.row_counts,
        model_info={
            "best_params": metadata.best_params,
            "best_iteration": metadata.best_iteration,
            "calibration_slope": round(metadata.calibration_slope, 4),
            "search": metadata.search,
        },
        importance=importance,
        n_resamples=n_resamples,
    )
    return report, write_report(report, predictions, settings.report_dir)
