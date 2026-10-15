"""Fit the baselines, the logistic regression and the calibrated XGBoost model."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from tennis_predictor.config import Settings, XGBoostConfig
from tennis_predictor.features.dataset import Dataset
from tennis_predictor.features.specs import ANTISYMMETRIC_FEATURES, swap_players
from tennis_predictor.modeling.baselines import EloBaseline, RankBaseline
from tennis_predictor.modeling.calibration import TemperatureScaler

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
ProbaFn = Callable[[pd.DataFrame], FloatArray]


def positive_proba(model: XGBClassifier | Pipeline) -> ProbaFn:
    """Wrap a scikit-learn style classifier as ``X -> P(player A wins)``."""

    def predict(X: pd.DataFrame) -> FloatArray:
        return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)

    return predict


def symmetric_proba(predict: ProbaFn, X: pd.DataFrame) -> FloatArray:
    """Average the predictions for (A, B) and for the swapped pair (B, A).

    Tree ensembles are not exactly symmetric, so this guarantees
    ``p(A beats B) + p(B beats A) == 1`` whatever order the players are given in.
    """
    return 0.5 * (predict(X) + 1.0 - predict(swap_players(X)))


def make_logistic(c: float) -> Pipeline:
    """Regularized logistic regression on the antisymmetric features only.

    No intercept, imputation with 0 and scaling without centering keep the model exactly
    symmetric: swapping the players flips the sign of every input and hence of the logit.
    """
    columns = list(ANTISYMMETRIC_FEATURES)
    preprocess = ColumnTransformer(
        [
            (
                "antisymmetric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant", fill_value=0.0, keep_empty_features=True
                            ),
                        ),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                columns,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(C=c, fit_intercept=False, max_iter=5000)),
        ]
    )


def logistic_coefficients(pipeline: Pipeline) -> dict[str, float]:
    """Standardized coefficient of every feature of a fitted :func:`make_logistic` pipeline."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    return {str(name): float(value) for name, value in zip(names, coefficients, strict=True)}


def _swap_augmented(data: Dataset) -> tuple[pd.DataFrame, npt.NDArray[np.int8]]:
    """Training rows plus their player-swapped mirror (both stay in the training split)."""
    X = pd.concat([data.X, swap_players(data.X)], ignore_index=True)
    y = np.concatenate([data.y, 1 - data.y]).astype(np.int8)
    return X, y


@dataclass(frozen=True)
class SearchResult:
    """Validation score of one hyper-parameter candidate."""

    params: dict[str, float]
    best_iteration: int
    val_log_loss: float


def fit_xgboost(
    train: Dataset, val: Dataset, cfg: XGBoostConfig, seed: int
) -> tuple[XGBClassifier, list[SearchResult]]:
    """Fit one model per grid candidate with early stopping; keep the best on validation."""
    X_train, y_train = _swap_augmented(train) if cfg.augment_swapped else (train.X, train.y)
    best: XGBClassifier | None = None
    results: list[SearchResult] = []
    for params in cfg.candidates():
        model = XGBClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=params["learning_rate"],
            max_depth=int(params["max_depth"]),
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=cfg.early_stopping_rounds,
            tree_method="hist",
            n_jobs=cfg.n_jobs,
            random_state=seed,
        )
        model.fit(X_train, y_train, eval_set=[(val.X, val.y)], verbose=False)
        score = float(log_loss(val.y, symmetric_proba(positive_proba(model), val.X)))
        result = SearchResult(
            params=params, best_iteration=int(model.best_iteration), val_log_loss=score
        )
        logger.info(
            "XGBoost %s -> %d trees, validation log-loss %.4f", params, result.best_iteration, score
        )
        if not results or score < min(r.val_log_loss for r in results):
            best = model
        results.append(result)
    assert best is not None
    return best, results


@dataclass(frozen=True)
class TrainingResult:
    """Fitted models plus validation and test predictions of every model."""

    xgboost: XGBClassifier
    calibrator: TemperatureScaler
    logistic: Pipeline
    rank: RankBaseline
    search: list[SearchResult]
    predictions: pd.DataFrame
    row_counts: dict[str, int]


def train_models(dataset: Dataset, settings: Settings) -> TrainingResult:
    """Fit every model on the training years and predict the validation and test years."""
    train, val, test = (dataset.subset(split) for split in ("train", "val", "test"))
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("train, validation and test splits must all be non-empty")
    logger.info("Rows: train=%d val=%d test=%d", len(train), len(val), len(test))

    rank = RankBaseline().fit(train.X, train.y)
    logistic = make_logistic(settings.logistic_c).fit(train.X, train.y)
    xgb, search = fit_xgboost(train, val, settings.xgboost, settings.seed)
    calibrator = TemperatureScaler.fit(symmetric_proba(positive_proba(xgb), val.X), val.y)
    logger.info("Temperature scaling slope: %.3f", calibrator.slope)

    elo = EloBaseline()
    frames = []
    for part in (val, test):
        p_xgb_raw = symmetric_proba(positive_proba(xgb), part.X)
        frame = part.meta.copy()
        frame["y"] = part.y
        frame["p_rank"] = rank.predict_proba_a(part.X)
        frame["p_elo"] = elo.predict_proba_a(part.X)
        frame["p_logistic"] = positive_proba(logistic)(part.X)
        frame["p_xgboost_raw"] = p_xgb_raw
        frame["p_xgboost"] = calibrator.transform(p_xgb_raw)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    counts = {"train": len(train), "val": len(val), "test": len(test)}
    return TrainingResult(xgb, calibrator, logistic, rank, search, predictions, counts)
