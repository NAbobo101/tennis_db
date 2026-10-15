"""Evaluation report: metrics tables, breakdowns, bootstrap intervals and figures."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from tennis_predictor.evaluation.metrics import (
    BootstrapResult,
    Metrics,
    compute_metrics,
    paired_bootstrap,
)

#: Probability columns of the predictions file and their display names.
MODELS: Final = {
    "p_rank": "Rank baseline",
    "p_elo": "Elo baseline",
    "p_logistic": "Logistic regression",
    "p_xgboost": "XGBoost (calibrated)",
}
BREAKDOWNS: Final = {"surface": "Surface", "tourney_level": "Level", "best_of": "Best of"}
#: Paired comparisons reported with bootstrap intervals: (model, reference).
COMPARISONS: Final = (("p_logistic", "p_elo"), ("p_xgboost", "p_elo"), ("p_xgboost", "p_logistic"))
Theme = Literal["light", "dark"]

# Reference data-viz palette: categorical slots 1-3 and chart chrome, per theme.
_THEMES: Final[dict[str, dict[str, Any]]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}


@dataclass
class Report:
    """Everything the evaluation step computes for one tour."""

    tour: str
    data_cutoff: str
    row_counts: dict[str, int]
    overall: dict[str, dict[str, Metrics]]
    breakdowns: dict[str, pd.DataFrame]
    bootstrap: dict[str, list[BootstrapResult]]
    model_info: dict[str, Any] = field(default_factory=dict)
    importance: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "tour": self.tour,
            "data_cutoff": self.data_cutoff,
            "row_counts": self.row_counts,
            "metrics": {
                split: {name: metrics.as_dict() for name, metrics in models.items()}
                for split, models in self.overall.items()
            },
            "breakdowns": {
                name: frame.to_dict(orient="records") for name, frame in self.breakdowns.items()
            },
            "bootstrap": {
                name: [result.as_dict() for result in results]
                for name, results in self.bootstrap.items()
            },
            "model": self.model_info,
            "feature_importance_gain": self.importance,
        }


def metrics_by_group(
    predictions: pd.DataFrame, column: str, models: Mapping[str, str]
) -> pd.DataFrame:
    """Log-loss and accuracy of each model within each value of ``column``."""
    records = []
    for value, group in predictions.groupby(column, sort=True):
        record: dict[str, Any] = {column: value, "n": len(group)}
        for prob_column, label in models.items():
            metrics = compute_metrics(group["y"], group[prob_column])
            record[f"{label} log-loss"] = metrics.log_loss
            record[f"{label} accuracy"] = metrics.accuracy
        records.append(record)
    return pd.DataFrame(records)


def build_report(
    predictions: pd.DataFrame,
    *,
    tour: str,
    data_cutoff: str,
    row_counts: dict[str, int],
    model_info: dict[str, Any] | None = None,
    importance: dict[str, float] | None = None,
    n_resamples: int = 2000,
) -> Report:
    """Score every model on the validation and test years."""
    overall: dict[str, dict[str, Metrics]] = {}
    for split in ("val", "test"):
        part = predictions.loc[predictions["split"] == split]
        overall[split] = {
            label: compute_metrics(part["y"], part[column]) for column, label in MODELS.items()
        }
    test = predictions.loc[predictions["split"] == "test"]
    focus = {"p_elo": "Elo", "p_logistic": "Logistic", "p_xgboost": "XGBoost"}
    breakdowns = {
        name: metrics_by_group(test, column, focus) for column, name in BREAKDOWNS.items()
    }
    bootstrap = {
        f"{MODELS[model]} vs {MODELS[reference]}": paired_bootstrap(
            test["y"], test[model], test[reference], n_resamples=n_resamples
        )
        for model, reference in COMPARISONS
    }
    return Report(
        tour=tour,
        data_cutoff=data_cutoff,
        row_counts=row_counts,
        overall=overall,
        breakdowns=breakdowns,
        bootstrap=bootstrap,
        model_info=model_info or {},
        importance=importance or {},
    )


def _fmt(value: float, digits: int = 4) -> str:
    return "-" if math.isnan(value) else f"{value:.{digits}f}"


def _metrics_table(models: Mapping[str, Metrics]) -> list[str]:
    lines = [
        "| Model | Log-loss ↓ | Brier ↓ | Accuracy ↑ | ROC-AUC ↑ | ECE ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    best = min(models.values(), key=lambda metrics: metrics.log_loss)
    for name, metrics in models.items():
        cells = [
            _fmt(metrics.log_loss),
            _fmt(metrics.brier),
            f"{metrics.accuracy:.1%}",
            _fmt(metrics.roc_auc, 3),
            _fmt(metrics.ece, 3),
        ]
        if metrics is best:
            cells[0] = f"**{cells[0]}**"
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def _frame_table(frame: pd.DataFrame) -> list[str]:
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    align = (
        "|" + "|".join("---" if index == 0 else "---:" for index in range(len(frame.columns))) + "|"
    )
    rows = []
    for record in frame.itertuples(index=False):
        cells = []
        for column, value in zip(frame.columns, record, strict=True):
            if str(column).endswith("accuracy"):
                cells.append(f"{value:.1%}")
            elif str(column).endswith("log-loss"):
                cells.append(f"{value:.4f}")
            else:
                cells.append(f"{value:,}" if isinstance(value, int) else str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return [header, align, *rows]


def render_markdown(report: Report) -> str:
    """Human-readable Markdown version of the report."""
    tour = report.tour.upper()
    counts = report.row_counts
    lines = [
        f"# {tour} evaluation report",
        "",
        f"Chronological split: train {counts.get('train', 0):,} matches, validation "
        f"{counts.get('val', 0):,}, test {counts.get('test', 0):,}. "
        f"Data cutoff: {report.data_cutoff}.",
        "Rows are completed tour-level main-draw matches (retirements and walkovers excluded).",
        "",
        "## Test set",
        "",
        *_metrics_table(report.overall["test"]),
        "",
        "## Validation set",
        "",
        "Used for early stopping, model selection and calibration.",
        "",
        *_metrics_table(report.overall["val"]),
        "",
        "## Paired comparisons (test set, bootstrap 95% CI)",
        "",
        "Mean per-match difference, first model minus second: negative values favour the "
        "first model. An interval containing 0 means no significant difference.",
        "",
        "| Comparison | Metric | Difference | 95% CI |",
        "|---|---|---:|---|",
    ]
    for model, results in report.bootstrap.items():
        lines += [
            f"| {model} | {r.metric} | {r.difference:+.4f} | [{r.lower:+.4f}, {r.upper:+.4f}] |"
            for r in results
        ]
    for name, frame in report.breakdowns.items():
        lines += ["", f"## Test set by {name.lower()}", "", *_frame_table(frame)]
    if report.model_info:
        lines += [
            "",
            "## Selected model",
            "",
            "```json",
            json.dumps(report.model_info, indent=2),
            "```",
        ]
    return "\n".join(lines) + "\n"


def _style_axes(ax: Axes, palette: Mapping[str, Any]) -> None:
    ax.set_facecolor(palette["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(palette["axis"])
    ax.tick_params(colors=palette["muted"], labelcolor=palette["ink_secondary"])
    ax.grid(True, color=palette["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def plot_reliability(
    predictions: pd.DataFrame, out: Path, *, theme: Theme = "light", n_bins: int = 10
) -> None:
    """Reliability diagram of the Elo, logistic and XGBoost probabilities on the test set."""
    palette = _THEMES[theme]
    test = predictions.loc[predictions["split"] == "test"]
    fig = Figure(figsize=(6.4, 5.2), dpi=150, facecolor=palette["surface"])
    ax = fig.add_subplot()
    _style_axes(ax, palette)
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle=(0, (4, 4)),
        linewidth=1.2,
        color=palette["muted"],
        label="Perfect calibration",
    )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for color, column in zip(palette["series"], ("p_elo", "p_logistic", "p_xgboost"), strict=True):
        proba = test[column].to_numpy()
        target = test["y"].to_numpy()
        bins = np.clip(np.digitize(proba, edges) - 1, 0, n_bins - 1)
        xs, ys = [], []
        for index in range(n_bins):
            mask = bins == index
            if mask.sum() >= 20:
                xs.append(proba[mask].mean())
                ys.append(target[mask].mean())
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=palette["surface"],
            markeredgewidth=1.5,
            label=MODELS[column],
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability that player A wins", color=palette["ink_secondary"])
    ax.set_ylabel("Observed win rate", color=palette["ink_secondary"])
    ax.set_title(
        f"{test.attrs.get('tour', '').upper()} calibration on the test years".strip(),
        color=palette["ink"],
        loc="left",
        fontsize=12,
    )
    legend = ax.legend(frameon=False, loc="upper left")
    for text in legend.get_texts():
        text.set_color(palette["ink_secondary"])
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=palette["surface"], metadata={"Software": None})


def plot_feature_importance(
    importance: Mapping[str, float], out: Path, *, theme: Theme = "light", top: int = 15
) -> None:
    """Horizontal bars of the most important XGBoost features (share of total gain)."""
    palette = _THEMES[theme]
    total = sum(importance.values()) or 1.0
    items = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:top][::-1]
    names = [name for name, _ in items]
    shares = [value / total for _, value in items]
    fig = Figure(figsize=(6.4, 0.32 * len(items) + 1.2), dpi=150, facecolor=palette["surface"])
    ax = fig.add_subplot()
    _style_axes(ax, palette)
    ax.grid(True, axis="x", color=palette["grid"], linewidth=0.8)
    ax.grid(False, axis="y")
    ax.barh(names, shares, height=0.6, color=palette["series"][0])
    for y, share in enumerate(shares):
        ax.text(share, y, f" {share:.0%}", va="center", fontsize=8, color=palette["ink_secondary"])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of total gain", color=palette["ink_secondary"])
    ax.set_title("XGBoost feature importance", color=palette["ink"], loc="left", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=palette["surface"], metadata={"Software": None})


def write_report(report: Report, predictions: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write ``metrics.json``, ``metrics.md`` and the figures (light and dark variants)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "metrics.json", out_dir / "metrics.md"]
    written[0].write_text(
        json.dumps(report.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
    )
    written[1].write_text(render_markdown(report), encoding="utf-8")
    predictions = predictions.copy()
    predictions.attrs["tour"] = report.tour
    figures = out_dir / "figures"
    for theme in ("light", "dark"):
        reliability = figures / f"reliability_{theme}.png"
        plot_reliability(predictions, reliability, theme=theme)
        written.append(reliability)
        if report.importance:
            importance = figures / f"feature_importance_{theme}.png"
            plot_feature_importance(report.importance, importance, theme=theme)
            written.append(importance)
    return written
