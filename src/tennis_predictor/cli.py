"""Command-line interface: ``tennis --help``."""

from __future__ import annotations

import datetime as dt
import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from tennis_predictor import __version__
from tennis_predictor.config import TIERS_BY_TOUR, load_settings
from tennis_predictor.data.download import download_files, plan_downloads
from tennis_predictor.data.schema import SURFACES, day_to_date
from tennis_predictor.players import AmbiguousPlayerError, PlayerNotFoundError, search_players
from tennis_predictor.predict import Event, MatchupRequest, Predictor

app = typer.Typer(
    name="tennis",
    help="Leakage-free pre-match tennis winner prediction (ATP / WTA).",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
data_app = typer.Typer(help="Manage the raw Sackmann data.", no_args_is_help=True)
players_app = typer.Typer(help="Look up players known to a trained model.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(players_app, name="players")

console = Console()
error_console = Console(stderr=True)


class TourOption(StrEnum):
    """Tour selector."""

    ATP = "atp"
    WTA = "wta"


class DownloadTour(StrEnum):
    """Tour selector for downloads."""

    ATP = "atp"
    WTA = "wta"
    ALL = "all"


class SurfaceOption(StrEnum):
    """Court surface."""

    HARD = "hard"
    CLAY = "clay"
    GRASS = "grass"


ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML configuration file."),
]
ModelsDirOption = Annotated[Path, typer.Option(help="Directory containing the trained models.")]
TourArg = Annotated[TourOption, typer.Option("--tour", "-t", help="Tour of the trained model.")]


def _version(value: bool) -> None:
    if value:
        console.print(f"tennis-predictor {__version__}")
        raise typer.Exit


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logs.")] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Configure logging for every command."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=error_console, show_path=False)],
        force=True,
    )


def _fail(message: str) -> typer.Exit:
    error_console.print(f"[bold red]Error:[/] {message}")
    return typer.Exit(code=1)


@data_app.command("download")
def download(
    tour: Annotated[DownloadTour, typer.Option(help="Which tour to download.")] = DownloadTour.ALL,
    from_year: Annotated[int, typer.Option(help="First season.")] = 1968,
    to_year: Annotated[int, typer.Option(help="Last season.")] = 2024,
    dest: Annotated[Path, typer.Option(help="Destination directory.")] = Path("data/raw"),
    ref: Annotated[
        str, typer.Option(help="Git branch, tag or commit of the upstream repositories.")
    ] = "master",
    force: Annotated[bool, typer.Option(help="Overwrite files that already exist.")] = False,
) -> None:
    """Download match files from Jeff Sackmann's GitHub repositories (CC BY-NC-SA 4.0)."""
    tours = ["atp", "wta"] if tour is DownloadTour.ALL else [tour.value]
    for name in tours:
        tiers = sorted(TIERS_BY_TOUR[name], key=lambda tier: tier != "main")
        files = plan_downloads(name, tiers, range(from_year, to_year + 1))  # type: ignore[arg-type]
        report = download_files(files, dest, ref=ref, overwrite=force)
        console.print(
            f"{name.upper()}: {len(report.downloaded)} downloaded, "
            f"{len(report.skipped)} already present, "
            f"{len(report.missing)} not published upstream, {len(report.failed)} failed"
        )
        if not report.ok:
            raise _fail("some files could not be downloaded; see the warnings above")


@app.command()
def train(
    config: ConfigOption,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Re-read the CSV files.")] = False,
    report: Annotated[bool, typer.Option(help="Also write the evaluation report.")] = True,
) -> None:
    """Build features, fit the models and save the artifacts (and the report)."""
    from tennis_predictor.pipeline import run_training

    settings = load_settings(config)
    metadata = run_training(settings, use_cache=not no_cache)
    console.print(
        f"[green]Trained {settings.tour.upper()} model[/] "
        f"on {metadata.row_counts['train']:,} matches "
        f"(data until {metadata.data_cutoff}); artifacts in {settings.model_dir}"
    )
    if report:
        _evaluate(config)


def _evaluate(config: Path) -> None:
    from tennis_predictor.pipeline import run_evaluation

    settings = load_settings(config)
    result, written = run_evaluation(settings)
    table = Table(
        title=f"{settings.tour.upper()} test set ({result.row_counts.get('test', 0):,} matches)"
    )
    for column in ("Model", "Log-loss", "Brier", "Accuracy", "ROC-AUC", "ECE"):
        table.add_column(column, justify="left" if column == "Model" else "right")
    for name, metrics in result.overall["test"].items():
        table.add_row(
            name,
            f"{metrics.log_loss:.4f}",
            f"{metrics.brier:.4f}",
            f"{metrics.accuracy:.1%}",
            f"{metrics.roc_auc:.3f}",
            f"{metrics.ece:.3f}",
        )
    console.print(table)
    console.print(f"Report written to {written[1].parent}")


@app.command()
def evaluate(config: ConfigOption) -> None:
    """Score the saved validation/test predictions and write the report."""
    try:
        _evaluate(config)
    except FileNotFoundError as error:
        raise _fail(str(error)) from error


def _load_predictor(models_dir: Path, tour: TourOption) -> Predictor:
    try:
        return Predictor.from_directory(models_dir / tour.value)
    except FileNotFoundError as error:
        raise _fail(f"{error} (looked in {models_dir / tour.value})") from error


@app.command()
def predict(
    player_a: Annotated[str, typer.Argument(help="Name or id of player A.")],
    player_b: Annotated[str, typer.Argument(help="Name or id of player B.")],
    tour: TourArg = TourOption.ATP,
    surface: Annotated[SurfaceOption, typer.Option(help="Court surface.")] = SurfaceOption.HARD,
    best_of: Annotated[
        int | None, typer.Option(help="3 or 5 (default: 5 at ATP Grand Slams).")
    ] = None,
    event: Annotated[Event, typer.Option(help="Kind of tournament.")] = Event.TOUR,
    round_: Annotated[str, typer.Option("--round", help="Round code, e.g. R32, QF, F.")] = "R32",
    as_of: Annotated[
        str | None, typer.Option(help="Prediction date YYYY-MM-DD (default: the data cutoff).")
    ] = None,
    models_dir: ModelsDirOption = Path("models"),
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    """Predict the winner of a match between two players."""
    predictor = _load_predictor(models_dir, tour)
    try:
        date = dt.date.fromisoformat(as_of) if as_of else None
        request = MatchupRequest(player_a, player_b, surface.value, best_of, event, round_, date)
        result = predictor.predict(request)
    except (AmbiguousPlayerError, PlayerNotFoundError, ValueError) as error:
        raise _fail(str(error)) from error

    if as_json:
        payload = {
            "player_a": {"id": result.a.player_id, "name": result.a.name},
            "player_b": {"id": result.b.player_id, "name": result.b.name},
            "probability_a": round(result.probability_a, 4),
            "elo_probability_a": round(result.elo_probability_a, 4),
            "surface": result.surface,
            "best_of": result.best_of,
            "as_of": result.as_of.isoformat(),
            "top_factors": [
                {"feature": name, "log_odds": round(value, 4)} for name, value in result.top_factors
            ],
        }
        console.print_json(json.dumps(payload))
        return

    table = Table(
        title=f"{result.a.name} vs {result.b.name} - {result.surface}, best of {result.best_of}"
    )
    table.add_column("Player")
    table.add_column("Model", justify="right")
    table.add_column("Elo only", justify="right")
    table.add_row(result.a.name, f"{result.probability_a:.1%}", f"{result.elo_probability_a:.1%}")
    table.add_row(
        result.b.name, f"{1 - result.probability_a:.1%}", f"{1 - result.elo_probability_a:.1%}"
    )
    console.print(table)
    factors = ", ".join(f"{name} ({value:+.2f})" for name, value in result.top_factors)
    console.print(
        f"Favourite: [bold]{result.favourite.name}[/]. Main factors for {result.a.name}: {factors}"
    )
    console.print(f"[dim]State as of {result.as_of} (data cutoff {predictor.data_cutoff}).[/]")
    if (result.as_of - predictor.data_cutoff).days > 60:
        console.print(
            "[yellow]Warning:[/] predicting far beyond the data cutoff; form and ranks are stale."
        )


@players_app.command("search")
def players_search(
    query: Annotated[str, typer.Argument(help="Part of a player's name.")],
    tour: TourArg = TourOption.ATP,
    limit: Annotated[int, typer.Option(min=1, max=50)] = 5,
    models_dir: ModelsDirOption = Path("models"),
) -> None:
    """Fuzzy-search players by name."""
    predictor = _load_predictor(models_dir, tour)
    matches = search_players(predictor.players, query, limit=limit)
    if not matches:
        raise _fail(f"no player matches {query!r}")
    table = Table(title=f"Players matching {query!r}")
    for column in ("Id", "Name", "Country", "Elo", "Matches", "Last match", "Score"):
        table.add_column(column, justify="right" if column not in {"Name", "Country"} else "left")
    for player, score in matches:
        table.add_row(
            str(player.player_id),
            player.name,
            player.ioc,
            f"{player.elo:.0f}",
            str(player.matches),
            str(player.last_played or "-"),
            f"{score:.0f}",
        )
    console.print(table)


@app.command()
def ratings(
    tour: TourArg = TourOption.ATP,
    surface: Annotated[
        SurfaceOption | None, typer.Option(help="Rank by surface Elo instead.")
    ] = None,
    top: Annotated[int, typer.Option(min=1, max=200)] = 20,
    active_days: Annotated[int, typer.Option(help="Only players active in this window.")] = 365,
    models_dir: ModelsDirOption = Path("models"),
) -> None:
    """Elo leaderboard at the data cutoff."""
    predictor = _load_predictor(models_dir, tour)
    tracker = predictor.artifacts.tracker
    surface_index = SURFACES.index(surface.value.capitalize()) if surface else None
    rows = []
    for player_id, state in tracker.players.items():
        if state.last_day < 0 or tracker.last_day - state.last_day > active_days:
            continue
        rating = state.elo if surface_index is None else state.surface_elo[surface_index]
        rows.append((rating.value, player_id, state, rating.matches))
    rows.sort(key=lambda row: row[0], reverse=True)
    label = f"{surface.value} " if surface else ""
    table = Table(title=f"{tour.value.upper()} {label}Elo, {day_to_date(tracker.last_day)}")
    for column in ("#", "Player", "Country", "Elo", "Matches"):
        table.add_column(column, justify="left" if column in {"Player", "Country"} else "right")
    for position, (value, _, state, matches) in enumerate(rows[:top], start=1):
        table.add_row(str(position), state.name, state.ioc, f"{value:.0f}", str(matches))
    console.print(table)
