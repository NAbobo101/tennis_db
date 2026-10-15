from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tennis_predictor import __version__
from tennis_predictor.cli import app
from tennis_predictor.config import Settings
from tennis_predictor.data.download import DownloadError, DownloadReport, RemoteFile, download_files
from tests.synthetic import write_settings

runner = CliRunner()


def invoke(*args: str):
    return runner.invoke(app, list(args), catch_exceptions=False)


def assert_clean_failure(result, message: str) -> None:
    """Exit code 1 with a friendly message, not an unexpected exception."""
    output = " ".join(result.output.split())  # Rich wraps long lines
    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Error:" in output
    assert message in output
    assert "Traceback" not in output


def test_version() -> None:
    result = invoke("--version")
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help_lists_commands() -> None:
    result = invoke("--help")
    for command in ("train", "evaluate", "predict", "ratings", "players", "data"):
        assert command in result.output


@pytest.fixture(scope="module")
def models_dir(config_path: Path, settings: Settings) -> Path:
    result = invoke("train", "--config", str(config_path), "--no-cache")
    assert result.exit_code == 0, result.output
    assert "test set" in result.output
    return settings.models_dir


def test_evaluate_rewrites_the_report(
    models_dir: Path, config_path: Path, settings: Settings
) -> None:
    (settings.report_dir / "metrics.md").unlink()
    result = invoke("evaluate", "--config", str(config_path))
    assert result.exit_code == 0, result.output
    assert (settings.report_dir / "metrics.md").is_file()


def test_predict_is_symmetric(models_dir: Path) -> None:
    common = ["--models-dir", str(models_dir), "--surface", "clay", "--json"]
    forward = json.loads(invoke("predict", "Adam Archer", "Adam Fletcher", *common).output)
    backward = json.loads(invoke("predict", "Adam Fletcher", "Adam Archer", *common).output)
    assert forward["probability_a"] + backward["probability_a"] == pytest.approx(1.0, abs=2e-4)
    assert forward["surface"] == "Clay"
    assert forward["best_of"] == 3
    assert len(forward["top_factors"]) == 5


def test_predict_grand_slam_defaults_to_best_of_five(models_dir: Path) -> None:
    result = invoke(
        "predict",
        "Adam Archer",
        "Adam Fletcher",
        "--models-dir",
        str(models_dir),
        "--event",
        "grand-slam",
        "--round",
        "F",
    )
    assert result.exit_code == 0, result.output
    assert "best of 5" in result.output
    assert "Favourite" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["Twin", "Adam Archer"], "ambiguous"),
        (["Nobody Here", "Adam Archer"], "no player matches"),
        (["100001", "Adam Archer"], "same player"),
        (["Adam Archer", "Adam Fletcher", "--as-of", "2001-01-01"], "before the data cutoff"),
        (["Adam Archer", "Adam Fletcher", "--round", "Q9"], "unknown round"),
    ],
)
def test_predict_errors(models_dir: Path, args: list[str], message: str) -> None:
    result = runner.invoke(app, ["predict", *args, "--models-dir", str(models_dir)])
    assert_clean_failure(result, message)


def test_options_are_case_insensitive(models_dir: Path) -> None:
    result = invoke(
        "predict", "Adam Archer", "Adam Fletcher", "--models-dir", str(models_dir),
        "--tour", "ATP", "--surface", "Hard", "--event", "Grand-Slam", "--json",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["best_of"] == 5


def test_inactive_player_warning(models_dir: Path) -> None:
    result = invoke(
        "predict", "Carl Solo", "Adam Archer", "--models-dir", str(models_dir), "--json"
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["warnings"]
    table = invoke("predict", "Carl Solo", "Adam Archer", "--models-dir", str(models_dir))
    assert "has not played since" in table.output


def test_predict_warns_far_beyond_cutoff(models_dir: Path) -> None:
    result = invoke(
        "predict", "Adam Archer", "Solo", "--models-dir", str(models_dir), "--as-of", "2030-01-01"
    )
    assert result.exit_code == 0
    assert "stale" in result.output


def test_players_search(models_dir: Path) -> None:
    result = invoke("players", "search", "twin", "--models-dir", str(models_dir))
    assert result.exit_code == 0
    assert "Ana Twin" in result.output
    assert "Bea Twin" in result.output
    missing = runner.invoke(app, ["players", "search", "zzzzzz", "--models-dir", str(models_dir)])
    assert_clean_failure(missing, "no player matches")


def test_ratings(models_dir: Path) -> None:
    for extra in ([], ["--surface", "grass"]):
        result = invoke("ratings", "--models-dir", str(models_dir), "--top", "5", *extra)
        assert result.exit_code == 0
        assert "Elo" in result.output


def test_missing_model(tmp_path: Path) -> None:
    result = runner.invoke(app, ["predict", "A", "B", "--models-dir", str(tmp_path)])
    assert_clean_failure(result, "tennis train")


def test_missing_report_inputs(tmp_path: Path, config_path: Path) -> None:
    config = config_path.read_text(encoding="utf-8").replace(
        str(config_path.parent / "models"), str(tmp_path)
    )
    other = tmp_path / "atp.yaml"
    other.write_text(config, encoding="utf-8")
    result = runner.invoke(app, ["evaluate", "--config", str(other)])
    assert_clean_failure(result, "tennis train")


def test_train_without_data_fails_cleanly(tmp_path: Path) -> None:
    config = write_settings(tmp_path / "atp.yaml", tmp_path)
    result = runner.invoke(app, ["train", "--config", str(config)])
    assert_clean_failure(result, "tennis data download")


def test_invalid_config_fails_cleanly(tmp_path: Path) -> None:
    config = write_settings(tmp_path / "atp.yaml", tmp_path, split={"train_start": 2023})
    for command in ("train", "evaluate"):
        result = runner.invoke(app, [command, "--config", str(config)])
        assert_clean_failure(result, "split years must be ordered")


def test_data_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_fetch(url: str, *, timeout: float = 60.0) -> bytes:
        calls.append(url)
        if "futures" in url:
            raise FileNotFoundError(url)
        return b"x\n"

    def offline_download(
        files: Sequence[RemoteFile], dest: Path, *, ref: str = "master", overwrite: bool = False
    ) -> DownloadReport:
        return download_files(files, dest, ref=ref, overwrite=overwrite, fetch=fake_fetch)

    monkeypatch.setattr("tennis_predictor.cli.download_files", offline_download)
    result = invoke(
        "data",
        "download",
        "--tour",
        "atp",
        "--from-year",
        "2024",
        "--to-year",
        "2024",
        "--dest",
        str(tmp_path),
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "tennis_atp" / "atp_matches_2024.csv").is_file()
    assert "1 not published upstream" in result.output
    assert any(url.endswith("atp_players.csv") for url in calls)


def test_data_download_reports_failures_after_every_tour(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetched: list[str] = []

    def flaky(
        files: Sequence[RemoteFile], dest: Path, *, ref: str = "master", overwrite: bool = False
    ) -> DownloadReport:
        def fetch(url: str) -> bytes:
            fetched.append(url)
            if url.endswith("atp_players.csv"):
                raise DownloadError("HTTP 500")
            return b"x\n"

        return download_files(files, dest, ref=ref, overwrite=overwrite, fetch=fetch)

    monkeypatch.setattr("tennis_predictor.cli.download_files", flaky)
    result = runner.invoke(
        app,
        ["data", "download", "--from-year", "2024", "--to-year", "2024", "--dest", str(tmp_path)],
    )
    assert_clean_failure(result, "1 files could not be downloaded")
    assert any("tennis_wta" in url for url in fetched)  # WTA still ran after the ATP failure
