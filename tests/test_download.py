from __future__ import annotations

from pathlib import Path

import pytest

from tennis_predictor.data.download import (
    DownloadError,
    RemoteFile,
    download_files,
    fetch_url,
    plan_downloads,
)


def test_plan_downloads() -> None:
    files = plan_downloads("atp", ("main", "futures"), range(2023, 2025))
    names = [file.name for file in files]
    assert names[:4] == [
        "atp_matches_2023.csv",
        "atp_matches_futures_2023.csv",
        "atp_matches_2024.csv",
        "atp_matches_futures_2024.csv",
    ]
    assert "atp_players.csv" in names
    assert {file.repo for file in files} == {"tennis_atp"}
    assert files[0].url("master") == (
        "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv"
    )
    assert plan_downloads("wta", ("main",), range(2024, 2025), include_extras=False) == [
        RemoteFile("tennis_wta", "wta_matches_2024.csv")
    ]


def test_download_files(tmp_path: Path) -> None:
    files = [
        RemoteFile("tennis_atp", "ok.csv"),
        RemoteFile("tennis_atp", "missing.csv"),
        RemoteFile("tennis_atp", "broken.csv"),
        RemoteFile("tennis_atp", "existing.csv"),
    ]
    (tmp_path / "tennis_atp").mkdir()
    (tmp_path / "tennis_atp" / "existing.csv").write_text("old", encoding="utf-8")
    requested: list[str] = []

    def fake_fetch(url: str) -> bytes:
        requested.append(url)
        if url.endswith("missing.csv"):
            raise FileNotFoundError(url)
        if url.endswith("broken.csv"):
            raise DownloadError("HTTP 500")
        return b"a,b\n1,2\n"

    report = download_files(files, tmp_path, ref="abc123", fetch=fake_fetch)
    assert [path.name for path in report.downloaded] == ["ok.csv"]
    assert [path.name for path in report.skipped] == ["existing.csv"]
    assert [file.name for file in report.missing] == ["missing.csv"]
    assert [file.name for file, _ in report.failed] == ["broken.csv"]
    assert not report.ok
    assert all("/abc123/" in url for url in requested)
    assert (tmp_path / "tennis_atp" / "ok.csv").read_bytes() == b"a,b\n1,2\n"
    assert not list(tmp_path.rglob("*.part"))

    report = download_files(files[3:], tmp_path, overwrite=True, fetch=fake_fetch)
    assert report.ok
    assert (tmp_path / "tennis_atp" / "existing.csv").read_bytes() == b"a,b\n1,2\n"


def test_fetch_url_refuses_plain_http() -> None:
    with pytest.raises(DownloadError, match="non-HTTPS"):
        fetch_url("http://example.com/file.csv")
