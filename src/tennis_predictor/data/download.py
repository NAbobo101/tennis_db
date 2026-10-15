"""Download the raw CSV files from Jeff Sackmann's GitHub repositories."""

from __future__ import annotations

import http.client
import logging
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tennis_predictor.config import Tier, Tour
from tennis_predictor.data.schema import FILE_FAMILIES

logger = logging.getLogger(__name__)

RAW_URL = "https://raw.githubusercontent.com/JeffSackmann/{repo}/{ref}/{name}"
EXTRA_FILES: dict[str, tuple[str, ...]] = {
    "atp": (
        "atp_players.csv",
        "atp_rankings_20s.csv",
        "atp_rankings_current.csv",
        "matches_data_dictionary.txt",
    ),
    "wta": ("wta_players.csv", "wta_rankings_20s.csv", "wta_rankings_current.csv"),
}


class DownloadError(RuntimeError):
    """Raised when a remote file cannot be fetched."""


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """One file inside an upstream repository."""

    repo: str
    name: str

    def url(self, ref: str) -> str:
        """Raw download URL of the file at a git reference."""
        return RAW_URL.format(repo=self.repo, ref=ref, name=self.name)


@dataclass(slots=True)
class DownloadReport:
    """Outcome of :func:`download_files`."""

    downloaded: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    missing: list[RemoteFile] = field(default_factory=list)
    failed: list[tuple[RemoteFile, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every file was downloaded, skipped or legitimately absent upstream."""
        return not self.failed


def plan_downloads(
    tour: Tour, tiers: Sequence[Tier], years: range, *, include_extras: bool = True
) -> list[RemoteFile]:
    """List the remote files needed for a tour, tiers and year range."""
    files = [
        RemoteFile(
            FILE_FAMILIES[(tour, tier)].repo, FILE_FAMILIES[(tour, tier)].pattern.format(year=year)
        )
        for year in years
        for tier in tiers
    ]
    if include_extras:
        repo = FILE_FAMILIES[(tour, "main")].repo
        files.extend(RemoteFile(repo, name) for name in EXTRA_FILES[tour])
    return files


def fetch_url(url: str, *, timeout: float = 60.0) -> bytes:
    """Fetch an HTTPS URL.

    Raises:
        FileNotFoundError: The server answered 404.
        DownloadError: Any other HTTP or network failure.
    """
    if not url.startswith("https://"):
        raise DownloadError(f"refusing non-HTTPS URL: {url}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if not response.geturl().startswith("https://"):
                raise DownloadError(f"refusing redirect to non-HTTPS URL {response.geturl()}")
            data: bytes = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise FileNotFoundError(url) from error
        raise DownloadError(f"HTTP {error.code} for {url}") from error
    except (OSError, http.client.HTTPException) as error:  # URLError, timeouts, resets
        raise DownloadError(f"cannot download {url}: {error}") from error
    return data


def _write_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def download_files(
    files: Sequence[RemoteFile],
    dest: Path,
    *,
    ref: str = "master",
    overwrite: bool = False,
    fetch: Callable[[str], bytes] = fetch_url,
) -> DownloadReport:
    """Download files into ``dest/<repo>/<name>``.

    Existing files are kept unless ``overwrite`` is set. Files absent upstream (404, e.g. a
    year that has no Futures events) are reported as missing rather than failing.
    """
    report = DownloadReport()
    for remote in files:
        target = dest / remote.repo / remote.name
        if target.exists() and not overwrite:
            report.skipped.append(target)
            continue
        try:
            data = fetch(remote.url(ref))
        except FileNotFoundError:
            report.missing.append(remote)
            continue
        except DownloadError as error:
            logger.warning("Failed to download %s: %s", remote.name, error)
            report.failed.append((remote, str(error)))
            continue
        _write_atomically(target, data)
        report.downloaded.append(target)
    return report
