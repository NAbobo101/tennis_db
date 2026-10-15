"""Leakage-free pre-match tennis winner prediction on Jeff Sackmann's ATP/WTA data."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tennis-predictor")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
