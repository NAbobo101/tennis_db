from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tennis_predictor.config import Settings, load_settings
from tennis_predictor.data.loader import load_matches
from tennis_predictor.features.dataset import Dataset, build_dataset
from tennis_predictor.features.tracker import FeatureTracker
from tests.synthetic import write_settings, write_synthetic_tour


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic")
    write_synthetic_tour(root / "raw")
    return root


@pytest.fixture(scope="session")
def config_path(synthetic_root: Path) -> Path:
    return write_settings(synthetic_root / "atp.yaml", synthetic_root)


@pytest.fixture(scope="session")
def settings(config_path: Path) -> Settings:
    return load_settings(config_path)


@pytest.fixture(scope="session")
def matches(settings: Settings) -> pd.DataFrame:
    return load_matches(settings.data, settings.tour, use_cache=False)


@pytest.fixture(scope="session")
def built(matches: pd.DataFrame, settings: Settings) -> tuple[Dataset, FeatureTracker]:
    return build_dataset(matches, settings)


@pytest.fixture(scope="session")
def dataset(built: tuple[Dataset, FeatureTracker]) -> Dataset:
    return built[0]
