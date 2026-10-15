import subprocess
import sys

import tennis_predictor


def test_version_is_exposed() -> None:
    assert tennis_predictor.__version__.count(".") == 2


def test_module_entry_point_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tennis_predictor", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert tennis_predictor.__version__ in result.stdout
