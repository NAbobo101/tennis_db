import tennis_predictor


def test_version_is_exposed() -> None:
    assert tennis_predictor.__version__.count(".") == 2
