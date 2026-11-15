import pytest

from adaptivetokstream.evaluation import validate_playback


def test_replay_rejects_impossible_mask_history():
    good = [
        {"block": 0, "prefix": 8, "history_budgets": []},
        {"block": 1, "prefix": 32, "history_budgets": [8]},
    ]
    validate_playback(good, 2)
    bad = [good[0], {**good[1], "history_budgets": [256]}]
    with pytest.raises(ValueError):
        validate_playback(bad, 2)
