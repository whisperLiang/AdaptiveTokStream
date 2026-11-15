import pytest

from adaptivetokstream.config import Config


@pytest.mark.parametrize(
    "field,value",
    [
        ("bandwidth_mbps", 0),
        ("initial_estimate_mbps", -1),
        ("rtt_ms", -1),
        ("jitter_ms", -1),
        ("ewma_alpha", 0),
        ("ewma_alpha", 1.1),
    ],
)
def test_invalid_network_configuration_rejected(field, value):
    config = Config()
    setattr(config.network, field, value)
    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("codec", "resolution", 256),
        ("codec", "budgets", [8, 32]),
        ("streaming", "mode", "live"),
        ("streaming", "policy", "unknown"),
        ("streaming", "initial_buffer_seconds", -1),
        ("scheduler", "horizon_blocks", 0),
        ("scheduler", "beam_width", 0),
        ("scheduler", "max_video_bytes", 0),
        ("evaluation", "warmup", -1),
        ("evaluation", "repeats", 0),
        ("evaluation", "history_budget", 12),
        ("evaluation", "metrics", ["psnr"]),
    ],
)
def test_inconsistent_domain_configuration_rejected(section, field, value):
    config = Config()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        config.validate()
