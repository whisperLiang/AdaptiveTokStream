import json
import math
import subprocess
import sys
from pathlib import Path

import av
import pytest
import yaml

from adaptivetokstream.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.model
def test_cli_offline_pipeline_profiles_real_frames_and_partial_window(tmp_path):
    source = Path("/home/whisperliang/RECAP/video_data/road.mp4")
    if not source.exists():
        pytest.skip("real source video unavailable; supply source from README")
    if not (ROOT / "artifacts/weights/AdapTok/model.safetensors").exists():
        pytest.skip("official checkpoint unavailable")
    config = load_config(ROOT / "configs/default.yaml")
    config.codec.device = "cuda:1"
    config.evaluation.repeats = 1
    config.evaluation.warmup = 1
    config.streaming.policy = "fixed"
    config.streaming.fixed_tokens = 32
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))

    def run(command, *args):
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "adaptivetokstream",
                command,
                "--config",
                str(path),
                *map(str, args),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert r.returncode == 0, r.stdout + r.stderr

    prepared = tmp_path / "prepared"
    encoded = tmp_path / "encoded"
    profile = tmp_path / "profile"
    sim = tmp_path / "sim"
    evaluated = tmp_path / "evaluation"
    run("prepare-video", "--input", source, "--frames", 19, "--output", prepared)
    run("encode-video", "--prepared", prepared, "--output", encoded)
    run("profile-codec", "--asset", encoded, "--output", profile)
    run("simulate", "--asset", encoded, "--rd", profile / "rd.json", "--output", sim)
    run("evaluate", "--asset", encoded, "--log", sim / "simulation.json", "--output", evaluated)
    assert json.loads((encoded / "asset.json").read_text())["encode_calls"] == 2
    rd = json.loads((profile / "rd.json").read_text())
    assert len(rd["records"]) == 5 * 6
    for row in rd["records"]:
        assert row["history_budgets"] == [256] * row["block_id"]
        assert all(math.isfinite(row[m]) for m in ["psnr", "lpips", "dists", "decode_time_ms"])
    result = json.loads((evaluated / "evaluation.json").read_text())
    assert result["summary"]["blocks_played"] == 5
    assert all(row["prefix"] == 32 for row in result["blocks"])
    for video in [
        evaluated / "reconstruction.nut",
        profile / "budget_8.nut",
        profile / "budget_256.nut",
    ]:
        with av.open(str(video)) as c:
            frames = list(c.decode(video=0))
        assert len(frames) == 19
