import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.socket
@pytest.mark.model
def test_two_process_real_model_socket_video(tmp_path):
    asset = ROOT / "artifacts/road/encoded"
    rd = ROOT / "artifacts/road/profile/rd.json"
    if not (asset / "tokens.npy").exists() or not rd.exists():
        pytest.skip("prepare/encode/profile real sample video first; see README")
    data = tmp_path / "asset"
    data.mkdir()
    meta = json.loads((asset / "asset.json").read_text())
    meta["frame_count"] = 16
    meta["pts_us"] = meta["pts_us"][:16]
    (data / "asset.json").write_text(json.dumps(meta))
    np.save(data / "tokens.npy", np.load(asset / "tokens.npy")[:1], allow_pickle=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ready = tmp_path / "ready"
    common = [sys.executable, "-m", "adaptivetokstream"]
    with open(tmp_path / "receiver.log", "w") as log:
        receiver = subprocess.Popen(
            common
            + [
                "run-receiver",
                "--port",
                str(port),
                "--output",
                str(tmp_path / "receiver"),
                "--ready-file",
                str(ready),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while not ready.exists():
                if receiver.poll() is not None:
                    pytest.fail((tmp_path / "receiver.log").read_text())
                if time.monotonic() > deadline:
                    pytest.fail("Receiver did not become ready")
                time.sleep(0.05)
            sender = subprocess.run(
                common
                + [
                    "run-sender",
                    "--asset",
                    str(data),
                    "--rd",
                    str(rd),
                    "--policy",
                    "greedy",
                    "--port",
                    str(port),
                    "--output",
                    str(tmp_path / "sender"),
                ],
                capture_output=True,
                text=True,
                timeout=45,
            )
            assert sender.returncode == 0, sender.stdout + sender.stderr
            assert receiver.wait(timeout=15) == 0, (tmp_path / "receiver.log").read_text()
        finally:
            if receiver.poll() is None:
                receiver.terminate()
                receiver.wait(timeout=5)
    tx = json.loads((tmp_path / "sender/sender.json").read_text())
    rx = json.loads((tmp_path / "receiver/receiver.json").read_text())
    assert tx["summary"]["sent_video_bytes"] == rx["summary"]["received_video_bytes"]
    assert (
        tx["summary"]["application_bytes"]["sent"] == rx["summary"]["application_bytes"]["received"]
    )
    assert (
        tx["summary"]["application_bytes"]["received"] == rx["summary"]["application_bytes"]["sent"]
    )
    assert rx["summary"]["duplicate_payload_bytes"] == 0
    assert len(rx["playback"]) == 4
    with av.open(str(tmp_path / "receiver/reconstruction.nut")) as c:
        frames = list(c.decode(video=0))
    assert len(frames) == 16
    assert all((f.width, f.height) == (128, 128) for f in frames)
    actual = [round(float(f.pts * f.time_base) * 1e6) for f in frames]
    assert actual == meta["pts_us"]
    ends = {}
    for packet in tx["packets"]:
        assert packet["start"] == ends.get(packet["block"], 0)
        ends[packet["block"]] = packet["end"]
