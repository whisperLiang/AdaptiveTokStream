import numpy as np

from adaptivetokstream.asset import VideoAsset
from adaptivetokstream.config import Config
from adaptivetokstream.network import BandwidthTrace
from adaptivetokstream.scheduler import RDTable
from adaptivetokstream.simulator import simulate


def fixtures(tmp_path):
    meta = {
        "frame_count": 32,
        "fps": "30",
        "model_id": "61" * 16,
        "pts_us": [round(i / 30 * 1e6) for i in range(32)],
    }
    a = VideoAsset(tmp_path, meta, np.ones((2, 4, 256), dtype=np.uint16))
    rows = [
        {
            "window_id": i // 4,
            "block_id": i % 4,
            "token_budget": k,
            "mse": 1 / k,
            "decode_time_ms": 30,
            "decode_p95_ms": 30,
        }
        for i in range(8)
        for k in [8, 16, 32, 64, 128, 256]
    ]
    return a, RDTable(rows)


def test_packet_arrival_decode_and_playback_are_ordered(tmp_path):
    a, rd = fixtures(tmp_path)
    c = Config()
    c.streaming.policy = "fixed"
    c.streaming.fixed_tokens = 32
    c.streaming.initial_buffer_seconds = 0.05
    c.network.rtt_ms = 10
    r = simulate(a, rd, c, BandwidthTrace.constant(0.01))
    events = r["events"]
    for i in range(8):
        recv = next(
            e["time"] for e in events if e["event"] == "PACKET_RECEIVED" and e["block"] == i
        )
        decode = next(
            e["time"] for e in events if e["event"] == "DECODE_COMPLETED" and e["block"] == i
        )
        play = next(e["time"] for e in events if e["event"] == "BLOCK_PLAYED" and e["block"] == i)
        assert recv <= decode <= play
    assert r["summary"]["sent_video_bytes"] == r["summary"]["received_video_bytes"]
    assert all(x["prefix"] == 32 for x in r["playback"])
    assert r["summary"]["duplicate_payload_bytes"] == 0


def test_congestion_and_slow_decoder_produce_real_stall(tmp_path):
    a, rd = fixtures(tmp_path)
    c = Config()
    c.streaming.initial_buffer_seconds = 0
    r = simulate(a, rd, c, BandwidthTrace([0, 0.3, 0.8, 2], [1, 1, 0.001, 1]), decode_scale=10)
    assert r["summary"]["stall_seconds"] > 0
    assert r["summary"]["stall_count"] > 0
    assert any(e["event"] == "BLOCK_DEADLINE_EXPIRED" for e in r["events"])
    assert r["summary"]["blocks_played"] == 8


def test_seed_reproducible_no_resend_of_prefix(tmp_path):
    a, rd = fixtures(tmp_path)
    c = Config()
    c.network.jitter_ms = 3
    r = simulate(a, rd, c, BandwidthTrace.constant(0.02))
    q = simulate(a, RDTable(list(rd.records.values())), c, BandwidthTrace.constant(0.02))
    assert r["playback"] == q["playback"]
    ends = {}
    for p in r["packets"]:
        assert p["start"] == ends.get(p["block"], 0)
        ends[p["block"]] = p["end"]
