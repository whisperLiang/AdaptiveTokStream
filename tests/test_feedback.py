from adaptivetokstream.socket_runtime import merge_feedback


def test_incremental_feedback_merges_prefixes_and_removes_played_blocks():
    first = {"blocks": [{"index": 0, "prefix": 8}, {"index": 1, "prefix": 0}], "played_blocks": 0}
    update = {"blocks": [{"index": 1, "prefix": 32}], "played_blocks": 1}
    result = merge_feedback(first, update)
    assert result["blocks"] == [{"index": 1, "prefix": 32}]


def test_sender_paces_service_without_per_packet_rtt(tmp_path):
    from dataclasses import asdict

    import numpy as np

    from adaptivetokstream.asset import VideoAsset
    from adaptivetokstream.config import Config
    from adaptivetokstream.network import BandwidthTrace, TraceLink
    from adaptivetokstream.scheduler import BlockView, RDTable
    from adaptivetokstream.socket_runtime import SenderRuntime

    meta = {
        "frame_count": 16,
        "fps": "30",
        "pts_us": [round(i / 30 * 1e6) for i in range(16)],
        "model_id": "61" * 16,
    }
    asset = VideoAsset(tmp_path, meta, np.ones((1, 4, 256), dtype=np.uint16))
    rows = [
        {"window_id": 0, "block_id": g, "token_budget": k, "mse": 1 / k, "decode_time_ms": 30}
        for g in range(4)
        for k in [8, 16, 32, 64, 128, 256]
    ]
    sender = SenderRuntime(Config(), asset, RDTable(rows), tmp_path / "sender")
    sender.pacer = TraceLink(BandwidthTrace.constant(8), rtt_ms=80)
    sender.feedback = {
        "blocks": [asdict(BlockView(i, 0, 1 + i / 30 * 4, 4 / 30)) for i in range(4)],
        "buffer_seconds": 0,
        "active_decode": None,
        "last_played": None,
    }
    clock = [0.0]
    sender.now = lambda: clock[0]
    sender.choose_next_packet()
    clock[0] = sender.pacer.available_at + 1e-8
    sender.choose_next_packet()
    assert len(sender.outbound) == 2
    assert sender.outbound[1]["requested"] < sender.outbound[0]["send_at"]
