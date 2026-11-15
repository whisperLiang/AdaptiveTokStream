from pathlib import Path

import pytest

from adaptivetokstream.network import BandwidthTrace, ThroughputEstimator, TraceLink


def test_trace_bytes_cross_bandwidth_drop_recovery():
    trace = BandwidthTrace([0, 1, 2, 3], [8, 8, 0.8, 8])
    assert trace.finish(1_000_000, 0) == pytest.approx(1)
    assert trace.finish(200_000, 0.9) == pytest.approx(2)
    assert trace.finish(300_000, 0.9) == pytest.approx(2.1)
    assert trace.times == [0, 1, 2, 3]


def test_swift_loader_preserves_original_file_samples():
    root = Path(__file__).resolve().parents[1]
    p = root / "third_party/swift/streamer/datasets/test/oboe_trace_2"
    trace = BandwidthTrace.from_swift(p, root / "third_party/swift")
    expected = [list(map(float, line.split()[:2])) for line in p.read_text().splitlines()]
    assert trace.times == [x[0] for x in expected]
    assert trace.mbps == [x[1] for x in expected]


def test_persistent_connection_no_per_packet_handshake_and_seed():
    t = BandwidthTrace([0, 10], [8, 8])
    a = TraceLink(t, rtt_ms=80, jitter_ms=0, seed=1)
    first = a.send(1000, 0.08)
    second = a.send(1000, first.service_end)
    assert second.delivery - first.delivery == pytest.approx(0.001)
    assert first.delivery == pytest.approx(0.121)
    b = TraceLink(t, rtt_ms=10, jitter_ms=2, seed=7)
    c = TraceLink(t, rtt_ms=10, jitter_ms=2, seed=7)
    assert b.send(1000, 0) == c.send(1000, 0)


def test_estimator_uses_completed_transfers_only():
    e = ThroughputEstimator(1000, 0.5)
    assert e.estimate == 1000
    e.observe(1000, 2)
    assert e.estimate == 750
    assert e.recent == [500]


@pytest.mark.parametrize("times,bw", [([0, 0], [1, 1]), ([0, 1], [-1, 2]), ([0, 1], [0, 0])])
def test_invalid_trace(times, bw):
    with pytest.raises(ValueError):
        BandwidthTrace(times, bw)
