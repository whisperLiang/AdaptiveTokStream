import numpy as np
import pytest

from adaptivetokstream.asset import VideoAsset
from adaptivetokstream.playback import PlaybackSession


def asset(tmp_path):
    meta = {
        "frame_count": 16,
        "fps": "30",
        "model_id": "61" * 16,
        "pts_us": [round(i / 30 * 1e6) for i in range(16)],
    }
    return VideoAsset(tmp_path, meta, np.ones((1, 4, 256), dtype=np.uint16))


def test_no_receive_or_decode_completion_means_no_playback(tmp_path):
    a = asset(tmp_path)
    s = PlaybackSession(a, initial_buffer=0.1, required_prefix=8)
    s.advance(0.1)
    assert not s.played and s.stall_start == 0.1
    s.append(a.packet(0, 0, 8), 0.2)
    job = s.begin_decode(0.2)
    s.advance(0.3)
    assert not s.played
    s.complete_decode(job, 0.4, elapsed_ms=200)
    s.advance(0.4)
    assert s.played[0]["stall_seconds"] == pytest.approx(0.3)
    assert s.played[0]["prefix"] == 8
    assert s.buffer.block(0, 0).played


def test_played_enhancement_and_stale_history_decode(tmp_path):
    a = asset(tmp_path)
    s = PlaybackSession(a, initial_buffer=1, required_prefix=8)
    s.append(a.packet(0, 0, 8), 0)
    s.append(a.packet(1, 0, 8), 0)
    j = s.begin_decode(0)
    s.complete_decode(j, 0.01, 10)
    j = s.begin_decode(0.01)
    s.complete_decode(j, 0.02, 10)
    s.append(a.packet(0, 8, 16), 0.03)
    j = s.begin_decode(0.03)
    s.complete_decode(j, 0.04, 10)
    s.advance(1)
    assert s.played[0]["prefix"] == 16
    assert s.append(a.packet(0, 16, 32), 1.01).status == "played"
    s.advance(1.133334)
    assert len(s.played) == 1
    j = s.begin_decode(1.134)
    assert j.index == 1


def test_expired_enhancement_keeps_base(tmp_path):
    a = asset(tmp_path)
    s = PlaybackSession(a, 1, 8)
    s.append(a.packet(0, 0, 8), 0)
    assert s.append(a.packet(0, 8, 16, deadline_us=1000), 0.002).status == "expired"
    assert s.buffer.prefix(0, 0) == 8


def test_late_decode_completion_records_miss_without_prior_timer_wakeup(tmp_path):
    a = asset(tmp_path)
    s = PlaybackSession(a, initial_buffer=0.1, required_prefix=8)
    s.append(a.packet(0, 0, 8), 0)
    job = s.begin_decode(0)
    # The OS may resume the receiver after its timer deadline has already passed.
    s.complete_decode(job, 0.4, elapsed_ms=400)
    s.advance(0.4)
    assert s.played[0]["stall_seconds"] == pytest.approx(0.3)
    assert s.stall_count == 1 and 0 in s.deadline_misses
