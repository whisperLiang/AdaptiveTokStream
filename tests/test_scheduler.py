from dataclasses import replace

import pytest

from adaptivetokstream.config import Config
from adaptivetokstream.protocol import TokenPacker, TokenPacket
from adaptivetokstream.scheduler import (
    BlockView,
    DeadlineAwareMPC,
    FixedToken,
    GreedyToken,
    RDTable,
    SchedulerState,
)


def size(index, start, end):
    return TokenPacket(
        1,
        b"a" * 16,
        0,
        0,
        0,
        start,
        end - start,
        10**9,
        TokenPacker.pack_tokens([1] * (end - start)),
    ).wire_bytes


def table(negative=False):
    rows = []
    for i in range(4):
        for k in [8, 16, 32, 64, 128, 256]:
            rows.append(
                {
                    "window_id": i // 4,
                    "block_id": i % 4,
                    "token_budget": k,
                    "mse": (k / 1000 if negative else 1 / k),
                    "decode_time_ms": 30,
                    "decode_p95_ms": 30,
                }
            )
    return RDTable(rows)


def state(prefixes=(8, 8, 8, 8), deadline=1):
    return SchedulerState(
        0,
        100000,
        (),
        0.02,
        0.4,
        deadline,
        tuple(
            BlockView(i, p, deadline + i * 0.133, 0.133, False, p) for i, p in enumerate(prefixes)
        ),
    )


def test_fixed_budget_and_actual_byte_cost():
    c = Config()
    c.streaming.fixed_tokens = 64
    p = FixedToken(c, table(), size)
    a = p.choose(state((0, 0, 0, 0)))
    assert (a.block_index, a.target_budget) == (0, 64)
    assert a.size_bytes == size(0, 0, 64)


def test_greedy_rejects_nonpositive_quality_gain():
    assert GreedyToken(Config(), table(True), size).choose(state()) is None


@pytest.mark.parametrize("cls", [GreedyToken, DeadlineAwareMPC])
def test_base_playback_priority_and_no_duplicates(cls):
    p = cls(Config(), table(), size)
    a = p.choose(state((64, 0, 0, 0)))
    assert (a.block_index, a.target_budget) == (1, 8)
    a = p.choose(state())
    assert a.target_budget > 8 and a.start == 8
    s = state()
    s = replace(s, blocks=tuple(replace(b, played=True) for b in s.blocks))
    assert p.choose(s) is None


def test_decode_cost_excludes_infeasible_enhancement():
    c = Config()
    s = state(deadline=0.015)
    s = replace(s, blocks=(s.blocks[0],))
    assert GreedyToken(c, table(), size).choose(s) is None
    c.scheduler.use_decode_cost = False
    assert GreedyToken(c, table(), size).choose(s) is not None


def test_deadline_mpc_feedback_only_deterministic_and_byte_cap():
    c = Config()
    rd = table()
    a = DeadlineAwareMPC(c, rd, size)
    b = DeadlineAwareMPC(c, rd, size)
    assert a.choose(state()) == b.choose(state())
    assert "trace" not in vars(a)
    s = replace(state(), remaining_bytes=10)
    assert a.choose(s) is None


def test_infeasible_base_explicit_recovery_action():
    s = state((0, 8, 8, 8), deadline=-1)
    a = DeadlineAwareMPC(Config(), table(), size).choose(s)
    assert a.target_budget == 8 and a.deadline_feasible is False
