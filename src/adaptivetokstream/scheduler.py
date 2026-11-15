"""Pure scheduling over completed receiver feedback; no trace or model access."""

import itertools
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from .codec import BUDGETS


@dataclass(frozen=True)
class BlockView:
    index: int
    prefix: int
    deadline: float
    duration: float
    played: bool = False
    ready_prefix: int = 0
    pending_prefix: int = 0


@dataclass(frozen=True)
class SchedulerState:
    now: float
    estimated_bandwidth: float
    recent_throughput: tuple
    network_rtt: float
    playback_buffer: float
    next_playback_deadline: float
    blocks: tuple[BlockView, ...]
    decoder_available_at: float = 0.0
    remaining_bytes: int | None = None
    previous_distortion: float | None = None


@dataclass(frozen=True)
class Action:
    block_index: int
    start: int
    target_budget: int
    size_bytes: int
    deadline_feasible: bool


class RDTable:
    def __init__(self, records):
        self.records = {(r["window_id"] * 4 + r["block_id"], r["token_budget"]): r for r in records}
        self.decode_scale = 1.0

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text())["records"])

    def distortion(self, index, k, content=True):
        if not content:
            return 1 / max(k, 1)  # explicitly labeled bandwidth-only ablation proxy
        if k == 0:
            return 1.0
        return self.records[index, k]["mse"]

    def decode_seconds(self, index, k):
        r = self.records[index, k]
        return r.get("decode_p95_ms", r["decode_time_ms"]) * self.decode_scale / 1000

    def observe_decode(self, index, k, elapsed_ms):
        baseline = self.records[index, k].get(
            "decode_p95_ms", self.records[index, k]["decode_time_ms"]
        )
        self.decode_scale = 0.8 * self.decode_scale + 0.2 * elapsed_ms / max(baseline, 1e-9)


class TokenPolicy:
    def __init__(self, config, rd, size_function):
        self.config = config
        self.rd = rd
        self.size_function = lru_cache(maxsize=4096)(size_function)

    def horizon(self, state):
        unplayed = [
            replace(b, prefix=max(b.prefix, b.pending_prefix)) for b in state.blocks if not b.played
        ]
        if not unplayed:
            return []
        duration = 0.0
        result = []
        for b in unplayed[: self.config.scheduler.horizon_blocks]:
            if result and duration + b.duration > self.config.streaming.max_buffer_seconds + 1e-9:
                break
            result.append(b)
            duration += b.duration
        return result

    def decode_cost(self, index, k):
        return self.rd.decode_seconds(index, k) if self.config.scheduler.use_decode_cost else 0.0

    def reserve_base_bytes(self, state, exclude=None):
        target = (
            self.config.streaming.fixed_tokens if self.config.streaming.policy == "fixed" else 8
        )
        return sum(
            self.size_function(b.index, max(b.prefix, b.pending_prefix), target)
            for b in state.blocks
            if not b.played and b.index != exclude and max(b.prefix, b.pending_prefix) < target
        )

    def action(self, state, block, target):
        size = self.size_function(block.index, block.prefix, target)
        if (
            state.remaining_bytes is not None
            and size + self.reserve_base_bytes(state, block.index) > state.remaining_bytes
        ):
            return None
        arrival = state.now + size / max(state.estimated_bandwidth, 1e-9) + state.network_rtt / 2
        finish = max(arrival, state.decoder_available_at) + self.decode_cost(block.index, target)
        return Action(block.index, block.prefix, target, size, finish <= block.deadline)

    def base_action(self, state):
        for b in self.horizon(state):
            if b.prefix < 8:
                return self.action(state, b, 8)
        return None

    def upgrades(self, state):
        content = self.config.scheduler.use_content_quality
        for b in self.horizon(state):
            for k in BUDGETS:
                if k <= b.prefix:
                    continue
                gain = self.rd.distortion(b.index, b.prefix, content) - self.rd.distortion(
                    b.index, k, content
                )
                if gain <= 0:
                    continue
                action = self.action(state, b, k)
                if action is not None and action.deadline_feasible:
                    yield b, k, gain, action


class FixedToken(TokenPolicy):
    def choose(self, state):
        for b in self.horizon(state):
            if b.prefix < self.config.streaming.fixed_tokens:
                return self.action(state, b, self.config.streaming.fixed_tokens)
        return None


class GreedyToken(TokenPolicy):
    def choose(self, state):
        base = self.base_action(state)
        if base is not None:
            return base
        candidates = list(self.upgrades(state))
        if not candidates:
            return None
        return max(candidates, key=lambda x: (x[2] / x[3].size_bytes, -x[0].index, -x[1]))[3]


class DeadlineAwareMPC(TokenPolicy):
    def _score(self, blocks, prefixes, finish_times, bytes_used, state):
        cfg = self.config.scheduler
        distortion = [
            self.rd.distortion(b.index, k, cfg.use_content_quality)
            for b, k in zip(blocks, prefixes)
        ]
        stall = sum(max(0, t - b.deadline) for t, b in zip(finish_times, blocks))
        previous = (
            [state.previous_distortion] if state.previous_distortion is not None else []
        ) + distortion
        variation = sum(abs(a - b) for a, b in itertools.pairwise(previous))
        return (
            sum(d * b.duration for d, b in zip(distortion, blocks))
            + cfg.stall_penalty * stall
            + cfg.variation_penalty * variation
            + cfg.byte_penalty * bytes_used
        )

    def choose(self, state):
        base = self.base_action(state)
        if base is not None:
            return base
        blocks = self.horizon(state)
        if not blocks:
            return None
        # Beam nodes: prefixes, network-free time, decoder-free time, bytes, actions,
        # completion times. Every edge uses a real serialized packet size.
        initial = tuple(b.prefix for b in blocks)
        initial_times = tuple(state.now for _ in blocks)
        beam = [(initial, state.now, state.decoder_available_at, 0, (), initial_times)]
        best_score = self._score(blocks, initial, initial_times, 0, state)
        best = None
        for _ in range(min(3, len(blocks))):
            expanded = []
            for prefixes, net, decode, used, actions, times in beam:
                for j, b in enumerate(blocks):
                    for k in BUDGETS:
                        if k <= prefixes[j]:
                            continue
                        gain = self.rd.distortion(
                            b.index, prefixes[j], self.config.scheduler.use_content_quality
                        ) - self.rd.distortion(
                            b.index, k, self.config.scheduler.use_content_quality
                        )
                        if gain <= 0:
                            continue
                        size = self.size_function(b.index, prefixes[j], k)
                        if (
                            state.remaining_bytes is not None
                            and used + size + self.reserve_base_bytes(state) > state.remaining_bytes
                        ):
                            continue
                        end = net + size / max(state.estimated_bandwidth, 1e-9)
                        done = max(end + state.network_rtt / 2, decode) + self.decode_cost(
                            b.index, k
                        )
                        if done > b.deadline:
                            continue
                        pp = list(prefixes)
                        pp[j] = k
                        tt = list(times)
                        tt[j] = done
                        a = Action(b.index, prefixes[j], k, size, True)
                        node = (tuple(pp), end, done, used + size, actions + (a,), tuple(tt))
                        score = self._score(blocks, node[0], node[5], node[3], state)
                        expanded.append((score, node))
                        if score < best_score:
                            best_score = score
                            best = node[4][0]
            if not expanded:
                break
            expanded.sort(key=lambda item: (item[0], item[1][3], item[1][0]))
            beam = [node for _, node in expanded[: self.config.scheduler.beam_width]]
        return best


def make_policy(config, rd, size_function):
    classes = {"fixed": FixedToken, "greedy": GreedyToken, "deadline_mpc": DeadlineAwareMPC}
    return classes[config.streaming.policy](config, rd, size_function)
