"""Byte-exact event simulation; quality is evaluated later by the real decoder."""

import heapq
import itertools
import json
import math
import time
from dataclasses import replace
from pathlib import Path

from .network import BandwidthTrace, ThroughputEstimator, TraceLink
from .playback import PlaybackSession
from .protocol import NO_EXPIRY, TokenPacket
from .scheduler import make_policy


def configured_trace(config):
    if config.network.trace_path:
        return BandwidthTrace.from_swift(config.network.trace_path, config.network.swift_path)
    return BandwidthTrace.constant(config.network.bandwidth_mbps)


class TokenSimulator:
    def __init__(self, asset, rd, config, trace, decode_scale):
        self.asset = asset
        self.rd = rd
        self.config = config
        self.decode_scale = decode_scale
        self.link = TraceLink(trace, config.network.rtt_ms, config.network.jitter_ms, config.seed)
        self.estimator = ThroughputEstimator(
            config.network.initial_estimate_mbps * 125000, config.network.ewma_alpha
        )
        required = config.streaming.fixed_tokens if config.streaming.policy == "fixed" else 8
        self.session = PlaybackSession(asset, config.streaming.initial_buffer_seconds, required)
        self.policy = make_policy(config, rd, lambda i, s, e: asset.packet(i, s, e).wire_bytes)
        self.queue = []
        self.sequence = itertools.count()
        self.now = 0.0
        self.network_busy = True
        self.pending = {}
        self.sent = 0
        self.payload_bytes = 0
        self.packets = []
        self.scheduler_ms = []
        self.decode_until = 0.0
        self.push(config.network.rtt_ms / 1000, "network_free")
        for w in range(len(asset.tokens)):
            self.session.emit("ENCODED_WINDOW_READY", 0, window_id=w)

    def push(self, at, kind, data=None):
        heapq.heappush(self.queue, (at, next(self.sequence), kind, data))

    def start_decode(self):
        job = self.session.begin_decode(self.now)
        if job is None:
            return
        cost = self.rd.records[job.index, job.prefix]["decode_time_ms"] / 1000 * self.decode_scale
        self.decode_until = self.now + cost
        self.push(self.decode_until, "decode", (job, cost))

    def choose_action(self):
        cap = self.config.scheduler.max_video_bytes
        remaining = None if cap is None else cap - self.sent
        state = self.session.scheduler_state(
            self.now,
            self.estimator.estimate,
            self.estimator.recent,
            self.config.network.rtt_ms / 1000,
            self.decode_until,
            remaining,
        )
        state = replace(
            state,
            blocks=tuple(
                replace(b, pending_prefix=self.pending.get(b.index, 0)) for b in state.blocks
            ),
        )
        start = time.perf_counter()
        action = self.policy.choose(state)
        self.scheduler_ms.append((time.perf_counter() - start) * 1000)
        return action

    def request_packet(self):
        if self.network_busy or self.session.next_index >= self.asset.block_count:
            return
        action = self.choose_action()
        if action is None:
            return
        deadline = (
            NO_EXPIRY
            if action.start == 0
            else max(0, round(self.session.deadline(action.block_index) * 1e6))
        )
        packet = self.asset.packet(action.block_index, action.start, action.target_budget, deadline)
        wire = packet.serialize()
        if packet.wire_bytes != action.size_bytes:
            raise AssertionError("Scheduler byte mismatch")
        transfer = self.link.send(packet.wire_bytes, self.now)
        self.session.emit(
            "TOKEN_PACKET_REQUESTED",
            self.now,
            block=action.block_index,
            start=action.start,
            end=action.target_budget,
            deadline_feasible=action.deadline_feasible,
        )
        self.session.emit(
            "PACKET_SENT", self.now, block=action.block_index, size_bytes=packet.wire_bytes
        )
        self.sent += packet.wire_bytes
        self.payload_bytes += len(packet.payload)
        self.pending[action.block_index] = action.target_budget
        self.packets.append(
            {
                "block": action.block_index,
                "start": action.start,
                "end": action.target_budget,
                "size_bytes": packet.wire_bytes,
                "payload_bytes": len(packet.payload),
                "request_time": self.now,
                "service_end": transfer.service_end,
                "arrival": transfer.delivery,
            }
        )
        self.push(transfer.service_end, "network_free")
        self.push(transfer.delivery, "receive", (wire, transfer))
        self.push(transfer.delivery + self.config.network.rtt_ms / 2000, "feedback", transfer)
        self.network_busy = True

    def receive_packet(self, data):
        wire, _transfer = data
        packet = TokenPacket.deserialize(wire)
        self.session.append(packet, self.now)
        index = packet.window_id * 4 + packet.block_id
        if self.pending.get(index, 0) <= packet.token_start + packet.token_count:
            self.pending.pop(index, None)
        self.session.emit("DOWNLOAD_COMPLETED", self.now, block=index, size_bytes=packet.wire_bytes)

    def dispatch(self, kind, data):
        if kind == "network_free":
            self.network_busy = False
        elif kind == "receive":
            self.receive_packet(data)
        elif kind == "feedback":
            self.estimator.observe(data.size, data.service_end - data.service_start)
        elif kind == "decode":
            job, cost = data
            self.session.complete_decode(job, self.now, cost * 1000)
            self.rd.observe_decode(job.index, job.prefix, cost * 1000)
        else:
            raise ValueError(f"Unknown simulation event: {kind}")

    def step(self):
        self.session.advance(self.now)
        self.start_decode()
        self.request_packet()
        wake = self.session.next_wakeup(self.now)
        next_event = self.queue[0][0] if self.queue else math.inf
        if wake < next_event:
            self.now = wake
            return
        if not self.queue:
            if self.session.next_index >= self.asset.block_count:
                return
            raise RuntimeError(
                "No feasible progress: byte cap exhausted or required token prefix unavailable"
            )
        self.now = next_event
        while self.queue and self.queue[0][0] <= self.now + 1e-12:
            _, _, kind, data = heapq.heappop(self.queue)
            self.dispatch(kind, data)

    def run(self):
        while self.session.next_index < self.asset.block_count or self.queue:
            self.step()
        summary = self.session.report()
        summary.update(
            sent_video_bytes=self.sent,
            token_payload_bytes=self.payload_bytes,
            packet_header_bytes=sum(p["size_bytes"] - p["payload_bytes"] - 5 for p in self.packets),
            framing_bytes=5 * len(self.packets),
            packet_count=len(self.packets),
            scheduler_mean_ms=sum(self.scheduler_ms) / max(1, len(self.scheduler_ms)),
            scheduler_max_ms=max(self.scheduler_ms, default=0),
            timing_source="device-profiled full-window inference; discrete event execution",
            peak_memory_bytes=None,
            policy=self.config.streaming.policy,
            seed=self.config.seed,
        )
        return {
            "summary": summary,
            "playback": self.session.played,
            "events": self.session.events,
            "packets": self.packets,
            "config": self.config.to_dict(),
        }


def simulate(asset, rd, config, trace=None, decode_scale=1.0, output_dir=None):
    if decode_scale <= 0:
        raise ValueError("Decoder scaling must be positive")
    result = TokenSimulator(
        asset, rd, config, trace or configured_trace(config), decode_scale
    ).run()
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "simulation.json").write_text(json.dumps(result, indent=2))
    return result
