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


def simulate(asset, rd, config, trace=None, decode_scale=1.0, output_dir=None):
    trace = trace or (
        BandwidthTrace.from_swift(config.network.trace_path, config.network.swift_path)
        if config.network.trace_path
        else BandwidthTrace.constant(config.network.bandwidth_mbps)
    )
    link = TraceLink(trace, config.network.rtt_ms, config.network.jitter_ms, config.seed)
    estimator = ThroughputEstimator(
        config.network.initial_estimate_mbps * 125000, config.network.ewma_alpha
    )
    required = config.streaming.fixed_tokens if config.streaming.policy == "fixed" else 8
    session = PlaybackSession(asset, config.streaming.initial_buffer_seconds, required)
    policy = make_policy(config, rd, lambda i, s, e: asset.packet(i, s, e).wire_bytes)
    queue = []
    sequence = itertools.count()
    now = 0.0
    network_busy = True

    def push(t, kind, data=None):
        heapq.heappush(queue, (t, next(sequence), kind, data))

    # One persistent connection establishment, then a pipelined stream.
    push(config.network.rtt_ms / 1000, "network_free")
    pending = {}
    sent = 0
    payload_bytes = 0
    packets = []
    scheduler_ms = []
    decode_until = 0.0
    for w in range(len(asset.tokens)):
        session.emit("ENCODED_WINDOW_READY", 0, window_id=w)
    while session.next_index < asset.block_count or queue:
        session.advance(now)
        job = session.begin_decode(now)
        if job is not None:
            record = rd.records[job.index, job.prefix]
            cost = record["decode_time_ms"] / 1000 * decode_scale
            decode_until = now + cost
            push(decode_until, "decode", (job, cost))
        if not network_busy and session.next_index < asset.block_count:
            remaining = (
                None
                if config.scheduler.max_video_bytes is None
                else config.scheduler.max_video_bytes - sent
            )
            state = session.scheduler_state(
                now,
                estimator.estimate,
                estimator.recent,
                config.network.rtt_ms / 1000,
                decode_until,
                remaining,
            )
            state = replace(
                state,
                blocks=tuple(
                    replace(b, pending_prefix=pending.get(b.index, 0)) for b in state.blocks
                ),
            )
            started = time.perf_counter()
            action = policy.choose(state)
            scheduler_ms.append((time.perf_counter() - started) * 1000)
            if action is not None:
                deadline = (
                    NO_EXPIRY
                    if action.start == 0
                    else max(0, round(session.deadline(action.block_index) * 1e6))
                )
                packet = asset.packet(
                    action.block_index, action.start, action.target_budget, deadline
                )
                wire = packet.serialize()
                if packet.wire_bytes != action.size_bytes:
                    raise AssertionError("Scheduler byte mismatch")
                transfer = link.send(packet.wire_bytes, now)
                session.emit(
                    "TOKEN_PACKET_REQUESTED",
                    now,
                    block=action.block_index,
                    start=action.start,
                    end=action.target_budget,
                    deadline_feasible=action.deadline_feasible,
                )
                session.emit(
                    "PACKET_SENT", now, block=action.block_index, size_bytes=packet.wire_bytes
                )
                sent += packet.wire_bytes
                payload_bytes += len(packet.payload)
                pending[action.block_index] = action.target_budget
                packets.append(
                    {
                        "block": action.block_index,
                        "start": action.start,
                        "end": action.target_budget,
                        "size_bytes": packet.wire_bytes,
                        "payload_bytes": len(packet.payload),
                        "request_time": now,
                        "service_end": transfer.service_end,
                        "arrival": transfer.delivery,
                    }
                )
                push(transfer.service_end, "network_free")
                push(transfer.delivery, "receive", (wire, transfer))
                push(transfer.delivery + config.network.rtt_ms / 2000, "feedback", transfer)
                network_busy = True
        wake = session.next_wakeup(now)
        next_event = queue[0][0] if queue else math.inf
        if wake < next_event:
            now = wake
            continue
        if not queue:
            if session.next_index >= asset.block_count:
                break
            raise RuntimeError(
                "No feasible progress: byte cap exhausted or required token prefix unavailable"
            )
        now = next_event
        while queue and queue[0][0] <= now + 1e-12:
            _, _, kind, data = heapq.heappop(queue)
            if kind == "network_free":
                network_busy = False
            elif kind == "receive":
                wire, transfer = data
                packet = TokenPacket.deserialize(wire)
                session.append(packet, now)
                index = packet.window_id * 4 + packet.block_id
                if pending.get(index, 0) <= packet.token_start + packet.token_count:
                    pending.pop(index, None)
                session.emit("DOWNLOAD_COMPLETED", now, block=index, size_bytes=packet.wire_bytes)
            elif kind == "feedback":
                estimator.observe(data.size, data.service_end - data.service_start)
            elif kind == "decode":
                job, cost = data
                session.complete_decode(job, now, cost * 1000)
                rd.observe_decode(job.index, job.prefix, cost * 1000)
    summary = session.report()
    summary.update(
        sent_video_bytes=sent,
        token_payload_bytes=payload_bytes,
        packet_header_bytes=sum(p["size_bytes"] - p["payload_bytes"] - 5 for p in packets),
        framing_bytes=5 * len(packets),
        packet_count=len(packets),
        scheduler_mean_ms=sum(scheduler_ms) / max(1, len(scheduler_ms)),
        scheduler_max_ms=max(scheduler_ms, default=0),
        timing_source="device-profiled full-window inference; discrete event execution",
        policy=config.streaming.policy,
        seed=config.seed,
    )
    result = {
        "summary": summary,
        "playback": session.played,
        "events": session.events,
        "packets": packets,
        "config": config.to_dict(),
    }
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "simulation.json").write_text(json.dumps(result, indent=2))
    return result
