"""Shared playback/decode state machine driven by simulated or monotonic time."""

import math
from dataclasses import dataclass

import numpy as np

from .buffer import TokenBuffer
from .scheduler import BlockView, SchedulerState


@dataclass(frozen=True)
class DecodeJob:
    index: int
    prefix: int
    ids: np.ndarray
    mask: np.ndarray
    signature: tuple
    started_at: float


@dataclass
class ReadyBlock:
    prefix: int
    signature: tuple
    frames: object = None


class PlaybackSession:
    def __init__(self, asset, initial_buffer=1.0, required_prefix=8):
        self.asset = asset
        self.initial_buffer = initial_buffer
        self.required_prefix = required_prefix
        self.buffer = TokenBuffer(1, bytes.fromhex(asset.metadata["model_id"]))
        self.next_index = 0
        self.ready = {}
        self.active_job = None
        self.played = []
        self.events = []
        self.stall_start = None
        self.stall_total = 0.0
        self.stall_count = 0
        self.deadline_misses = set()
        self.decoder_ms = []
        self.outputs = []
        self.peak_memory_bytes = 0

    def emit(self, kind, now, **fields):
        self.events.append({"event": kind, "time": float(now), **fields})

    def deadline(self, index):
        return (
            self.initial_buffer
            + (self.asset.pts(index) - self.asset.pts(0)) / 1e6
            + self.stall_total
        )

    def signature(self, index):
        w, g = divmod(index, 4)
        _, mask = self.buffer.snapshot(w, g)
        return tuple(sum(row) for row in mask[: g + 1])

    def append(self, packet, now):
        result = self.buffer.append(packet, round(now * 1e6))
        self.emit(
            "PACKET_RECEIVED",
            now,
            block=packet.window_id * 4 + packet.block_id,
            token_start=packet.token_start,
            token_count=packet.token_count,
            size_bytes=packet.wire_bytes,
            status=result.status,
            prefix=result.prefix,
        )
        return result

    def valid_ready(self, index):
        ready = self.ready.get(index)
        if ready is None:
            return None
        sig = self.signature(index)
        return ready if sig[:-1] == ready.signature[:-1] and sig[-1] >= ready.prefix else None

    def begin_decode(self, now):
        if self.active_job is not None:
            return None
        for index in range(self.next_index, self.asset.block_count):
            w, g = divmod(index, 4)
            if self.buffer.prefix(w, g) < self.required_prefix:
                continue
            ready = self.valid_ready(index)
            sig = self.signature(index)
            if ready is not None and ready.prefix == sig[-1]:
                continue
            ids, mask = self.buffer.snapshot(w, g)
            job = DecodeJob(index, sig[-1], np.array(ids), np.array(mask), sig, now)
            self.active_job = job
            self.emit(
                "DECODE_STARTED",
                now,
                block=index,
                prefix=job.prefix,
                history_budgets=list(sig[:-1]),
                scope="full_16_frame_window",
            )
            return job
        return None

    def complete_decode(self, job, now, elapsed_ms, frames=None, peak_memory_bytes=0):
        if self.active_job is not job:
            raise ValueError("Unexpected decode completion")
        self.active_job = None
        self.decoder_ms.append(elapsed_ms)
        self.peak_memory_bytes = max(self.peak_memory_bytes, peak_memory_bytes)
        self.check_decode_deadline(job.index, now)
        self.emit(
            "DECODE_COMPLETED", now, block=job.index, prefix=job.prefix, elapsed_ms=elapsed_ms
        )
        if job.index < self.next_index:
            return
        sig = self.signature(job.index)
        if sig[:-1] == job.signature[:-1]:
            self.ready[job.index] = ReadyBlock(job.prefix, job.signature, frames)

    def check_decode_deadline(self, index, now):
        if index == self.next_index and now > self.deadline(index) + 1e-9:
            if self.valid_ready(index) is None:
                self.record_underrun(index, self.deadline(index))

    def advance(self, now):
        """Freeze only completed, causally valid reconstructions at their PTS."""
        if self.next_index >= self.asset.block_count:
            return
        index = self.next_index
        due = self.deadline(index)
        if now + 1e-9 < due:
            return
        ready = self.valid_ready(index)
        if ready is None:
            self.record_underrun(index, due)
            return
        stall = 0.0
        if self.stall_start is not None:
            stall = max(0, now - self.stall_start)
            self.stall_total += stall
            self.stall_start = None
        start = max(now, due)
        if not self.played:
            self.emit("PLAYBACK_STARTED", start, block=index)
        w, g = divmod(index, 4)
        self.buffer.finalize(w, g, ready.prefix)
        record = {
            "block": index,
            "window_id": w,
            "block_id": g,
            "frame_pts": self.asset.pts(index),
            "nominal_deadline": self.initial_buffer
            + (self.asset.pts(index) - self.asset.pts(0)) / 1e6,
            "scheduled_deadline": due,
            "playback_time": start,
            "prefix": ready.prefix,
            "history_budgets": list(ready.signature[:-1]),
            "stall_seconds": stall,
            "duration": self.asset.duration(index),
        }
        self.played.append(record)
        self.emit("BLOCK_PLAYED", start, **record)
        if ready.frames is not None:
            self.outputs.append(ready.frames)
        self.ready.pop(index, None)
        self.next_index += 1
        if g == 3 or self.next_index == self.asset.block_count:
            # Final partial window may have unused groups; release it when stream ends.
            self.buffer.clear_window(w, g + 1)

    def record_underrun(self, index, due):
        if self.stall_start is None:
            self.stall_start = due
            self.stall_count += 1
            self.deadline_misses.add(index)
            self.emit("BLOCK_DEADLINE_EXPIRED", due, block=index)
            self.emit("BUFFER_UNDERRUN", due, block=index)

    def next_wakeup(self, now):
        if self.next_index >= self.asset.block_count:
            return math.inf
        if self.stall_start is not None:
            return math.inf
        return max(now, self.deadline(self.next_index))

    def scheduler_state(
        self, now, estimate, recent, rtt, decoder_available=0, remaining_bytes=None
    ):
        blocks = []
        buffer_seconds = 0.0
        for i in range(self.next_index, self.asset.block_count):
            w, g = divmod(i, 4)
            prefix = self.buffer.prefix(w, g)
            ready = self.valid_ready(i)
            blocks.append(
                BlockView(
                    i,
                    prefix,
                    self.deadline(i),
                    self.asset.duration(i),
                    False,
                    ready.prefix if ready else 0,
                )
            )
            if ready:
                buffer_seconds += self.asset.duration(i)
        return SchedulerState(
            now,
            estimate,
            tuple(recent),
            rtt,
            buffer_seconds,
            self.deadline(self.next_index)
            if self.next_index < self.asset.block_count
            else math.inf,
            tuple(blocks),
            decoder_available,
            remaining_bytes,
        )

    def report(self):
        duration = sum(r["duration"] for r in self.played)
        elapsed = (
            (self.played[-1]["playback_time"] + self.played[-1]["duration"]) if self.played else 0
        )
        stats = self.buffer.stats
        return {
            "frame_count": self.asset.metadata["frame_count"],
            "blocks_played": len(self.played),
            "stall_count": self.stall_count,
            "stall_seconds": self.stall_total,
            "deadline_miss_rate": len(self.deadline_misses) / max(1, len(self.played)),
            "received_video_bytes": stats.received_bytes,
            "duplicate_payload_bytes": stats.duplicate_payload_bytes,
            "expired_packets": stats.expired_packets,
            "accepted_tokens": stats.accepted_tokens,
            "video_effective_bitrate_bps": stats.accepted_payload_bytes * 8 / max(duration, 1e-9),
            "transport_bitrate_bps": stats.received_bytes * 8 / max(duration, 1e-9),
            "average_playback_latency_seconds": float(
                np.mean(
                    [
                        r["playback_time"] - (r["frame_pts"] - self.asset.pts(0)) / 1e6
                        for r in self.played
                    ]
                )
            )
            if self.played
            else 0,
            "elapsed_seconds": elapsed,
            "effective_fps": self.asset.metadata["frame_count"] / max(elapsed, 1e-9),
            "decode_mean_ms": float(np.mean(self.decoder_ms)) if self.decoder_ms else 0,
            "decode_p50_ms": float(np.percentile(self.decoder_ms, 50)) if self.decoder_ms else 0,
            "decode_p95_ms": float(np.percentile(self.decoder_ms, 95)) if self.decoder_ms else 0,
            "decoder_calls": len(self.decoder_ms),
            "peak_memory_bytes": self.peak_memory_bytes,
        }
