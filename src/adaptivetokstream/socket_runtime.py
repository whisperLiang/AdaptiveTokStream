"""Independent processes over persistent TCP; shared scheduling/playback core.

Trace pacing controls application dispatch on a real socket, not the kernel link.
Sender and receiver use the same host's monotonic clock. Feedback contains deltas.
"""

import itertools
import json
import select
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .asset import VideoAsset
from .codec import BUDGETS
from .media import to_uint8, write_video
from .network import BandwidthTrace, ThroughputEstimator, TraceLink
from .playback import PlaybackSession
from .protocol import NO_EXPIRY, TokenPacket
from .scheduler import BlockView, RDTable, SchedulerState, make_policy
from .transport import ByteCounter, FrameReader, frame_message, json_message, stream_manifest


def _receive(sock, reader, counter):
    data = sock.recv(65536)
    if not data:
        reader.finish()
        raise ConnectionError("Peer disconnected before stream completion")
    counter.received += len(data)
    return reader.feed(data)


def _send(sock, data, counter, video=False):
    sock.sendall(data)
    counter.sent += len(data)
    if video:
        counter.video_sent += len(data)


def _dump(out, name, value):
    (out / name).write_text(json.dumps(value, indent=2, allow_nan=False))


def _decode(backend, job):
    started = time.perf_counter()
    result = backend.decode_tokens(job.ids, job.mask)
    g = job.index % 4
    frames = to_uint8(result.frames[:, :, g * 4 : g * 4 + 4])
    return frames, (time.perf_counter() - started) * 1000, result.peak_memory_bytes


def _validate_manifest(manifest, backend):
    meta = manifest["metadata"]
    if meta["model_id"] != backend.model_id.hex():
        raise ValueError("Sender/receiver checkpoint mismatch")
    layout = (meta["native_tokens_per_block"], meta["stream_tokens_per_block"])
    if layout != (backend.native_tokens_per_block, 256):
        raise ValueError("Sender/receiver group layout mismatch")
    _validate_timeline(meta)
    if manifest["required_prefix"] not in BUDGETS or manifest["initial_buffer_seconds"] < 0:
        raise ValueError("Invalid receiver playback settings")
    return VideoAsset(Path(), meta, None)


def _validate_timeline(meta):
    count = meta["frame_count"]
    pts = meta["pts_us"]
    if not isinstance(count, int) or count < 1 or len(pts) != count:
        raise ValueError("Invalid frame count or PTS manifest")
    if any(a >= b for a, b in itertools.pairwise(pts)):
        raise ValueError("PTS must increase")


def merge_feedback(previous, update):
    previous = previous or {"blocks": []}
    blocks = {b["index"]: b for b in previous["blocks"] if b["index"] >= update["played_blocks"]}
    blocks.update({b["index"]: b for b in update["blocks"]})
    return {**update, "blocks": [blocks[i] for i in sorted(blocks)]}


class ReceiverRuntime:
    def __init__(self, config, backend, output_dir):
        self.config = config
        self.backend = backend
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.counter = ByteCounter()
        self.reader = FrameReader()
        self.future = None
        self.job = None
        self.acks = []
        self.last_decode = None
        self.decode_seq = 0
        self.packet_log = []
        self.last_feedback_signature = None
        self.previous_blocks = {}

    def manifest(self, conn):
        while True:
            for kind, payload, _ in _receive(conn, self.reader, self.counter):
                if kind != b"H":
                    raise ValueError("Expected stream manifest")
                return json.loads(payload)

    def accept_packet(self, payload, size):
        packet = TokenPacket.deserialize(payload)
        index = packet.window_id * 4 + packet.block_id
        if index >= self.asset.block_count:
            raise ValueError("Packet targets a nonexistent block")
        if packet.frame_pts != self.asset.pts(index):
            raise ValueError("Packet PTS disagrees with manifest")
        now = time.monotonic() - self.epoch
        result = self.session.append(packet, now)
        self.counter.video_received += size
        end = packet.token_start + packet.token_count
        self.acks.append({"block": index, "end": end, "size_bytes": size, "status": result.status})
        self.packet_log.append(
            {
                "block": index,
                "start": packet.token_start,
                "end": end,
                "arrival": now,
                "size_bytes": size,
                "status": result.status,
            }
        )

    def complete_decode(self, now):
        if self.future is None or not self.future.done():
            return
        frames, elapsed, peak = self.future.result()
        self.session.complete_decode(self.job, now, elapsed, frames, peak)
        self.decode_seq += 1
        self.last_decode = {
            "sequence": self.decode_seq,
            "block": self.job.index,
            "prefix": self.job.prefix,
            "elapsed_ms": elapsed,
        }
        self.future = None

    def feedback(self, conn, now):
        s = self.session
        signature = (s.next_index, s.stall_total, self.decode_seq, s.stall_start)
        if not self.acks and signature == self.last_feedback_signature:
            return
        state = s.scheduler_state(now, 1, (), 0)
        blocks = {b.index: asdict(b) for b in state.blocks}
        delta = [b for i, b in blocks.items() if self.previous_blocks.get(i) != b]
        self.previous_blocks = blocks
        active = s.active_job
        active_data = (
            {"block": active.index, "prefix": active.prefix, "started_at": active.started_at}
            if active
            else None
        )
        payload = {
            "epoch": self.epoch,
            "now": now,
            "blocks": delta,
            "buffer_seconds": state.playback_buffer,
            "stall_seconds": s.stall_total,
            "acks": self.acks,
            "last_decode": self.last_decode,
            "received_video_bytes": self.counter.video_received,
            "played_blocks": s.next_index,
            "active_decode": active_data,
        }
        _send(conn, json_message(b"F", payload), self.counter)
        self.acks = []
        self.last_feedback_signature = signature

    def process_messages(self, conn):
        for kind, payload, size in _receive(conn, self.reader, self.counter):
            if kind == b"P":
                self.accept_packet(payload, size)
            elif kind != b"E":
                raise ValueError("Unexpected receiver message")

    def loop(self, conn, executor):
        while self.session.next_index < self.asset.block_count:
            now = time.monotonic() - self.epoch
            self.complete_decode(now)
            self.session.advance(now)
            if self.future is None:
                self.job = self.session.begin_decode(now)
                if self.job is not None:
                    self.future = executor.submit(_decode, self.backend, self.job)
            self.feedback(conn, now)
            timeout = min(0.005, max(0, self.session.next_wakeup(now) - now))
            readable, _, _ = select.select([conn], [], [], timeout)
            if readable:
                self.process_messages(conn)
        if self.future is not None:
            self.future.result()
            self.complete_decode(time.monotonic() - self.epoch)

    def drain(self, conn):
        _send(conn, json_message(b"C", {}), self.counter)
        while True:
            for kind, payload, size in _receive(conn, self.reader, self.counter):
                if kind == b"E":
                    return
                if kind != b"P":
                    raise ValueError("Unexpected drain message")
                self.accept_packet(payload, size)

    def finish(self, conn):
        summary = self.session.report()
        summary.update(
            transport="persistent TCP",
            network_byte_scope="application frames including length/type and control; excludes TCP/IP headers",
            online_wall_seconds=time.monotonic() - self.epoch,
            preencoding_in_online_latency=False,
        )
        _send(
            conn,
            json_message(
                b"D",
                {
                    "summary": summary,
                    "received_application_bytes": self.counter.received,
                    "received_video_bytes": self.counter.video_received,
                },
            ),
            self.counter,
        )
        summary["application_bytes"] = asdict(self.counter)
        pixels = np.concatenate(self.session.outputs)[: self.asset.metadata["frame_count"]]
        write_video(
            self.out / "reconstruction.nut",
            pixels,
            np.array(self.asset.metadata["pts_us"]),
            self.asset.fps,
        )
        _dump(
            self.out,
            "receiver.json",
            {
                "summary": summary,
                "playback": self.session.played,
                "events": self.session.events,
                "packets": self.packet_log,
                "metadata": self.asset.metadata,
                "config": self.config.to_dict(),
            },
        )
        return summary

    def run(self, conn):
        manifest = self.manifest(conn)
        self.asset = _validate_manifest(manifest, self.backend)
        self.session = PlaybackSession(
            self.asset, manifest["initial_buffer_seconds"], manifest["required_prefix"]
        )
        self.epoch = time.monotonic()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="adaptok-decoder") as executor:
            self.loop(conn, executor)
        self.drain(conn)
        return self.finish(conn)


def run_receiver(config, host, port, output_dir, ready_file=None):
    backend = config.backend()
    for _ in range(config.evaluation.warmup):
        backend.decode_tokens(np.zeros((4, 256), np.uint16), np.ones((4, 256), bool))
    runtime = ReceiverRuntime(config, backend, output_dir)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        if ready_file:
            Path(ready_file).write_text(str(server.getsockname()[1]))
        conn, _ = server.accept()
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(60)
            return runtime.run(conn)


class SenderRuntime:
    def __init__(self, config, asset, rd, output_dir):
        self.config = config
        self.asset = asset
        self.rd = rd
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.estimator = ThroughputEstimator(
            config.network.initial_estimate_mbps * 125000, config.network.ewma_alpha
        )
        self.policy = make_policy(config, rd, lambda i, s, e: asset.packet(i, s, e).wire_bytes)
        self.pacer = None
        if config.network.trace_path:
            trace = BandwidthTrace.from_swift(config.network.trace_path, config.network.swift_path)
            self.pacer = TraceLink(
                trace, config.network.rtt_ms, config.network.jitter_ms, config.seed
            )
        self.counter = ByteCounter()
        self.reader = FrameReader()
        self.feedback = None
        self.epoch = None
        self.pending = {}
        self.transfers = {}
        self.outbound = []
        self.network_free_at = 0.0
        self.packets = []
        self.scheduler_ms = []
        self.last_decode_seq = 0
        self.done = None
        self.closing = False
        self.started = time.monotonic()

    def now(self):
        return time.monotonic() - self.epoch if self.epoch is not None else 0

    def _write_packet(self, conn, waiting):
        data = frame_message(b"P", waiting["packet"].serialize())
        _send(conn, data, self.counter, video=True)
        index = waiting["index"]
        end = waiting["end"]
        now = self.now()
        self.transfers[index, end] = (len(data), max(now - waiting["requested"], 1e-9))
        self.packets.append(
            {
                "block": index,
                "start": waiting["packet"].token_start,
                "end": end,
                "size_bytes": len(data),
                "request_time": waiting["requested"],
                "send_time": now,
                "payload_bytes": len(waiting["packet"].payload),
            }
        )

    def write_due_packet(self, conn):
        while self.outbound and self.now() >= self.outbound[0]["send_at"]:
            self._write_packet(conn, self.outbound.pop(0))

    def state(self, now):
        cap = self.config.scheduler.max_video_bytes
        remaining = (
            None
            if cap is None
            else cap - self.counter.video_sent - sum(p["packet"].wire_bytes for p in self.outbound)
        )
        blocks = tuple(BlockView(**b) for b in self.feedback["blocks"])
        blocks = tuple(replace(b, pending_prefix=self.pending.get(b.index, 0)) for b in blocks)
        active = self.feedback.get("active_decode")
        available = (
            max(
                now,
                active["started_at"] + self.rd.decode_seconds(active["block"], active["prefix"]),
            )
            if active
            else now
        )
        return SchedulerState(
            now,
            self.estimator.estimate,
            tuple(self.estimator.recent),
            self.config.network.rtt_ms / 1000,
            self.feedback["buffer_seconds"],
            blocks[0].deadline if blocks else now,
            blocks,
            available,
            remaining,
        )

    def choose_next_packet(self):
        if self.feedback is None or self.now() < self.network_free_at or self.closing:
            return
        now = self.now()
        state = self.state(now)
        start = time.perf_counter()
        action = self.policy.choose(state)
        self.scheduler_ms.append((time.perf_counter() - start) * 1000)
        if action is None:
            return
        self.enqueue_packet(action, state, now)

    def enqueue_packet(self, action, state, now):
        block = next(b for b in state.blocks if b.index == action.block_index)
        deadline = NO_EXPIRY if action.start == 0 else max(0, round(block.deadline * 1e6))
        packet = self.asset.packet(action.block_index, action.start, action.target_budget, deadline)
        transfer = self.pacer.send(packet.wire_bytes, now) if self.pacer else None
        at = transfer.delivery if transfer else now
        self.network_free_at = transfer.service_end if transfer else now
        self.outbound.append(
            {
                "packet": packet,
                "index": action.block_index,
                "end": action.target_budget,
                "send_at": at,
                "requested": now,
            }
        )
        self.pending[action.block_index] = action.target_budget

    def update_feedback(self, payload):
        self.feedback = merge_feedback(self.feedback, json.loads(payload))
        if self.epoch is None:
            self.epoch = self.feedback["epoch"]
        for ack in self.feedback["acks"]:
            index, end = ack["block"], ack["end"]
            transfer = self.transfers.pop((index, end), None)
            if transfer is not None:
                self.estimator.observe(*transfer)
            if self.pending.get(index, 0) <= end:
                self.pending.pop(index, None)
        latest = self.feedback["last_decode"]
        if latest and latest["sequence"] > self.last_decode_seq:
            self.rd.observe_decode(latest["block"], latest["prefix"], latest["elapsed_ms"])
            self.last_decode_seq = latest["sequence"]

    def process_messages(self, conn):
        for kind, payload, _ in _receive(conn, self.reader, self.counter):
            if kind == b"F":
                self.update_feedback(payload)
            elif kind == b"C":
                self.closing = True
                self.outbound.clear()
                _send(conn, json_message(b"E", {}), self.counter)
            elif kind == b"D":
                self.done = json.loads(payload)
            else:
                raise ValueError("Unexpected sender feedback type")

    def loop(self, conn):
        _send(conn, json_message(b"H", stream_manifest(self.asset, self.config)), self.counter)
        while self.done is None:
            self.write_due_packet(conn)
            self.choose_next_packet()
            next_write = self.outbound[0]["send_at"] if self.outbound else self.now() + 0.005
            timeout = min(0.005, max(0, next_write - self.now()))
            readable, _, _ = select.select([conn], [], [], timeout)
            if readable:
                self.process_messages(conn)

    def finish(self):
        if (
            self.counter.video_sent != self.done["received_video_bytes"]
            or self.counter.sent != self.done["received_application_bytes"]
        ):
            raise RuntimeError("Sender/receiver byte counters disagree")
        summary = {
            "sent_video_bytes": self.counter.video_sent,
            "received_video_bytes": self.done["received_video_bytes"],
            "application_bytes": asdict(self.counter),
            "packet_count": len(self.packets),
            "measured_throughput_bps": self.counter.video_sent
            * 8
            / max(time.monotonic() - self.started, 1e-9),
            "mean_packet_transfer_seconds": float(
                np.mean([p["send_time"] - p["request_time"] for p in self.packets])
            )
            if self.packets
            else 0,
            "scheduler_mean_ms": float(np.mean(self.scheduler_ms)) if self.scheduler_ms else 0,
            "scheduler_max_ms": max(self.scheduler_ms, default=0),
            "trace_application_pacing": self.pacer is not None,
            "receiver_summary": self.done["summary"],
        }
        _dump(
            self.out,
            "sender.json",
            {"summary": summary, "packets": self.packets, "config": self.config.to_dict()},
        )
        return summary


def run_sender(config, asset_path, rd_path, host, port, output_dir):
    runtime = SenderRuntime(config, VideoAsset.load(asset_path), RDTable.load(rd_path), output_dir)
    with socket.create_connection((host, port), timeout=60) as conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        runtime.loop(conn)
    return runtime.finish()
