"""Independent sender/receiver processes, shared scheduling and playback core.

The trace option is an application pacer on a real TCP socket. It does not claim
kernel-level bandwidth control. Clocks use the same host's monotonic epoch.
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
    if (
        meta["native_tokens_per_block"] != backend.native_tokens_per_block
        or meta["stream_tokens_per_block"] != 256
    ):
        raise ValueError("Sender/receiver group layout mismatch")
    count = meta["frame_count"]
    pts = meta["pts_us"]
    if (
        not isinstance(count, int)
        or count < 1
        or len(pts) != count
        or any(a >= b for a, b in itertools.pairwise(pts))
    ):
        raise ValueError("Invalid frame count or PTS manifest")
    if manifest["required_prefix"] not in BUDGETS or manifest["initial_buffer_seconds"] < 0:
        raise ValueError("Invalid receiver playback settings")
    return VideoAsset(Path(), meta, None)


def _feedback(session, now, epoch, acks, last_decode, counter):
    state = session.scheduler_state(now, 1, (), 0)
    return {
        "epoch": epoch,
        "now": now,
        "blocks": [asdict(b) for b in state.blocks],
        "buffer_seconds": state.playback_buffer,
        "stall_seconds": session.stall_total,
        "acks": acks,
        "last_decode": last_decode,
        "received_video_bytes": counter.video_received,
        "played_blocks": session.next_index,
        "active_decode": (
            {
                "block": session.active_job.index,
                "prefix": session.active_job.prefix,
                "started_at": session.active_job.started_at,
            }
            if session.active_job
            else None
        ),
    }


def run_receiver(config, host, port, output_dir, ready_file=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    backend = config.backend()
    for _ in range(config.evaluation.warmup):
        backend.decode_tokens(np.zeros((4, 256), np.uint16), np.ones((4, 256), bool))
    counter = ByteCounter()
    reader = FrameReader()
    manifest = None
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
            while manifest is None:
                for kind, payload, _ in _receive(conn, reader, counter):
                    if kind != b"H":
                        raise ValueError("Expected stream manifest")
                    manifest = json.loads(payload)
            asset = _validate_manifest(manifest, backend)
            session = PlaybackSession(
                asset, manifest["initial_buffer_seconds"], manifest["required_prefix"]
            )
            epoch = time.monotonic()
            future = None
            job = None
            acks = []
            last_decode = None
            decode_seq = 0
            packet_log = []
            last_feedback = -1
            last_played = -1
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="adaptok-decoder"
            ) as executor:
                while session.next_index < asset.block_count:
                    now = time.monotonic() - epoch
                    if future is not None and future.done():
                        frames, elapsed, peak = future.result()
                        session.complete_decode(job, now, elapsed, frames, peak)
                        decode_seq += 1
                        last_decode = {
                            "sequence": decode_seq,
                            "block": job.index,
                            "prefix": job.prefix,
                            "elapsed_ms": elapsed,
                        }
                        future = None
                    session.advance(now)
                    if future is None:
                        job = session.begin_decode(now)
                        if job is not None:
                            future = executor.submit(_decode, backend, job)
                    changed = bool(acks) or session.next_index != last_played
                    if changed or now - last_feedback >= 0.04:
                        _send(
                            conn,
                            json_message(
                                b"F", _feedback(session, now, epoch, acks, last_decode, counter)
                            ),
                            counter,
                        )
                        acks = []
                        last_feedback = now
                        last_played = session.next_index
                    wake = session.next_wakeup(now)
                    timeout = min(0.005, max(0, wake - now))
                    readable, _, _ = select.select([conn], [], [], timeout)
                    if readable:
                        for kind, payload, size in _receive(conn, reader, counter):
                            if kind == b"P":
                                packet = TokenPacket.deserialize(payload)
                                if packet.window_id * 4 + packet.block_id >= asset.block_count:
                                    raise ValueError("Packet targets a nonexistent video block")
                                if packet.frame_pts != asset.pts(
                                    packet.window_id * 4 + packet.block_id
                                ):
                                    raise ValueError("Packet PTS disagrees with manifest")
                                now = time.monotonic() - epoch
                                result = session.append(packet, now)
                                counter.video_received += size
                                index = packet.window_id * 4 + packet.block_id
                                acks.append(
                                    {
                                        "block": index,
                                        "end": packet.token_start + packet.token_count,
                                        "size_bytes": size,
                                        "status": result.status,
                                    }
                                )
                                packet_log.append(
                                    {
                                        "block": index,
                                        "start": packet.token_start,
                                        "end": packet.token_start + packet.token_count,
                                        "arrival": now,
                                        "size_bytes": size,
                                        "status": result.status,
                                    }
                                )
                            elif kind != b"E":
                                raise ValueError("Unexpected receiver message")
                if future is not None:
                    frames, elapsed, peak = future.result()
                    session.complete_decode(job, time.monotonic() - epoch, elapsed, frames, peak)
            # Explicit drain handshake accounts for every already-written TCP frame.
            _send(conn, json_message(b"C", {}), counter)
            drained = False
            while not drained:
                for kind, payload, size in _receive(conn, reader, counter):
                    if kind == b"E":
                        drained = True
                    elif kind == b"P":
                        packet = TokenPacket.deserialize(payload)
                        result = session.append(packet, time.monotonic() - epoch)
                        counter.video_received += size
                        packet_log.append(
                            {
                                "block": packet.window_id * 4 + packet.block_id,
                                "start": packet.token_start,
                                "end": packet.token_start + packet.token_count,
                                "arrival": time.monotonic() - epoch,
                                "size_bytes": size,
                                "status": result.status,
                            }
                        )
                    else:
                        raise ValueError("Unexpected drain message")
            summary = session.report()
            summary.update(
                transport="persistent TCP",
                network_byte_scope="application frames, includes length/type and control messages; excludes TCP/IP headers",
                online_wall_seconds=time.monotonic() - epoch,
                preencoding_in_online_latency=False,
            )
            # D contains final downstream count; local log includes D in upstream count.
            _send(
                conn,
                json_message(
                    b"D",
                    {
                        "summary": summary,
                        "received_application_bytes": counter.received,
                        "received_video_bytes": counter.video_received,
                    },
                ),
                counter,
            )
            summary.update(application_bytes=asdict(counter))
            pixels = np.concatenate(session.outputs)[: asset.metadata["frame_count"]]
            write_video(
                out / "reconstruction.nut", pixels, np.array(asset.metadata["pts_us"]), asset.fps
            )
            result = {
                "summary": summary,
                "playback": session.played,
                "events": session.events,
                "packets": packet_log,
                "metadata": asset.metadata,
                "config": config.to_dict(),
            }
            _dump(out, "receiver.json", result)
            return summary


def _state_from_feedback(feedback, now, estimator, config, pending, remaining, decode_available):
    blocks = tuple(BlockView(**b) for b in feedback["blocks"])
    blocks = tuple(replace(b, pending_prefix=pending.get(b.index, 0)) for b in blocks)
    next_deadline = blocks[0].deadline if blocks else now
    return SchedulerState(
        now,
        estimator.estimate,
        tuple(estimator.recent),
        config.network.rtt_ms / 1000,
        feedback["buffer_seconds"],
        next_deadline,
        blocks,
        decode_available,
        remaining,
    )


def run_sender(config, asset_path, rd_path, host, port, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    asset = VideoAsset.load(asset_path)
    rd = RDTable.load(rd_path)
    estimator = ThroughputEstimator(
        config.network.initial_estimate_mbps * 125000, config.network.ewma_alpha
    )
    policy = make_policy(config, rd, lambda i, s, e: asset.packet(i, s, e).wire_bytes)
    pacer = None
    if config.network.trace_path:
        trace = BandwidthTrace.from_swift(config.network.trace_path, config.network.swift_path)
        pacer = TraceLink(trace, config.network.rtt_ms, config.network.jitter_ms, config.seed)
    counter = ByteCounter()
    reader = FrameReader()
    feedback = None
    epoch = None
    pending = {}
    transfers = {}
    waiting = None
    packets = []
    scheduler_ms = []
    last_decode_seq = 0
    done = None
    closing = False
    started = time.monotonic()
    with socket.create_connection((host, port), timeout=60) as conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _send(conn, json_message(b"H", stream_manifest(asset, config)), counter)
        while done is None:
            now = time.monotonic() - epoch if epoch is not None else 0
            if waiting is not None and now >= waiting["send_at"]:
                data = frame_message(b"P", waiting["packet"].serialize())
                _send(conn, data, counter, video=True)
                index = waiting["index"]
                end = waiting["end"]
                transfers[index, end] = (
                    len(data),
                    max(time.monotonic() - epoch - waiting["requested"], 1e-9),
                )
                packets.append(
                    {
                        "block": index,
                        "start": waiting["packet"].token_start,
                        "end": end,
                        "size_bytes": len(data),
                        "request_time": waiting["requested"],
                        "send_time": time.monotonic() - epoch,
                        "payload_bytes": len(waiting["packet"].payload),
                    }
                )
                waiting = None
            if feedback is not None and waiting is None and not closing:
                now = time.monotonic() - epoch
                remaining = (
                    None
                    if config.scheduler.max_video_bytes is None
                    else config.scheduler.max_video_bytes - counter.video_sent
                )
                active = feedback.get("active_decode")
                decode_available = (
                    max(
                        now,
                        active["started_at"] + rd.decode_seconds(active["block"], active["prefix"]),
                    )
                    if active
                    else now
                )
                state = _state_from_feedback(
                    feedback, now, estimator, config, pending, remaining, decode_available
                )
                t = time.perf_counter()
                action = policy.choose(state)
                scheduler_ms.append((time.perf_counter() - t) * 1000)
                if action is not None:
                    block = next(b for b in state.blocks if b.index == action.block_index)
                    deadline = (
                        NO_EXPIRY if action.start == 0 else max(0, round(block.deadline * 1e6))
                    )
                    packet = asset.packet(
                        action.block_index, action.start, action.target_budget, deadline
                    )
                    send_at = pacer.send(packet.wire_bytes, now).delivery if pacer else now
                    waiting = {
                        "packet": packet,
                        "index": action.block_index,
                        "end": action.target_budget,
                        "send_at": send_at,
                        "requested": now,
                    }
                    pending[action.block_index] = action.target_budget
            timeout = 0.005
            if waiting is not None:
                timeout = min(timeout, max(0, waiting["send_at"] - (time.monotonic() - epoch)))
            readable, _, _ = select.select([conn], [], [], timeout)
            if readable:
                for kind, payload, _ in _receive(conn, reader, counter):
                    if kind == b"F":
                        feedback = json.loads(payload)
                        if epoch is None:
                            epoch = feedback["epoch"]
                        for ack in feedback["acks"]:
                            index, end = ack["block"], ack["end"]
                            transfer = transfers.pop((index, end), None)
                            if transfer is not None:
                                estimator.observe(*transfer)
                            if pending.get(index, 0) <= end:
                                pending.pop(index, None)
                        latest = feedback["last_decode"]
                        if latest and latest["sequence"] > last_decode_seq:
                            rd.observe_decode(
                                latest["block"], latest["prefix"], latest["elapsed_ms"]
                            )
                            last_decode_seq = latest["sequence"]
                    elif kind == b"C":
                        closing = True
                        waiting = None
                        _send(conn, json_message(b"E", {}), counter)
                    elif kind == b"D":
                        done = json.loads(payload)
                    else:
                        raise ValueError("Unexpected sender feedback type")
    if (
        counter.video_sent != done["received_video_bytes"]
        or counter.sent != done["received_application_bytes"]
    ):
        raise RuntimeError("Sender/receiver byte counters disagree")
    summary = {
        "sent_video_bytes": counter.video_sent,
        "received_video_bytes": done["received_video_bytes"],
        "application_bytes": asdict(counter),
        "packet_count": len(packets),
        "measured_throughput_bps": counter.video_sent * 8 / max(time.monotonic() - started, 1e-9),
        "mean_packet_transfer_seconds": float(
            np.mean([p["send_time"] - p["request_time"] for p in packets])
        )
        if packets
        else 0,
        "scheduler_mean_ms": float(np.mean(scheduler_ms)) if scheduler_ms else 0,
        "scheduler_max_ms": max(scheduler_ms, default=0),
        "trace_application_pacing": pacer is not None,
        "receiver_summary": done["summary"],
    }
    _dump(out, "sender.json", {"summary": summary, "packets": packets, "config": config.to_dict()})
    return summary
