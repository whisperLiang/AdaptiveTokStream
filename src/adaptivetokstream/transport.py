"""A single persistent TCP framing format and exact application-byte counters."""

import json
import struct
from dataclasses import dataclass

MAX_MESSAGE = 16 * 1024 * 1024


def frame_message(kind, payload):
    if len(kind) != 1 or len(payload) + 1 > MAX_MESSAGE:
        raise ValueError("Invalid message type or length")
    return struct.pack(">I", len(payload) + 1) + kind + payload


def json_message(kind, data):
    return frame_message(kind, json.dumps(data, separators=(",", ":"), allow_nan=False).encode())


class FrameReader:
    def __init__(self):
        self.buffer = bytearray()

    @property
    def pending_bytes(self):
        return len(self.buffer)

    def feed(self, data):
        self.buffer.extend(data)
        result = []
        while len(self.buffer) >= 4:
            length = struct.unpack_from(">I", self.buffer)[0]
            if not 1 <= length <= MAX_MESSAGE:
                raise ValueError("Invalid TCP message length")
            if len(self.buffer) < length + 4:
                break
            result.append((bytes(self.buffer[4:5]), bytes(self.buffer[5 : length + 4]), length + 4))
            del self.buffer[: length + 4]
        return result

    def finish(self):
        if self.buffer:
            raise ValueError("Truncated TCP message at EOF")


@dataclass
class ByteCounter:
    sent: int = 0
    received: int = 0
    video_sent: int = 0
    video_received: int = 0


def stream_manifest(asset, config):
    keys = (
        "frame_count",
        "fps",
        "model_id",
        "pts_us",
        "native_tokens_per_block",
        "stream_tokens_per_block",
    )
    return {
        "metadata": {key: asset.metadata[key] for key in keys},
        "initial_buffer_seconds": config.streaming.initial_buffer_seconds,
        "required_prefix": config.streaming.fixed_tokens
        if config.streaming.policy == "fixed"
        else 8,
    }
