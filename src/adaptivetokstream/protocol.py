"""Single canonical wire format: network byte order, MSB-first 13-bit IDs.

IDs are concatenated most-significant bit first, padded with zero low bits in
only the final byte. CRC32 covers the entire header (CRC field zeroed) and payload.
Time fields are unsigned microseconds relative to the stream's monotonic epoch.
model_id is the first 16 bytes of the checkpoint SHA256. No tensor serialization.
"""

import struct
import zlib
from dataclasses import dataclass
from numbers import Integral

HEADER = struct.Struct(">4sBI16sIBQHHQIHI")
HEADER_SIZE = HEADER.size
NO_EXPIRY = (1 << 64) - 1
BASE = 1


def _validate_unsigned(value, limit):
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < limit:
        raise ValueError("Invalid packet header value")


class TokenPacker:
    @staticmethod
    def pack_tokens(token_ids) -> bytes:
        bits = 0
        count = 0
        for value in token_ids:
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 8192:
                raise ValueError("Token IDs must be integers in [0, 8192)")
            bits = (bits << 13) | int(value)
            count += 1
        padding = (-13 * count) % 8
        return (bits << padding).to_bytes((13 * count + 7) // 8, "big")

    @staticmethod
    def unpack_tokens(payload: bytes, count: int) -> list[int]:
        if not isinstance(count, int) or count < 0 or len(payload) != (13 * count + 7) // 8:
            raise ValueError("Invalid token count or payload length")
        bits = int.from_bytes(payload, "big")
        padding = (-13 * count) % 8
        if bits & ((1 << padding) - 1):
            raise ValueError("Nonzero token padding")
        bits >>= padding
        return [(bits >> (13 * i)) & 8191 for i in reversed(range(count))]


@dataclass(frozen=True)
class TokenPacket:
    stream_id: int
    model_id: bytes
    window_id: int
    block_id: int
    frame_pts: int
    token_start: int
    token_count: int
    deadline: int
    payload: bytes
    flags: int = 0
    version: int = 1

    def validate(self):
        bounds = (
            (self.stream_id, 2**32),
            (self.window_id, 2**32),
            (self.block_id, 4),
            (self.frame_pts, 2**64),
            (self.deadline, 2**64),
        )
        for value, limit in bounds:
            _validate_unsigned(value, limit)
        if self.version != 1 or len(self.model_id) != 16 or self.flags not in (0, BASE):
            raise ValueError("Unsupported version, model ID, or flags")
        _validate_unsigned(self.token_start, 256)
        _validate_unsigned(self.token_count, 257)
        if not 1 <= self.token_count <= 256 - self.token_start:
            raise ValueError("Invalid token range")
        TokenPacker.unpack_tokens(self.payload, self.token_count)

    def _header(self, checksum):
        return HEADER.pack(
            b"ATKS",
            self.version,
            self.stream_id,
            self.model_id,
            self.window_id,
            self.block_id,
            self.frame_pts,
            self.token_start,
            self.token_count,
            self.deadline,
            len(self.payload),
            self.flags,
            checksum,
        )

    def serialize(self) -> bytes:
        self.validate()
        checksum = zlib.crc32(self._header(0) + self.payload)
        return self._header(checksum) + self.payload

    @classmethod
    def deserialize(cls, data: bytes):
        if len(data) < HEADER_SIZE:
            raise ValueError("Truncated packet header")
        (
            magic,
            version,
            stream,
            model,
            window,
            block,
            pts,
            start,
            count,
            deadline,
            length,
            flags,
            checksum,
        ) = HEADER.unpack_from(data)
        if magic != b"ATKS" or len(data) != HEADER_SIZE + length:
            raise ValueError("Packet magic or length mismatch")
        packet = cls(
            stream,
            model,
            window,
            block,
            pts,
            start,
            count,
            deadline,
            data[HEADER_SIZE:],
            flags,
            version,
        )
        packet.validate()
        if zlib.crc32(packet._header(0) + packet.payload) != checksum:
            raise ValueError("Packet checksum mismatch")
        return packet

    @property
    def wire_bytes(self):
        """TCP message length (4), message kind (1), and serialized packet."""
        return 5 + len(self.serialize())
