"""One receiver owns all mutable token state; committed playback masks never grow."""
from dataclasses import dataclass, field
from .protocol import TokenPacket


@dataclass
class ReceiveStats:
    received_bytes: int = 0
    received_payload_bytes: int = 0
    duplicate_payload_bytes: int = 0
    duplicate_packets: int = 0
    expired_packets: int = 0
    played_packets: int = 0
    accepted_tokens: int = 0


@dataclass
class BlockTokens:
    ids: list = field(default_factory=lambda: [None]*256)
    prefix: int = 0
    played: bool = False
    frozen_prefix: int = 0
    deadline: int | None = None
    pts: int | None = None


@dataclass(frozen=True)
class AppendResult:
    status: str
    new_tokens: int = 0
    prefix: int = 0


class TokenBuffer:
    def __init__(self, stream_id: int, model_id: bytes):
        self.stream_id = stream_id
        self.model_id = model_id
        self.windows: dict[int, list[BlockTokens]] = {}
        # Contiguous VOD retirement watermark avoids keeping unbounded tombstones.
        self.retired_through = -1
        self.stats = ReceiveStats()

    def _window(self, window_id):
        if window_id <= self.retired_through:
            raise ValueError('Window already retired')
        return self.windows.setdefault(window_id, [BlockTokens() for _ in range(4)])

    def block(self, window_id, block_id):
        if not 0 <= block_id < 4:
            raise ValueError('Block ID must be in [0,4)')
        return self._window(window_id)[block_id]

    def prefix(self, window_id, block_id):
        return self.block(window_id, block_id).prefix

    def append(self, packet: TokenPacket, now_us: int) -> AppendResult:
        from .protocol import TokenPacker
        packet.validate()
        if packet.stream_id != self.stream_id or packet.model_id != self.model_id:
            raise ValueError('Packet belongs to a different stream or model')
        self.stats.received_bytes += packet.wire_bytes
        self.stats.received_payload_bytes += len(packet.payload)
        if packet.window_id <= self.retired_through:
            return AppendResult('retired')
        block = self.block(packet.window_id, packet.block_id)
        if block.played:
            self.stats.played_packets += 1
            return AppendResult('played', prefix=block.prefix)
        if now_us > packet.deadline:
            self.stats.expired_packets += 1
            return AppendResult('expired', prefix=block.prefix)
        values = TokenPacker.unpack_tokens(packet.payload, packet.token_count)
        positions = range(packet.token_start, packet.token_start + packet.token_count)
        if block.pts is not None and block.pts != packet.frame_pts:
            raise ValueError('Conflicting block PTS')
        if any(block.ids[i] is not None and block.ids[i] != v for i, v in zip(positions, values)):
            raise ValueError('Conflicting overlapping token ID')
        new = sum(block.ids[i] is None for i in positions)
        duplicates = len(values)-new
        self.stats.duplicate_payload_bytes += (13*duplicates+7)//8
        for i, v in zip(positions, values):
            block.ids[i] = v
        while block.prefix < 256 and block.ids[block.prefix] is not None:
            block.prefix += 1
        block.pts = packet.frame_pts
        self.stats.accepted_tokens += new
        if not new:
            self.stats.duplicate_packets += 1
        return AppendResult('accepted' if new else 'duplicate', new, block.prefix)

    def snapshot(self, window_id, target_block):
        """Hide future groups; use immutable displayed prefixes for past groups."""
        groups = self._window(window_id)
        ids, masks = [], []
        for g, block in enumerate(groups):
            prefix = 0 if g > target_block else (block.frozen_prefix if block.played else block.prefix)
            masks.append([i < prefix for i in range(256)])
            ids.append([int(block.ids[i]) if i < prefix else 0 for i in range(256)])
        return ids, masks

    def finalize(self, window_id, block_id, displayed_prefix=None):
        block = self.block(window_id, block_id)
        k = block.prefix if displayed_prefix is None else displayed_prefix
        if not 0 < k <= block.prefix:
            raise ValueError('Cannot freeze missing or empty token prefix')
        if block.played and k != block.frozen_prefix:
            raise ValueError('Cannot revise a played block')
        block.frozen_prefix = k
        block.played = True

    def clear_window(self, window_id):
        if window_id != self.retired_through+1:
            raise ValueError('Retire windows in order')
        if not all(b.played for b in self._window(window_id)):
            raise ValueError('Window still has unplayed blocks')
        del self.windows[window_id]
        self.retired_through = window_id
