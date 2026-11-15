import random

import pytest

from adaptivetokstream.protocol import HEADER_SIZE, TokenPacker, TokenPacket

MODEL = bytes(range(16))


def packet(ids=tuple(range(8)), **kw):
    return TokenPacket(
        stream_id=1,
        model_id=MODEL,
        window_id=0,
        block_id=0,
        frame_pts=0,
        token_start=0,
        token_count=len(ids),
        deadline=1_000_000,
        payload=TokenPacker.pack_tokens(ids),
        **kw,
    )


@pytest.mark.parametrize("n", [0, 1, 7, 8, 16, 31, 32, 64, 128, 256, 1024])
def test_lossless_13_bit_roundtrip(n):
    rng = random.Random(n)
    values = [rng.randrange(8192) for _ in range(n)]
    data = TokenPacker.pack_tokens(values)
    assert len(data) == (13 * n + 7) // 8
    assert TokenPacker.unpack_tokens(data, n) == values


def test_wire_known_vector_and_padding():
    assert TokenPacker.pack_tokens([8191]) == b"\xff\xf8"
    assert TokenPacker.pack_tokens([1, 2]) == b"\x00\x08\x00\x80"
    with pytest.raises(ValueError):
        TokenPacker.unpack_tokens(b"\xff\xff", 1)


@pytest.mark.parametrize("values", [[-1], [8192], [1.5], [True]])
def test_reject_invalid_token(values):
    with pytest.raises(ValueError):
        TokenPacker.pack_tokens(values)


@pytest.mark.parametrize("data,count", [(b"", 1), (b"\x00" * 3, 1), (b"", -1)])
def test_reject_noncanonical_payload(data, count):
    with pytest.raises(ValueError):
        TokenPacker.unpack_tokens(data, count)


def test_packet_stable_wire_and_checksum():
    p = packet()
    wire = p.serialize()
    assert wire[:5] == b"ATKS\x01"
    assert len(wire) == HEADER_SIZE + 13
    assert TokenPacket.deserialize(wire) == p
    assert TokenPacket.deserialize(wire).serialize() == wire
    for i in [0, 4, 9, HEADER_SIZE - 1, len(wire) - 1]:
        corrupted = bytearray(wire)
        corrupted[i] ^= 1
        with pytest.raises(ValueError):
            TokenPacket.deserialize(bytes(corrupted))
    with pytest.raises(ValueError):
        TokenPacket.deserialize(wire[:-1])
    with pytest.raises(ValueError):
        TokenPacket.deserialize(wire + b"0")


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_id", 4),
        ("token_start", 256),
        ("token_count", 257),
        ("window_id", -1),
        ("deadline", -1),
        ("model_id", b"bad"),
        ("flags", 128),
    ],
)
def test_packet_rejects_invalid_header(field, value):
    import dataclasses

    with pytest.raises(ValueError):
        dataclasses.replace(packet(), **{field: value}).serialize()
