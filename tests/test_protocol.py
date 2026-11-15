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


@pytest.mark.parametrize(
    "field,limit",
    [("stream_id", 2**32), ("window_id", 2**32), ("frame_pts", 2**64), ("deadline", 2**64)],
)
def test_header_integer_boundaries(field, limit):
    from dataclasses import replace

    p = replace(packet(), **{field: limit - 1})
    assert TokenPacket.deserialize(p.serialize()) == p
    for value in [limit, limit + 1, -1, True, 1.5]:
        with pytest.raises(ValueError):
            replace(packet(), **{field: value}).validate()


@pytest.mark.parametrize("start,count", [(0, 1), (0, 256), (255, 1), (8, 248)])
def test_complete_legal_range_boundaries(start, count):
    from dataclasses import replace

    p = replace(packet([1] * count), token_start=start)
    assert TokenPacket.deserialize(p.serialize()) == p


@pytest.mark.parametrize("start,count", [(0, 0), (0, 257), (256, 0), (8, 249), (-1, 8), (255, 2)])
def test_invalid_token_range_boundaries(start, count):
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(packet([0] * max(1, count)), token_start=start, token_count=count).validate()


def test_reject_noninteger_unpack_count():
    with pytest.raises(ValueError):
        TokenPacker.unpack_tokens(b"\x00\x00", 1.0)


def test_truncated_header_before_unpack():
    for size in [0, 1, HEADER_SIZE - 1]:
        with pytest.raises(ValueError):
            TokenPacket.deserialize(b"\x00" * size)
