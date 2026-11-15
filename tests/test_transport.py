import pytest

from adaptivetokstream.transport import FrameReader, frame_message


def test_length_prefixed_messages_survive_arbitrary_tcp_chunks():
    data = frame_message(b"P", b"hello") + frame_message(b"F", b"{}")
    reader = FrameReader()
    messages = []
    for byte in data:
        messages.extend(reader.feed(bytes([byte])))
    assert messages == [(b"P", b"hello", 10), (b"F", b"{}", 7)]
    assert reader.pending_bytes == 0


def test_invalid_length_and_truncated_eof():
    import struct

    reader = FrameReader()
    with pytest.raises(ValueError):
        reader.feed(struct.pack(">I", 0))
    reader = FrameReader()
    reader.feed(b"\x00\x00")
    with pytest.raises(ValueError):
        reader.finish()
