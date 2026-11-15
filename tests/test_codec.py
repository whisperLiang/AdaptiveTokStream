import os
from pathlib import Path

import numpy as np
import pytest

from adaptivetokstream.buffer import TokenBuffer
from adaptivetokstream.codec import AdapTokBackend, ProgressiveVideoDecoder, ProgressiveVideoEncoder
from adaptivetokstream.protocol import NO_EXPIRY, TokenPacker, TokenPacket

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def backend():
    path = Path(os.environ.get("ADAPTOK_WEIGHTS", ROOT / "artifacts/weights/AdapTok"))
    if not (path / "model.safetensors").exists():
        pytest.skip("official checkpoint missing; see README preparation instructions")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return AdapTokBackend(
        ROOT / "third_party/AdapTok", path, device="cuda:0", native_tokens_per_block=512
    )


@pytest.fixture(scope="module")
def encoded(backend):
    import torch

    torch.manual_seed(7)
    frames = torch.rand(1, 3, 16, 128, 128, device=backend.device)
    encoder = ProgressiveVideoEncoder(backend)
    out = encoder.encode_window(frames)
    return frames, encoder, out


@pytest.mark.model
def test_once_only_and_nested_prefixes(encoded):
    frames, e, out = encoded
    assert out.tokens.shape == (4, 256)
    assert out.tokens.min() >= 0 and out.tokens.max() < 8192
    assert e.encode_window(frames, window_id=out.window_id) is out
    assert e.encoding_calls == 1
    for g in range(4):
        assert np.array_equal(e.get_prefix(0, g, 0, 32), e.get_prefix(0, g, 0, 128)[:32])
    import torch

    with pytest.raises(ValueError):
        e.encode_window(torch.zeros(1, 3, 15, 128, 128))
    with pytest.raises(ValueError):
        e.get_prefix(0, 0, 8, 257)


def fill(decoder, tokens, budgets, incremental):
    for g, k in enumerate(budgets):
        steps = [x for x in [8, 16, 32, 64, 128, 256] if x <= k] if incremental else [k]
        start = 0
        for end in steps:
            p = TokenPacket(
                1,
                decoder.backend.model_id,
                0,
                g,
                g * 133333,
                start,
                end - start,
                NO_EXPIRY,
                TokenPacker.pack_tokens(tokens[g, start:end]),
            )
            decoder.append_tokens(TokenPacket.deserialize(p.serialize()), now_us=0)
            start = end


@pytest.mark.model
def test_native_incremental_and_single_packet_agree(backend, encoded):
    import torch

    _, _, out = encoded
    budgets = [8, 32, 64, 128]
    a = ProgressiveVideoDecoder(backend, TokenBuffer(1, backend.model_id))
    b = ProgressiveVideoDecoder(backend, TokenBuffer(1, backend.model_id))
    fill(a, out.tokens, budgets, True)
    fill(b, out.tokens, budgets, False)
    for g in range(4):
        ids, mask = a.buffer.snapshot(0, g)
        native = backend.decode_tokens(np.array(ids), np.array(mask)).frames[
            :, :, g * 4 : g * 4 + 4
        ]
        ar = a.decode_block(0, g)
        br = b.decode_block(0, g)
        assert ar.frames.shape == (1, 3, 4, 128, 128)
        torch.testing.assert_close(ar.frames, br.frames, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(ar.frames, native, atol=2e-5, rtol=2e-5)
        assert ar.elapsed_ms > 0
        a.finalize_block(0, g)
        b.finalize_block(0, g)


@pytest.mark.model
def test_future_groups_cannot_change_past_native_output(backend, encoded):
    import torch

    _, _, out = encoded
    tokens = out.tokens.copy()
    mask = np.zeros((4, 256), bool)
    mask[:, :64] = True
    x = backend.decode_tokens(tokens, mask).frames[:, :, :8]
    tokens[2:] = (tokens[2:] + 123) % 8192
    mask[2:, :] = True
    y = backend.decode_tokens(tokens, mask).frames[:, :, :8]
    torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-5)


@pytest.mark.model
def test_mismatched_native_layout_rejected():
    with pytest.raises(ValueError, match="native"):
        AdapTokBackend(
            ROOT / "third_party/AdapTok",
            ROOT / "artifacts/weights/AdapTok",
            native_tokens_per_block=256,
        )
