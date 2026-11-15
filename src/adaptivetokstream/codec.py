"""Inference adapter; leaves upstream architecture and checkpoint tensors untouched."""

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .buffer import TokenBuffer

BUDGETS = (8, 16, 32, 64, 128, 256)


def checkpoint_id(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


@dataclass
class DecodeResult:
    frames: object
    elapsed_ms: float
    peak_memory_bytes: int
    prefix: int = 0


class AdapTokBackend:
    def __init__(self, upstream_path, weights, device="cuda:0", native_tokens_per_block=256):
        self.weights = Path(weights).resolve()
        config_path = self.weights / "config.json"
        weight_path = self.weights / "model.safetensors"
        if not config_path.is_file() or not weight_path.is_file():
            raise FileNotFoundError(
                f"Missing AdapTok config.json/model.safetensors at {self.weights}. "
                "Run scripts/fetch_weights.py; never use random weights."
            )
        cfg = json.loads(config_path.read_text())
        native = cfg.get("bottleneck_token_num", 0) // 4
        if native != native_tokens_per_block:
            raise ValueError(
                f"Checkpoint native group width is {native}, configured "
                f"{native_tokens_per_block}. See docs/UPSTREAM_AUDIT.md."
            )
        spec = (
            cfg.get("input_size"),
            cfg.get("frame_num"),
            cfg.get("temporal_patch_size"),
            cfg.get("bottleneck", {})
            .get("args", {})
            .get("regularizer", {})
            .get("args", {})
            .get("codebook_size"),
        )
        if spec != (128, 16, 4, 8192) or native < 256:
            raise ValueError("Checkpoint incompatible with 128x128/16-frame/8192-codebook VOD")
        upstream_path = Path(upstream_path).resolve()
        if not (upstream_path / "models/adaptok.py").is_file():
            raise FileNotFoundError(
                f"AdapTok source missing at {upstream_path}; git submodule update --init"
            )
        sys.path.insert(0, str(upstream_path))
        import torch
        from models.adaptok import AdapTok
        from safetensors.torch import load_file

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        # Set device during construction so ordinary upstream attention-mask tensors
        # are created on the same device as the module (they are not registered buffers).
        cfg["device"] = str(self.device)
        self.model = AdapTok(**cfg)
        self.model.load_state_dict(load_file(str(weight_path)), strict=True)
        self.model = self.model.to(self.device).eval().requires_grad_(False)
        self.native_tokens_per_block = native
        self.sha256 = checkpoint_id(weight_path).hex()
        self.model_id = bytes.fromhex(self.sha256)[:16]
        self.dtype = next(self.model.parameters()).dtype

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def encode(self, frames):
        torch = self.torch
        if tuple(frames.shape) != (1, 3, 16, 128, 128):
            raise ValueError("Expected [1,3,16,128,128] RGB frames")
        if (
            not frames.is_floating_point()
            or not torch.isfinite(frames).all()
            or frames.min() < 0
            or frames.max() > 1
        ):
            raise ValueError("Frames must be finite floating RGB in [0,1]")
        with torch.inference_mode():
            output = self.model.encode(
                frames.to(self.device, dtype=self.dtype), with_latent_mask=False
            )
            # Preserve native four-group layout. Public weights have 512/group;
            # only the first 256/group belong to the stream's restricted budget.
            ids = output["bottleneck_rep"].reshape(4, self.native_tokens_per_block)
            return ids[:, :256].cpu().numpy().astype(np.uint16)

    def decode_tokens(self, tokens, mask):
        torch = self.torch
        tokens = np.asarray(tokens)
        mask = np.asarray(mask)
        if tokens.shape != (4, 256) or mask.shape != (4, 256) or mask.dtype != np.bool_:
            raise ValueError("Expected [4,256] IDs and boolean group prefix mask")
        if (
            not np.issubdtype(tokens.dtype, np.integer)
            or np.any(tokens < 0)
            or np.any(tokens >= 8192)
        ):
            raise ValueError("Invalid codebook index")
        # Prefix masks are a protocol invariant, including the zero prefix.
        if np.any(np.diff(mask.astype(np.int8), axis=1) > 0):
            raise ValueError("Each group must use a contiguous prefix")
        self.synchronize()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            ids = torch.zeros(
                (1, 4, self.native_tokens_per_block), dtype=torch.long, device=self.device
            )
            lm = torch.zeros_like(ids, dtype=torch.bool)
            ids[0, :, :256] = torch.as_tensor(tokens.astype(np.int64), device=self.device)
            lm[0, :, :256] = torch.as_tensor(mask, device=self.device)
            embeddings = self.model.bottleneck.decode(ids.flatten(1))
            frames = self.model.decode_with_mask(embeddings, lm.flatten(1))
        self.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        if tuple(frames.shape) != (1, 3, 16, 128, 128) or not torch.isfinite(frames).all():
            raise RuntimeError("Native decoder returned invalid video")
        return DecodeResult(frames, elapsed, peak)


@dataclass(frozen=True)
class EncodedWindow:
    window_id: int
    tokens: np.ndarray
    encode_time_ms: float


class ProgressiveVideoEncoder:
    def __init__(self, backend):
        self.backend = backend
        self.windows = {}
        self.encoding_calls = 0

    def encode_window(self, frames, window_id=None):
        window_id = len(self.windows) if window_id is None else window_id
        if window_id < 0:
            raise ValueError("Window ID must be nonnegative")
        if window_id in self.windows:
            return self.windows[window_id]
        self.backend.synchronize()
        start = time.perf_counter()
        tokens = self.backend.encode(frames)
        self.backend.synchronize()
        tokens.setflags(write=False)
        out = EncodedWindow(window_id, tokens, (time.perf_counter() - start) * 1000)
        self.windows[window_id] = out
        self.encoding_calls += 1
        return out

    def get_prefix(self, window_id, block_id, start, end):
        if not (0 <= block_id < 4 and 0 <= start < end <= 256):
            raise ValueError("Invalid group or token range")
        return self.windows[window_id].tokens[block_id, start:end].copy()


class ProgressiveVideoDecoder:
    def __init__(self, backend, buffer: TokenBuffer):
        self.backend = backend
        self.buffer = buffer
        self.decoded_prefixes = {}

    def append_tokens(self, packet, now_us=0):
        return self.buffer.append(packet, now_us)

    def decode_block(self, window_id, block_id):
        ids, mask = self.buffer.snapshot(window_id, block_id)
        prefix = sum(mask[block_id])
        if prefix < 8:
            raise ValueError("Base prefix has not arrived")
        result = self.backend.decode_tokens(np.array(ids), np.array(mask))
        result.frames = result.frames[:, :, block_id * 4 : (block_id + 1) * 4].clone()
        result.prefix = prefix
        self.decoded_prefixes[(window_id, block_id)] = prefix
        return result

    def finalize_block(self, window_id, block_id, displayed_prefix=None):
        prefix = (
            self.decoded_prefixes.get((window_id, block_id))
            if displayed_prefix is None
            else displayed_prefix
        )
        if prefix is None:
            raise ValueError("Cannot play a block before decode completes")
        self.buffer.finalize(window_id, block_id, prefix)
        self.decoded_prefixes.pop((window_id, block_id), None)
