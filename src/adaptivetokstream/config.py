"""One validated configuration shared by offline, simulated and TCP execution."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class CodecConfig:
    upstream_path: str = "third_party/AdapTok"
    weights: str = "artifacts/weights/AdapTok"
    device: str = "cuda:0"
    native_tokens_per_block: int = 256
    resolution: int = 128
    frames_per_window: int = 16
    frames_per_block: int = 4
    tokens_per_block: int = 256
    codebook_size: int = 8192
    budgets: tuple = (8, 16, 32, 64, 128, 256)


@dataclass
class StreamingConfig:
    mode: str = "vod"
    policy: str = "deadline_mpc"
    fixed_tokens: int = 64
    initial_buffer_seconds: float = 1.0
    max_buffer_seconds: float = 5.0
    min_tokens: int = 8


@dataclass
class NetworkConfig:
    swift_path: str = "third_party/swift"
    trace_path: str | None = None
    bandwidth_mbps: float = 1.0
    initial_estimate_mbps: float = 1.0
    rtt_ms: float = 20.0
    jitter_ms: float = 0.0
    ewma_alpha: float = 0.3


@dataclass
class SchedulerConfig:
    horizon_blocks: int = 4
    beam_width: int = 12
    use_decode_cost: bool = True
    use_content_quality: bool = True
    stall_penalty: float = 10.0
    variation_penalty: float = 0.1
    byte_penalty: float = 0.000001
    max_video_bytes: int | None = None


@dataclass
class EvaluationConfig:
    metrics: tuple = ("psnr", "lpips", "dists")
    save_reconstructions: bool = True
    save_packet_logs: bool = True
    save_playback_logs: bool = True
    warmup: int = 3
    repeats: int = 5
    history_budget: int = 256


@dataclass
class Config:
    codec: CodecConfig = field(default_factory=CodecConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42

    def validate(self):
        c = self.codec
        if (
            c.resolution,
            c.frames_per_window,
            c.frames_per_block,
            c.tokens_per_block,
            c.codebook_size,
        ) != (128, 16, 4, 256, 8192):
            raise ValueError(
                "Only native 128x128, 16-frame windows, four groups, 256 stream IDs/group and 8192 entries are supported"
            )
        if tuple(c.budgets) != (8, 16, 32, 64, 128, 256):
            raise ValueError("Budgets must be [8,16,32,64,128,256]")
        if self.streaming.mode != "vod" or self.streaming.min_tokens != 8:
            raise ValueError("Only pre-encoded VOD with 8-token base is supported")
        if (
            self.streaming.policy not in ("fixed", "greedy", "deadline_mpc")
            or self.streaming.fixed_tokens not in c.budgets
        ):
            raise ValueError("Unknown policy or fixed budget")
        if not 0 <= self.streaming.initial_buffer_seconds <= self.streaming.max_buffer_seconds:
            raise ValueError("Invalid playback buffer limits")
        if (
            self.streaming.max_buffer_seconds <= 0
            or self.network.bandwidth_mbps <= 0
            or self.network.initial_estimate_mbps <= 0
        ):
            raise ValueError("Positive buffer and bandwidth required")
        if (
            self.network.rtt_ms < 0
            or self.network.jitter_ms < 0
            or not 0 < self.network.ewma_alpha <= 1
        ):
            raise ValueError("Invalid network timing or estimator alpha")
        if self.scheduler.horizon_blocks < 1 or self.scheduler.beam_width < 1:
            raise ValueError("Positive MPC horizon and beam width required")
        if self.scheduler.max_video_bytes is not None and self.scheduler.max_video_bytes < 1:
            raise ValueError("Byte cap must be positive")
        if (
            self.evaluation.repeats < 1
            or self.evaluation.warmup < 0
            or self.evaluation.history_budget not in c.budgets
        ):
            raise ValueError("Invalid profiling settings")
        if tuple(self.evaluation.metrics) != ("psnr", "lpips", "dists"):
            raise ValueError("All three reference metrics are required")
        return self

    def backend(self):
        from .codec import AdapTokBackend

        return AdapTokBackend(
            self.codec.upstream_path,
            self.codec.weights,
            self.codec.device,
            self.codec.native_tokens_per_block,
        )

    def to_dict(self):
        return asdict(self)


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    classes = {
        "codec": CodecConfig,
        "streaming": StreamingConfig,
        "network": NetworkConfig,
        "scheduler": SchedulerConfig,
        "evaluation": EvaluationConfig,
    }
    unknown = set(raw) - set(classes) - {"seed"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    cfg = Config(
        **{key: cls(**raw.get(key, {})) for key, cls in classes.items()}, seed=raw.get("seed", 42)
    )
    for obj, key in [
        (cfg.codec, "upstream_path"),
        (cfg.codec, "weights"),
        (cfg.network, "swift_path"),
        (cfg.network, "trace_path"),
    ]:
        value = getattr(obj, key)
        if value is not None:
            setattr(obj, key, str((path.parent / value).resolve()))
    return cfg.validate()
