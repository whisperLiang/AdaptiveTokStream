"""Unified, reproducible CLI for preprocessing, inference, simulation and TCP."""

import argparse
import json
import sys
from pathlib import Path

from .config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/default.yaml"


def parser():
    p = argparse.ArgumentParser(prog="adaptivetokstream")
    sub = p.add_subparsers(dest="command", required=True)
    for name in [
        "prepare-video",
        "encode-video",
        "profile-codec",
        "simulate",
        "run-sender",
        "run-receiver",
        "evaluate",
        "experiments",
        "socket-demo",
    ]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", default=str(DEFAULT_CONFIG))
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--seed", type=int)
        cmd.add_argument("--weights")
        cmd.add_argument("--device")
        if name == "prepare-video":
            cmd.add_argument("--input", required=True)
            cmd.add_argument("--frames", type=int)
            cmd.add_argument("--fps", type=float)
        if name == "encode-video":
            cmd.add_argument("--prepared", required=True)
        if name in (
            "profile-codec",
            "simulate",
            "run-sender",
            "evaluate",
            "experiments",
            "socket-demo",
        ):
            cmd.add_argument("--asset", required=True)
        if name in ("simulate", "run-sender", "experiments", "socket-demo"):
            cmd.add_argument("--rd", required=True)
        if name in ("simulate", "run-sender", "experiments", "socket-demo"):
            cmd.add_argument("--trace")
            cmd.add_argument("--bandwidth-mbps", type=float)
            cmd.add_argument("--policy", choices=["fixed", "greedy", "deadline_mpc"])
            cmd.add_argument("--token-budget", type=int, choices=[8, 16, 32, 64, 128, 256])
            cmd.add_argument("--byte-cap", type=int)
        if name == "simulate":
            cmd.add_argument("--decode-scale", type=float, default=1)
        if name == "evaluate":
            cmd.add_argument("--log", required=True)
        if name in ("run-sender", "run-receiver"):
            cmd.add_argument("--host", default="127.0.0.1")
            cmd.add_argument("--port", type=int, default=8765)
        if name == "run-receiver":
            cmd.add_argument("--ready-file")
    return p


def execute(args):
    cfg = load_config(args.config)
    for arg, obj, key in [
        ("seed", cfg, "seed"),
        ("weights", cfg.codec, "weights"),
        ("device", cfg.codec, "device"),
        ("trace", cfg.network, "trace_path"),
        ("bandwidth_mbps", cfg.network, "bandwidth_mbps"),
        ("policy", cfg.streaming, "policy"),
        ("token_budget", cfg.streaming, "fixed_tokens"),
        ("byte_cap", cfg.scheduler, "max_video_bytes"),
    ]:
        value = getattr(args, arg, None)
        if value is not None:
            setattr(obj, key, value)
    cfg.validate()
    import random

    import numpy as np

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "effective_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    cmd = args.command
    if cmd == "prepare-video":
        from .media import prepare_video

        result = prepare_video(args.input, args.output, args.frames, args.fps)
        return {key: result[key] for key in ["frame_count", "fps", "resolution"]}
    if cmd == "encode-video":
        from .asset import encode_video

        a = encode_video(args.prepared, args.output, cfg.backend())
        return {"windows": len(a.tokens), "encode_time_ms": a.metadata["encode_time_ms"]}
    if cmd == "profile-codec":
        from .profiling import profile_codec

        r = profile_codec(args.asset, args.output, cfg.backend(), cfg)
        return {
            key: r[key]
            for key in ["precompute_seconds", "metadata_storage_bytes", "peak_memory_bytes"]
        }
    if cmd == "simulate":
        from .asset import VideoAsset
        from .scheduler import RDTable
        from .simulator import simulate

        return simulate(
            VideoAsset.load(args.asset),
            RDTable.load(args.rd),
            cfg,
            decode_scale=args.decode_scale,
            output_dir=args.output,
        )["summary"]
    if cmd == "evaluate":
        from .evaluation import evaluate_playback

        return evaluate_playback(
            args.asset, args.log, args.output, cfg.backend(), cfg.evaluation.save_reconstructions
        )
    if cmd == "run-receiver":
        from .socket_runtime import run_receiver

        return run_receiver(cfg, args.host, args.port, args.output, args.ready_file)
    if cmd == "run-sender":
        from .socket_runtime import run_sender

        return run_sender(cfg, args.asset, args.rd, args.host, args.port, args.output)
    if cmd == "socket-demo":
        from .experiments import run_socket_experiment

        return run_socket_experiment(cfg, args.asset, args.rd, args.output)
    from .experiments import run_experiments

    return run_experiments(cfg, args.asset, args.rd, args.output)


def main():
    args = parser().parse_args()
    try:
        result = execute(args)
    except (ValueError, FileNotFoundError, RuntimeError, ConnectionError, OSError) as error:
        print(f"AdaptiveTokStream: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
