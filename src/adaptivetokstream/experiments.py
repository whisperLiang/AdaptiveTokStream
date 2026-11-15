"""Small reproducible B/C suite: every quality number uses native reconstruction."""

import csv
import json
from copy import deepcopy
from pathlib import Path

from .asset import VideoAsset
from .evaluation import evaluate_playback
from .metrics import QualityMetrics
from .network import BandwidthTrace
from .scheduler import RDTable
from .simulator import simulate


def run_experiments(config, asset_path, rd_path, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    asset = VideoAsset.load(asset_path)
    backend = config.backend()
    metrics = QualityMetrics(backend.device)
    root = Path(__file__).resolve().parents[2]
    swift_trace = (
        Path(config.network.trace_path)
        if config.network.trace_path
        else Path(config.network.swift_path) / "streamer/datasets/test/oboe_trace_2"
    )
    scenarios = {
        "swift_original": BandwidthTrace.from_swift(swift_trace, config.network.swift_path),
        "drop_recovery": BandwidthTrace.from_swift(
            root / "configs/traces/drop_recovery.txt", config.network.swift_path
        ),
        "fixed_0.02mbps": BandwidthTrace.constant(0.02),
    }
    rows = []

    def run(label, cfg, scenario, experiment):
        folder = out / experiment / scenario / label
        simulate(asset, RDTable.load(rd_path), cfg, scenarios[scenario], output_dir=folder)
        summary = evaluate_playback(
            asset_path,
            folder / "simulation.json",
            folder,
            backend,
            cfg.evaluation.save_reconstructions,
            metrics,
        )
        rows.append(
            {
                "experiment": experiment,
                "scenario": scenario,
                "label": label,
                "byte_cap": cfg.scheduler.max_video_bytes,
                **summary,
            }
        )
        (out / "results.json").write_text(json.dumps(rows, indent=2))
        print(
            f"{experiment}/{scenario}/{label}: PSNR={summary['average_psnr']:.3f}, bytes={summary['sent_video_bytes']}, stall={summary['stall_seconds']:.3f}s",
            flush=True,
        )

    for scenario in scenarios:
        for k in config.codec.budgets:
            cfg = deepcopy(config)
            cfg.streaming.policy = "fixed"
            cfg.streaming.fixed_tokens = k
            run(f"fixed_{k}", cfg, scenario, "B")
        for policy in ["greedy", "deadline_mpc"]:
            cfg = deepcopy(config)
            cfg.streaming.policy = policy
            run(policy, cfg, scenario, "B")
    # Same source, trace, decoder profile and serialized-byte cap. Each method may
    # leave capacity unused; consumed bytes are reported, never rescaled to match.
    cap = config.scheduler.max_video_bytes or max(4000, asset.block_count * 100)
    for content, decode, label in [
        (False, False, "bandwidth_only"),
        (True, False, "bandwidth_content"),
        (True, True, "bandwidth_content_decode"),
    ]:
        cfg = deepcopy(config)
        cfg.streaming.policy = "deadline_mpc"
        cfg.scheduler.use_content_quality = content
        cfg.scheduler.use_decode_cost = decode
        cfg.scheduler.max_video_bytes = cap
        run(label, cfg, "drop_recovery", "C")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "runs": len(rows),
        "results": str((out / "results.csv").resolve()),
        "shared_ablation_byte_cap": cap,
    }


def run_socket_experiment(config, asset_path, rd_path, output_dir):
    """Experiment D launches the same two CLI programs as separate OS processes."""
    import socket
    import subprocess
    import sys
    import time

    import yaml

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "run_config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    ready = out / "receiver.ready"
    if ready.exists():
        ready.unlink()
    base = [sys.executable, "-m", "adaptivetokstream"]
    with open(out / "receiver.log", "w") as log:
        receiver = subprocess.Popen(
            base
            + [
                "run-receiver",
                "--config",
                str(config_path),
                "--output",
                str(out / "receiver"),
                "--port",
                str(port),
                "--ready-file",
                str(ready),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while not ready.exists():
                if receiver.poll() is not None:
                    raise RuntimeError((out / "receiver.log").read_text())
                if time.monotonic() > deadline:
                    raise RuntimeError("Receiver startup timed out")
                time.sleep(0.05)
            with open(out / "sender.log", "w") as sender_log:
                subprocess.run(
                    base
                    + [
                        "run-sender",
                        "--config",
                        str(config_path),
                        "--asset",
                        str(Path(asset_path).resolve()),
                        "--rd",
                        str(Path(rd_path).resolve()),
                        "--output",
                        str(out / "sender"),
                        "--port",
                        str(port),
                    ],
                    stdout=sender_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=180,
                )
            if receiver.wait(timeout=30) != 0:
                raise RuntimeError((out / "receiver.log").read_text())
        finally:
            if receiver.poll() is None:
                receiver.terminate()
                receiver.wait(timeout=5)
    rx = json.loads((out / "receiver/receiver.json").read_text())
    tx = json.loads((out / "sender/sender.json").read_text())
    if tx["summary"]["application_bytes"]["received"] != rx["summary"]["application_bytes"]["sent"]:
        raise RuntimeError("Reverse-direction byte counters disagree")
    # Score actual socket receiver output; it never loads the sender's token array.
    from .evaluation import evaluate_reconstruction

    summary = evaluate_reconstruction(
        asset_path,
        out / "receiver/reconstruction.nut",
        out / "receiver/receiver.json",
        out / "evaluation",
        config.codec.device,
    )
    result = {"sender": tx["summary"], "receiver": rx["summary"], "quality": summary}
    (out / "experiment_d.json").write_text(json.dumps(result, indent=2))
    ready.unlink()
    return result
