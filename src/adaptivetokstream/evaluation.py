"""Replay displayed historical masks through the real decoder; never score RD estimates."""

import csv
import json
from pathlib import Path

import numpy as np

from .asset import VideoAsset
from .media import read_prepared, to_uint8, write_video
from .metrics import QualityMetrics


def validate_playback(records, block_count):
    if len(records) != block_count or [r["block"] for r in records] != list(range(block_count)):
        raise ValueError("Playback log must include every block in PTS order")
    for r in records:
        i = r["block"]
        g = i % 4
        expected = [records[j]["prefix"] for j in range(i - g, i)]
        if r["history_budgets"] != expected or r["prefix"] not in (8, 16, 32, 64, 128, 256):
            raise ValueError("Playback mask uses a historical budget that was not displayed")


def evaluate_playback(asset_path, log_path, output_dir, backend, save_video=True, metrics=None):
    asset = VideoAsset.load(asset_path)
    if asset.metadata["model_id"] != backend.model_id.hex():
        raise ValueError("Evaluation model ID mismatch")
    log = json.loads(Path(log_path).read_text())
    records = log["playback"]
    validate_playback(records, asset.block_count)
    video = read_prepared(asset.metadata["prepared_dir"])
    metrics = metrics or QualityMetrics(backend.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    recon = []
    for record in records:
        i = record["block"]
        w, g = divmod(i, 4)
        mask = np.zeros((4, 256), bool)
        for j, k in enumerate(record["history_budgets"]):
            mask[j, :k] = True
        mask[g, : record["prefix"]] = True
        result = backend.decode_tokens(asset.tokens[w], mask)
        count = min(4, len(video.frames) - i * 4)
        pixels = result.frames[:, :, g * 4 : g * 4 + count]
        q = metrics.evaluate(video.frames[i * 4 : i * 4 + count], pixels)
        rows.append({**record, **q, "replay_decode_ms": result.elapsed_ms})
        recon.append(to_uint8(pixels))
    if save_video:
        write_video(
            output_dir / "reconstruction.nut", np.concatenate(recon), video.pts_us, video.fps
        )
    summary = {
        **log.get("summary", {}),
        **{
            f"average_{m}": float(np.mean([r[m] for r in rows]))
            for m in ["psnr", "lpips", "dists", "mse"]
        },
        "quality_variation_psnr": float(np.mean(np.abs(np.diff([r["psnr"] for r in rows]))))
        if len(rows) > 1
        else 0,
        "quality_source": "actual native decoder replay with displayed historical masks",
        "evaluation_replay_seconds": sum(r["replay_decode_ms"] for r in rows) / 1000,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps({"summary": summary, "blocks": rows}, indent=2)
    )
    with open(output_dir / "blocks.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def evaluate_reconstruction(asset_path, video_path, log_path, output_dir, device):
    """Score the actual receiver file and verify every preserved source PTS."""
    import av

    asset = VideoAsset.load(asset_path)
    source = read_prepared(asset.metadata["prepared_dir"])
    with av.open(str(video_path)) as container:
        frames = list(container.decode(video=0))
    actual_pts = [round(float(f.pts * f.time_base) * 1e6) for f in frames]
    if actual_pts != asset.metadata["pts_us"] or any(
        (f.width, f.height) != (128, 128) for f in frames
    ):
        raise ValueError("Receiver frame count, dimensions or PTS differ from original")
    pixels = np.stack([f.to_ndarray(format="rgb24") for f in frames])
    log = json.loads(Path(log_path).read_text())
    validate_playback(log["playback"], asset.block_count)
    metrics = QualityMetrics(device)
    rows = []
    for i, record in enumerate(log["playback"]):
        begin = i * 4
        end = min(begin + 4, len(frames))
        rows.append({**record, **metrics.evaluate(source.frames[begin:end], pixels[begin:end])})
    summary = {
        **log["summary"],
        **{
            f"average_{m}": float(np.mean([r[m] for r in rows]))
            for m in ["psnr", "lpips", "dists", "mse"]
        },
        "quality_source": "actual receiver FFV1 file after TCP transport",
        "quality_variation_psnr": float(np.mean(np.abs(np.diff([r["psnr"] for r in rows]))))
        if len(rows) > 1
        else 0,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps({"summary": summary, "blocks": rows}, indent=2))
    return summary
