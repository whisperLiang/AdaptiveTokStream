"""Measured content RD metadata and full-window decoder latency distributions."""

import csv
import json
import time
from pathlib import Path

import numpy as np

from .asset import VideoAsset
from .codec import BUDGETS
from .media import read_prepared, to_uint8, write_video
from .metrics import QualityMetrics


def latency_summary(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "samples_ms": list(map(float, a)),
    }


class CodecProfiler:
    def __init__(self, asset_path, output_dir, backend, config):
        self.asset = VideoAsset.load(asset_path)
        if self.asset.metadata["model_id"] != backend.model_id.hex():
            raise ValueError("Profiling model differs from encoder checkpoint")
        self.video = read_prepared(self.asset.metadata["prepared_dir"])
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.config = config
        self.started = time.perf_counter()
        self.metrics = QualityMetrics(backend.device)
        self.rows = []
        self.uniform_rows = []
        self.timing = {}
        self.peak = 0

    def metric_record(self, index, k, result, decode_ms):
        w, g = divmod(index, 4)
        count = min(4, len(self.video.frames) - index * 4)
        quality = self.metrics.evaluate(
            self.video.frames[index * 4 : index * 4 + count],
            result.frames[:, :, g * 4 : g * 4 + count],
        )
        packet = self.asset.packet(index, 0, k)
        return {
            "window_id": w,
            "block_id": g,
            "token_budget": k,
            "actual_bytes": packet.wire_bytes,
            "token_payload_bytes": len(packet.payload),
            "decode_time_ms": decode_ms,
            **quality,
        }

    def repeated_decode(self, tokens, mask):
        trials = []
        for _ in range(self.config.evaluation.repeats):
            result = self.backend.decode_tokens(tokens, mask)
            trials.append(result.elapsed_ms)
            self.peak = max(self.peak, result.peak_memory_bytes)
        return result, trials

    def uniform_budget(self, k):
        measurements = []
        recon = []
        mask = np.arange(256)[None, :] < np.full((4, 1), k)
        for w, tokens in enumerate(self.asset.tokens):
            result, trials = self.repeated_decode(tokens, mask)
            measurements.extend(trials)
            valid = min(16, len(self.video.frames) - w * 16)
            recon.append(to_uint8(result.frames)[:valid])
            for index in range(w * 4, min(w * 4 + 4, self.asset.block_count)):
                self.uniform_rows.append(
                    self.metric_record(index, k, result, float(np.mean(trials)))
                )
        self.timing[str(k)] = latency_summary(measurements)
        if self.config.evaluation.save_reconstructions:
            self.save_budget(k, np.concatenate(recon))

    def save_budget(self, k, pixels):
        from PIL import Image

        write_video(self.out / f"budget_{k}.nut", pixels, self.video.pts_us, self.video.fps)
        Image.fromarray(pixels[len(pixels) // 2]).save(self.out / f"budget_{k}.png")

    def group_curves(self):
        history = self.config.evaluation.history_budget
        for index in range(self.asset.block_count):
            w, g = divmod(index, 4)
            for k in BUDGETS:
                mask = np.zeros((4, 256), bool)
                mask[:g, :history] = True
                mask[g, :k] = True
                result = self.backend.decode_tokens(self.asset.tokens[w], mask)
                record = self.metric_record(index, k, result, result.elapsed_ms)
                record.update(
                    history_budgets=[history] * g, decode_p95_ms=self.timing[str(k)]["p95_ms"]
                )
                self.rows.append(record)

    def save(self):
        result = {
            "format": 1,
            "model_id": self.backend.model_id.hex(),
            "asset_path": str(self.asset.path.resolve()),
            "history_budget": self.config.evaluation.history_budget,
            "estimator": "measured VOD curves, history-dependent approximation",
            "decode_scope": "full native 16-frame window, then extract 4 frames",
            "timing": self.timing,
            "peak_memory_bytes": self.peak,
            "records": self.rows,
            "precompute_seconds": time.perf_counter() - self.started,
        }
        (self.out / "rd.json").write_text(json.dumps(result, indent=2))
        result["metadata_storage_bytes"] = (self.out / "rd.json").stat().st_size
        summary = {key: value for key, value in result.items() if key != "records"}
        (self.out / "profile_summary.json").write_text(json.dumps(summary, indent=2))
        with open(self.out / "experiment_a.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.uniform_rows[0]))
            writer.writeheader()
            writer.writerows(self.uniform_rows)
        return result

    def run(self):
        for _ in range(self.config.evaluation.warmup):
            self.backend.decode_tokens(self.asset.tokens[0], np.ones((4, 256), bool))
        for k in BUDGETS:
            self.uniform_budget(k)
        self.group_curves()
        return self.save()


def profile_codec(asset_path, output_dir, backend, config):
    return CodecProfiler(asset_path, output_dir, backend, config).run()
