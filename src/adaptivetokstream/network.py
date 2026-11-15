"""Swift trace intervals, byte-exact service, and feedback-only EWMA."""

import importlib.util
import itertools
import math
import os
import random
import tempfile
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path


class BandwidthTrace:
    def __init__(self, times, mbps):
        if len(times) < 2 or len(times) != len(mbps):
            raise ValueError("Trace requires at least two matched samples")
        if any(not math.isfinite(x) for x in list(times) + list(mbps)):
            raise ValueError("Trace contains non-finite values")
        if any(a >= b for a, b in itertools.pairwise(times)):
            raise ValueError("Trace timestamps must increase")
        self._validate_capacity(mbps)
        self.times = list(times)
        self.mbps = list(mbps)
        self.period = times[-1] - times[0]

    @staticmethod
    def _validate_capacity(mbps):
        if any(b < 0 for b in mbps) or max(mbps[1:]) == 0:
            raise ValueError("Trace timestamps must increase and carry positive capacity")

    @classmethod
    def constant(cls, mbps):
        return cls([0.0, 3600.0], [mbps, mbps])

    @classmethod
    def from_swift(cls, path, swift_path):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Network trace missing: {path}")
        module_path = Path(swift_path) / "streamer/common.py"
        spec = importlib.util.spec_from_file_location("_adaptivetok_swift_common", module_path)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # The upstream loader only accepts directories. A temporary symlink selects
        # one original trace without copying or rewriting its samples/timestamps.
        with tempfile.TemporaryDirectory(prefix="adaptivetok-trace-") as folder:
            os.symlink(path, Path(folder) / path.name)
            times, bw, _ = module.load_trace(folder + os.sep)
        return cls(times[0], bw[0])

    def rate_and_boundary(self, now):
        cycle = math.floor(now / self.period)
        relative = now - cycle * self.period + self.times[0]
        index = min(bisect_right(self.times, relative), len(self.times) - 1)
        return self.mbps[index] * 125000, cycle * self.period + self.times[index] - self.times[0]

    def finish(self, size_bytes, start):
        if size_bytes < 0 or start < 0:
            raise ValueError("Nonnegative bytes and time required")
        remaining = float(size_bytes)
        now = float(start)
        while remaining > 1e-8:
            rate, boundary = self.rate_and_boundary(now)
            duration = boundary - now
            if duration < 1e-12:
                now = math.nextafter(now, math.inf)
                continue
            capacity = rate * duration
            if rate > 0 and remaining <= capacity + 1e-8:
                return now + remaining / rate
            remaining -= capacity
            now = boundary
        return now


@dataclass(frozen=True)
class Transfer:
    size: int
    service_start: float
    service_end: float
    delivery: float


class TraceLink:
    def __init__(self, trace, rtt_ms=20, jitter_ms=0, seed=42):
        self.trace = trace
        self.one_way = rtt_ms / 2000
        self.jitter = jitter_ms / 1000
        self.rng = random.Random(seed)
        self.available_at = 0.0
        self.last_delivery = 0.0

    def send(self, size, now):
        start = max(now, self.available_at)
        end = self.trace.finish(size, start)
        delay = max(0, self.one_way + self.rng.uniform(-self.jitter, self.jitter))
        # TCP preserves application message order even with fluctuating delay.
        delivery = max(self.last_delivery, end + delay)
        self.available_at = end
        self.last_delivery = delivery
        return Transfer(size, start, end, delivery)


class ThroughputEstimator:
    def __init__(self, initial_bytes_per_second, alpha=0.3):
        self.estimate = float(initial_bytes_per_second)
        self.alpha = alpha
        self.recent = []

    def observe(self, size_bytes, service_seconds):
        if service_seconds <= 0 or size_bytes <= 0:
            raise ValueError("Positive completed sample required")
        throughput = size_bytes / service_seconds
        self.recent = (self.recent + [throughput])[-8:]
        self.estimate = (1 - self.alpha) * self.estimate + self.alpha * throughput
