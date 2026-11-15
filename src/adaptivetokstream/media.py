"""RGB preparation and lossless FFV1 output with explicit source PTS."""

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


@dataclass
class PreparedVideo:
    frames: np.ndarray
    pts_us: np.ndarray
    fps: Fraction
    metadata: dict

    def window(self, index):
        import torch

        frames = self.frames[index * 16 : (index + 1) * 16]
        if len(frames) == 0:
            raise IndexError("Window out of range")
        if len(frames) < 16:
            frames = np.concatenate([frames, np.repeat(frames[-1:], 16 - len(frames), axis=0)])
        return torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).unsqueeze(0).float() / 255


def write_video(path, frames, pts_us, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(frames) != len(pts_us) or len(frames) == 0:
        raise ValueError("Frames and PTS must have equal, nonzero length")
    if np.any(np.diff(pts_us) <= 0):
        raise ValueError("PTS must be strictly increasing")
    # Matroska stores millisecond PTS; NUT retains the requested microsecond timebase.
    with av.open(str(path), "w", format="nut") as container:
        stream = container.add_stream("ffv1", rate=Fraction(fps))
        stream.width = 128
        stream.height = 128
        stream.pix_fmt = "bgr0"
        stream.time_base = Fraction(1, 1_000_000)
        stream.codec_context.time_base = Fraction(1, 1_000_000)
        for image, pts in zip(frames, pts_us):
            frame = av.VideoFrame.from_ndarray(np.asarray(image, dtype=np.uint8), format="rgb24")
            frame.pts = int(pts)
            frame.time_base = Fraction(1, 1_000_000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def prepare_video(input_path, output_dir, limit_frames=None, fps_override=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    pts = []
    with av.open(str(input_path)) as container:
        stream = container.streams.video[0]
        fps = Fraction(str(fps_override)) if fps_override else stream.average_rate
        if fps is None or fps <= 0:
            raise ValueError("Video FPS unavailable; specify --fps")
        frames, pts = _read_frames(container, stream, limit_frames)
    if not frames or np.any(np.diff(pts) <= 0):
        raise ValueError("Video is empty or has non-increasing timestamps")
    original_pts = pts.copy()
    if fps_override:
        pts = [pts[0] + round(i / float(fps) * 1e6) for i in range(len(frames))]
    meta = {
        "source": str(Path(input_path).resolve()),
        "fps": str(fps),
        "frame_count": len(frames),
        "resolution": 128,
        "fps_override": fps_override,
        "source_pts_us": original_pts,
        "pts_us": pts,
        "padding": "repeat last frame only inside final encoder window",
    }
    np.savez_compressed(
        output_dir / "prepared.npz", frames=np.stack(frames), pts_us=np.array(pts, dtype=np.int64)
    )
    (output_dir / "video.json").write_text(json.dumps(meta, indent=2))
    return meta


def read_prepared(path):
    path = Path(path)
    with np.load(path / "prepared.npz", allow_pickle=False) as data:
        frames = data["frames"]
        pts = data["pts_us"]
    meta = json.loads((path / "video.json").read_text())
    return PreparedVideo(frames, pts, Fraction(meta["fps"]), meta)


def to_uint8(frames):
    return (
        (frames.detach().float().clamp(0, 1)[0].permute(1, 2, 3, 0).cpu().numpy() * 255)
        .round()
        .astype(np.uint8)
    )


def _read_frames(container, stream, limit_frames):
    frames, pts = [], []
    for f in container.decode(stream):
        if f.pts is None:
            raise ValueError("Input frame missing PTS")
        pts.append(round(float(f.pts * f.time_base) * 1_000_000))
        # Match upstream evaluation: resize shortest edge then center crop.
        scale = 128 / min(f.width, f.height)
        arr = f.reformat(
            width=max(128, round(f.width * scale)),
            height=max(128, round(f.height * scale)),
            format="rgb24",
        ).to_ndarray()
        top = (arr.shape[0] - 128) // 2
        left = (arr.shape[1] - 128) // 2
        frames.append(arr[top : top + 128, left : left + 128].copy())
        if limit_frames and len(frames) >= limit_frames:
            break
    return frames, pts
