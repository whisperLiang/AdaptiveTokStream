from fractions import Fraction

import av
import numpy as np

from adaptivetokstream.media import prepare_video, read_prepared, write_video


def test_pts_frame_count_and_window_padding(tmp_path):
    source = tmp_path / "input.mkv"
    pts = np.arange(19, dtype=np.int64) * 40000 + 120000
    frames = np.full((19, 128, 128, 3), 100, dtype=np.uint8)
    write_video(source, frames, pts, Fraction(25, 1))
    out = tmp_path / "prepared"
    prepare_video(source, out)
    data = read_prepared(out)
    assert data.frames.shape == (19, 128, 128, 3)
    assert np.array_equal(data.pts_us, pts)
    assert data.window(1).shape == (1, 3, 16, 128, 128)
    result = tmp_path / "output.mkv"
    write_video(result, data.frames, data.pts_us, data.fps)
    with av.open(str(result)) as c:
        decoded = list(c.decode(video=0))
        assert len(decoded) == 19
        actual = np.array([round(float(f.pts * f.time_base) * 1e6) for f in decoded])
        np.testing.assert_array_equal(actual, pts)
