import numpy as np
import pytest

from brain_lora.constants import (
    HRF_DELAY_S,
    TR,
    WINDOW_FRAMES,
    WINDOW_S,
    WINDOW_TRS,
    segment_frame_bounds,
    slice_fmri_window,
    window_starts,
)
from brain_lora.model.readouts import build_lag_operator

BLOCKS = [(0.5, 2), (2.5, 2), (4.5, 4), (8.5, 8), (16.5, 16), (32.5, 32), (64.5, 64),
          (128.5, 64)]


def test_window_constants():
    assert WINDOW_TRS * TR == WINDOW_S == 149.0
    assert WINDOW_FRAMES == 298


@pytest.mark.parametrize("duration", [10.0, 148.9, 149.0, 452.3, 700.01, 1203.7])
def test_segments_tile_the_clip(duration):
    n_out, bounds = segment_frame_bounds(duration)
    assert n_out == round(duration * 2)
    assert bounds[0][0] == 0 and bounds[-1][1] == n_out
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
    # every segment but the first starts on a window boundary
    starts = window_starts(duration)
    assert [b[0] for b in bounds[1:]] == [round(s * 2) for s in starts[1:len(bounds)]]


def test_first_window_starts_before_the_stimulus():
    starts = window_starts(600.0)
    assert starts[0] == -HRF_DELAY_S
    assert np.allclose(np.diff(starts), WINDOW_S)


def test_fmri_window_is_shifted_and_padded():
    fmri = np.arange(250, dtype=np.float32)[:, None].repeat(3, 1)
    w0 = slice_fmri_window(fmri, window_starts(400.0)[0])
    assert np.array_equal(w0[:, 0], np.arange(100))
    w2 = slice_fmri_window(fmri, window_starts(400.0)[2])
    assert np.array_equal(w2[:50, 0], np.arange(200, 250))
    assert not w2[50:].any()


def test_lag_operator_rows_average_or_vanish():
    A = build_lag_operator(BLOCKS).numpy()
    sums = A.sum(1)
    assert np.allclose(sums[sums > 0], 1.0)
    # the longest block lies before the window for early TRs
    assert sums.reshape(WINDOW_TRS, len(BLOCKS))[0, -1] == 0
    # causal: TR j never averages frames after its own centre
    for j in range(WINDOW_TRS):
        rows = A[j * len(BLOCKS):(j + 1) * len(BLOCKS)]
        last = np.nonzero(rows.any(0))[0].max()
        assert last <= 9 + 2.98 * j
