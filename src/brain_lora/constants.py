"""Clocks, window geometry and identifiers shared by every module.

TR ``i`` of a run sits at stimulus time ``TR * i``. Training window ``k`` covers
stimulus time ``[WINDOW_S * k - HRF_DELAY_S, WINDOW_S * (k + 1) - HRF_DELAY_S)``
and owns TRs ``100k .. 100k + 99``; the fixed hemodynamic delay is baked into
the window origin. Features live on a 2 Hz grid, so a window is 298 frames and
TR ``j`` of a window sits at window frame ``WINDOW_OFFSET_FRAMES + 2.98 j``.
"""

import numpy as np

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05")
N_PARCELS = 1000
MODALITIES = ("text", "audio", "vision")

TR = 1.49                    # fMRI repetition time (s)
STEP_HZ = 2.0                # feature grid (Hz)
HRF_DELAY_S = 4.47           # fixed hemodynamic shift (s)
AUDIO_SR = 16000             # waveform sample rate (Hz)

WINDOW_TRS = 100
WINDOW_S = WINDOW_TRS * TR                          # 149 s
WINDOW_FRAMES = int(WINDOW_S * STEP_HZ)             # 298
WINDOW_OFFSET_FRAMES = 9                            # round(HRF_DELAY_S * STEP_HZ)
FRAMES_PER_TR = TR * STEP_HZ                        # 2.98


def window_starts(stop_s):
    """Start times (s) of the windows tiling a run of length ``stop_s``.

    The first window starts at ``-HRF_DELAY_S``; windows are back to back and the
    last one may extend past the end of the run.
    """
    start, stop = -HRF_DELAY_S, stop_s - HRF_DELAY_S
    if stop <= start:
        return [start]
    return np.arange(start, stop + 1e-8, WINDOW_S).astype(float).tolist()


def segment_frame_bounds(duration_s):
    """Tile a clip's 2 Hz grid into the per-window encoding segments.

    Returns ``(n_frames, [(lo, hi), ...])``. Segment ``k`` holds the frames of
    window ``k`` that lie inside the clip; trailing frames shorter than a window
    are folded into the last segment so the segments cover the whole clip.
    """
    n_out = max(1, int(round(duration_s * STEP_HZ)))
    bounds = []
    for s0 in window_starts(duration_s):
        lo = max(0, int(round(s0 * STEP_HZ)))
        hi = min(n_out, int(round((s0 + WINDOW_S) * STEP_HZ)))
        if hi > lo:
            bounds.append([lo, hi])
    bounds[-1][1] = max(bounds[-1][1], n_out)
    if bounds[0][0] != 0 or any(a[1] != b[0] for a, b in zip(bounds, bounds[1:])):
        raise AssertionError(f"segments do not tile [0, {n_out}): {bounds}")
    return n_out, [tuple(b) for b in bounds]


def window_overlap(start_idx, n_out, src_len):
    """Overlap of a length-``n_out`` window starting at ``start_idx`` with a
    source of length ``src_len``: ``(src_lo, src_hi, dst_lo, dst_hi)``."""
    src_lo = max(0, start_idx)
    src_hi = min(src_len, start_idx + n_out)
    dst_lo = max(0, -start_idx)
    dst_hi = dst_lo + max(0, src_hi - src_lo)
    return src_lo, src_hi, dst_lo, dst_hi


def slice_fmri_window(fmri, window_start_s):
    """The 100-TR target block of the window starting at ``window_start_s``,
    zero-padded where the window extends past the run."""
    out = np.zeros((WINDOW_TRS, fmri.shape[1]), dtype=fmri.dtype)
    start_idx = int(round((window_start_s + HRF_DELAY_S) / TR))
    src_lo, src_hi, dst_lo, dst_hi = window_overlap(start_idx, WINDOW_TRS,
                                                    fmri.shape[0])
    if src_hi > src_lo:
        out[dst_lo:dst_hi] = fmri[src_lo:src_hi]
    return out
