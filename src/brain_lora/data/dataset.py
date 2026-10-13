"""Training windows: one item is one 149 s window of one stimulus.

Features do not depend on the subject, so the manifest rows of one window
(four subjects, twice for the Movie10 clips watched twice) share one encode and
the item carries all of their target blocks. Data-loader workers do the CPU
work (mmap'd media slices and the Qwen processor); the model runs in the main
process.
"""

import os
from dataclasses import dataclass

import h5py
import numpy as np
import torch

from brain_lora.constants import (
    MODEL_ID,
    STEP_HZ,
    WINDOW_FRAMES,
    segment_frame_bounds,
    slice_fmri_window,
    window_overlap,
)
from brain_lora.data.media import cache_meta, load_cached_clip
from brain_lora.model.inputs import build_span_inputs, build_token_index


@dataclass
class Window:
    media: str               # "<task>/<stem>" in the media cache
    stimulus_id: str
    segment_id: str
    seg_lo: int              # frames of the encoded segment in the clip's 2 Hz grid
    seg_hi: int
    rel_lo: int              # the window reads segment frames [rel_lo, rel_lo + n_take)
    n_take: int              # into window frames starting at dst_lo;
    dst_lo: int              # n_take == 0: the window lies past the end of the media
    start_s: float           # window start (s), for slicing the BOLD targets
    subject_idx: tuple
    fmri_keys: tuple


def build_windows(df, media_cache):
    """Group manifest rows into windows, in a deterministic order."""
    windows = []
    for (media, seg_idx), grp in df.groupby(["media", "segment_idx"], sort=True):
        row = grp.iloc[0]
        n_out, bounds = segment_frame_bounds(
            float(cache_meta(media_cache, media)["duration_s"]))
        start = float(row["segment_start_s"])
        src_lo, src_hi, dst_lo, _ = window_overlap(int(round(start * STEP_HZ)),
                                                   WINDOW_FRAMES, n_out)
        n_take = max(0, src_hi - src_lo)
        if n_take == 0:
            (seg_lo, seg_hi), rel_lo = bounds[-1], 0
        else:
            owner = [b for b in bounds if b[0] <= src_lo < b[1]]
            if len(owner) != 1 or src_hi > owner[0][1]:
                raise RuntimeError(f"{media}@{seg_idx}: window [{src_lo}, {src_hi}) "
                                   f"is not inside one segment of {bounds}")
            seg_lo, seg_hi = owner[0]
            rel_lo = src_lo - seg_lo
        windows.append(Window(
            media=media, stimulus_id=str(row["stimulus_id"]),
            segment_id=str(row["segment_id"]), seg_lo=seg_lo, seg_hi=seg_hi,
            rel_lo=rel_lo, n_take=n_take, dst_lo=dst_lo, start_s=start,
            subject_idx=tuple(int(s) for s in grp["subject_idx"]),
            fmri_keys=tuple(str(k) for k in grp["fmri_item_id"])))
    return windows


class WindowDataset(torch.utils.data.Dataset):
    """-> dict(inputs, grid, tokens, targets, subject_idx, window geometry).

    ``tokens=True`` also builds the token index the cross-attention readout
    reads. ``fmri_h5=None`` serves inputs without targets.
    """

    def __init__(self, df, media_cache, fmri_h5, *, context_lead_s=60.0,
                 chunk_s=20.0, use_audio=True, tokens=False, processor=None):
        self.media_cache = str(media_cache)
        self.fmri_h5 = None if fmri_h5 is None else str(fmri_h5)
        self.context_lead_s = float(context_lead_s)
        self.chunk_s = float(chunk_s)
        self.use_audio = bool(use_audio)
        self.tokens = bool(tokens)
        self.windows = build_windows(df, media_cache)
        self._proc = processor
        self._h5 = None

    def __len__(self):
        return len(self.windows)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_proc"], state["_h5"] = None, None      # each worker opens its own
        return state

    def _processor(self):
        if self._proc is None:
            from transformers import AutoProcessor
            self._proc = AutoProcessor.from_pretrained(MODEL_ID)
        return self._proc

    def _targets(self, w):
        if self._h5 is None or self._h5[0] != os.getpid():
            self._h5 = (os.getpid(), h5py.File(self.fmri_h5, "r"))
        f = self._h5[1]
        return torch.from_numpy(np.stack([
            slice_fmri_window(np.asarray(f[k], dtype=np.float32), w.start_s)
            for k in w.fmri_keys]))

    def __getitem__(self, i):
        w = self.windows[i]
        inputs = grid = tokens = None
        n_tokens = 0
        if w.n_take:
            meta, frames, audio = load_cached_clip(self.media_cache, w.media)
            t1 = w.seg_hi / STEP_HZ
            t0 = max(0.0, w.seg_lo / STEP_HZ - self.context_lead_s)
            inputs, grid, span = build_span_inputs(
                self._processor(), audio, frames, float(meta["fps"]),
                meta["words"], meta["onsets"], meta["durations"], t0, t1,
                chunk_s=self.chunk_s, use_audio=self.use_audio)
            n_tokens = span["total_tokens"]
            if self.tokens:
                tokens = build_token_index(
                    self._processor(), inputs, span, grid,
                    n_keep=w.seg_hi - w.seg_lo, rel_lo=w.rel_lo,
                    n_take=w.n_take, dst_lo=w.dst_lo)
        return {
            "inputs": inputs, "grid": grid, "tokens": tokens,
            "n_keep": w.seg_hi - w.seg_lo, "rel_lo": w.rel_lo,
            "n_take": w.n_take, "dst_lo": w.dst_lo,
            "targets": None if self.fmri_h5 is None else self._targets(w),
            "subject_idx": torch.tensor(w.subject_idx, dtype=torch.long),
            "segment_id": w.segment_id, "stimulus_id": w.stimulus_id,
            "n_tokens": n_tokens,
        }


def _single(batch):
    return batch[0]


def make_loader(dataset, *, shuffle, seed=0, num_workers=6):
    """One window per step; ``seed`` fixes the shuffle order."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=shuffle, generator=g,
        num_workers=num_workers, collate_fn=_single,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None)
