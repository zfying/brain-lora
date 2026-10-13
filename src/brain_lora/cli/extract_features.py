"""Extract frozen-backbone features at one layer for every labelled clip.

    python -m brain_lora.cli.extract_features --data-dir data \\
        --out data/features_L32_lead30.h5 --context-lead-s 30

Each clip is encoded window by window exactly as during training (one causal
pass per 149 s window, preceded by ``--context-lead-s`` of context), with no
adapter. The store holds one ``(n_frames, 3, 2048)`` float16 array per clip,
modalities in the order text, audio, vision, on the 2 Hz grid. It is the input
of ``brain_lora.cli.fit_readout_init``. Re-running resumes.
"""

import argparse
import logging

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor

from brain_lora.constants import MODALITIES, MODEL_ID, STEP_HZ, segment_frame_bounds
from brain_lora.data.algonauts import load_manifests
from brain_lora.data.media import cache_meta, load_cached_clip
from brain_lora.model.inputs import build_span_inputs
from brain_lora.training import data_paths

log = logging.getLogger("extract")


class Segments(torch.utils.data.Dataset):
    """Model inputs of every encoding segment of the given clips, in order."""

    def __init__(self, media_cache, clips, context_lead_s, chunk_s):
        self.media_cache, self.lead, self.chunk_s = media_cache, context_lead_s, chunk_s
        self.items = []
        for media in clips:
            n_out, bounds = segment_frame_bounds(
                float(cache_meta(media_cache, media)["duration_s"]))
            self.items += [(media, n_out, lo, hi, k == len(bounds) - 1)
                           for k, (lo, hi) in enumerate(bounds)]
        self.proc = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.proc is None:
            self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        media, n_out, lo, hi, last = self.items[i]
        meta, frames, audio = load_cached_clip(self.media_cache, media)
        inputs, grid, _ = build_span_inputs(
            self.proc, audio, frames, float(meta["fps"]), meta["words"],
            meta["onsets"], meta["durations"], max(0.0, lo / STEP_HZ - self.lead),
            hi / STEP_HZ, chunk_s=self.chunk_s)
        return media, n_out, lo, hi, last, inputs, grid


def _as_is(item):
    return item


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tap-layer", type=int, default=32)
    ap.add_argument("--context-lead-s", type=float, default=30.0)
    ap.add_argument("--chunk-s", type=float, default=20.0)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")

    from brain_lora.model.encoder import OmniEncoder

    paths = data_paths(args.data_dir)
    clips = sorted(pd.concat(load_manifests(paths["manifests"]))["media"].unique())
    with h5py.File(args.out, "a") as f:
        pending = [c for c in clips if c not in f]
        f.attrs.update(tap_layer=args.tap_layer, context_lead_s=args.context_lead_s,
                       chunk_s=args.chunk_s, modalities=list(MODALITIES))
    log.info("%d clips, %d to extract", len(clips), len(pending))
    encoder = OmniEncoder(tap_layer=args.tap_layer, rank=None, device=args.device)
    encoder.eval()
    loader = torch.utils.data.DataLoader(
        Segments(paths["media_cache"], pending, args.context_lead_s, args.chunk_s),
        batch_size=None, shuffle=False, num_workers=args.num_workers,
        collate_fn=_as_is)

    buf = None
    for media, n_out, lo, hi, last, inputs, grid in loader:
        if grid["n_out"] < hi - lo:
            raise RuntimeError(f"{media}: span of {grid['n_out']} steps < {hi - lo}")
        with torch.no_grad():
            feats = encoder.encode(inputs, grid, hi - lo)
        encoder.tap.release()
        if buf is None:
            dim = feats["text"].shape[-1]
            buf = np.zeros((n_out, len(MODALITIES), dim), dtype=np.float16)
        for m, name in enumerate(MODALITIES):
            buf[lo:hi, m] = feats[name].to(torch.float16).cpu().numpy()
        if last:
            with h5py.File(args.out, "a") as f:
                f.create_dataset(media, data=buf)
            log.info("%s: %d frames", media, n_out)
            buf = None


if __name__ == "__main__":
    main()
