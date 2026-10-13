"""Stimulus decoding and the memory-mapped media cache.

Every clip is decoded once: 16 kHz mono audio, frames sampled at 2 Hz and
resized to 336x224, and the word-level transcript. Training reads mmap'd
slices of the cache, so no video decoding happens in the data loader.

Cache layout, per clip::

    <cache>/<task>/<stem>_frames.npy   (n_frames, 224, 336, 3) uint8
    <cache>/<task>/<stem>_audio.npy    (n_samples,) float32
    <cache>/<task>/<stem>_meta.json    fps, duration_s, words, onsets, durations
"""

import ast
import json
from pathlib import Path

import numpy as np

from brain_lora.constants import AUDIO_SR, STEP_HZ

FRAME_SIZE = (336, 224)       # (width, height)


def load_transcript(path):
    """Algonauts transcript TSV -> (words, onsets, durations)."""
    import pandas as pd

    words, onsets, durations = [], [], []
    if not path or not Path(path).exists():
        return words, onsets, durations
    df = pd.read_csv(path, sep="\t")
    for _, row in df.iterrows():
        try:
            w = ast.literal_eval(row.get("words_per_tr", "[]"))
            o = ast.literal_eval(row.get("onsets_per_tr", "[]"))
            d = ast.literal_eval(row.get("durations_per_tr", "[]"))
        except (SyntaxError, ValueError):
            continue
        for word, onset, dur in zip(w, o, d):
            if word:
                words.append(str(word))
                onsets.append(float(onset))
                durations.append(float(dur))
    return words, onsets, durations


def _video_clip(path, **kw):
    from moviepy import VideoFileClip  # moviepy 2.x

    return VideoFileClip(str(path), **kw)


def decode_audio(path):
    """Mono float32 waveform at 16 kHz."""
    import julius
    import torch

    clip = _video_clip(path)
    native_sr = float(clip.audio.fps)
    wav = clip.audio.to_soundarray()
    clip.close()
    wav = (wav.mean(axis=1) if wav.ndim == 2 else wav).astype(np.float32)
    if int(round(native_sr)) == AUDIO_SR:
        return wav
    resample = julius.resample.ResampleFrac(old_sr=int(round(native_sr)),
                                            new_sr=AUDIO_SR)
    return resample(torch.from_numpy(wav)[None])[0].numpy().astype(np.float32)


def decode_frames(path, fps=STEP_HZ, frame_size=FRAME_SIZE):
    """Frames sampled at ``fps`` and resized to ``frame_size`` (INTER_AREA).

    Returns ``(frames, fps, duration_s)``.
    """
    import cv2

    clip = _video_clip(path, audio=False)
    duration = float(clip.duration or 0.0)
    n = max(1, int(np.ceil(duration * fps)))
    times = np.arange(n, dtype=np.float32) / np.float32(fps)
    last = max(0.0, np.nextafter(duration, 0.0))
    frames = [np.asarray(clip.get_frame(max(0.0, min(last, float(t)))),
                         dtype=np.uint8) for t in times]
    clip.close()
    w, h = frame_size
    if frames[0].shape[:2] != (h, w):
        frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
                  for f in frames]
    return np.stack(frames), float(fps), duration


def build_cache_entry(video, transcript, out_dir):
    """Decode one clip into ``out_dir``. Idempotent: ``meta.json`` is written
    last and marks a finished entry."""
    out_dir = Path(out_dir)
    stem = Path(video).stem
    meta_path = out_dir / f"{stem}_meta.json"
    if meta_path.exists():
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = decode_audio(video)
    frames, fps, duration = decode_frames(video)
    words, onsets, durations = load_transcript(transcript)
    keep = [i for i, o in enumerate(onsets) if o < duration]
    np.save(out_dir / f"{stem}_frames.npy", np.ascontiguousarray(frames))
    np.save(out_dir / f"{stem}_audio.npy", audio)
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "stem": stem, "fps": fps, "duration_s": duration,
        "n_frames": int(len(frames)), "n_audio": int(len(audio)),
        "words": [words[i] for i in keep],
        "onsets": [onsets[i] for i in keep],
        "durations": [durations[i] for i in keep],
    }))
    tmp.rename(meta_path)
    return True


def cache_meta(cache_dir, media):
    """The ``meta.json`` of ``media`` (``"<task>/<stem>"``)."""
    return json.loads((Path(cache_dir) / f"{media}_meta.json").read_text())


def load_cached_clip(cache_dir, media):
    """-> (meta, frames, audio) with the arrays memory-mapped."""
    stem = Path(cache_dir) / media
    meta = json.loads(Path(f"{stem}_meta.json").read_text())
    frames = np.load(f"{stem}_frames.npy", mmap_mode="r")
    audio = np.load(f"{stem}_audio.npy", mmap_mode="r")
    return meta, frames, audio
