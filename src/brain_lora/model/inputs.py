"""Single-pass causal input layout for Qwen3-Omni.

A span of stimulus is cut into ``chunk_s`` chunks. Each chunk contributes one
native audio-in-video block (time-interleaved audio and video tokens) followed
by the words whose end time falls in that chunk::

    <|im_start|>user\\n
    [vision_bos audio_bos <A/V of chunk 0> audio_eos vision_eos] words(chunk 0)
    [vision_bos audio_bos <A/V of chunk 1> audio_eos vision_eos] words(chunk 1)
    ...
    <|im_end|>\\n

Every token carries the stimulus time it describes, so one causal forward pass
yields a representation of each stream at every 2 Hz step. This replaces the
sliding window of one forward pass per fMRI timestep.

Two readout contracts are built here:

* the 2 Hz frame grid (``grid_readout``): for each step, the last token of each
  stream at or before it -- what the linear readout consumes;
* the token index (``build_token_index``): every audio token (13 Hz), every
  word, and one mean-pooled vision vector per 1 s video step, each with its
  window-frame coordinate -- what the cross-attention readout consumes.
"""

import logging

import numpy as np
import torch
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    _get_feat_extract_output_lengths,
)

from brain_lora.constants import AUDIO_SR, STEP_HZ, WINDOW_FRAMES

AUDIO_TOK_HZ = 13.0
POSITION_ID_PER_SECONDS = 13.0
SECONDS_PER_CHUNK = 2.0
VIDEO_SIZE = {"shortest_edge": 224 * 336, "longest_edge": 768 * 32 * 32}

log = logging.getLogger(__name__)


def slice_span_media(audio, frames, fps, t0, t1):
    """Audio samples and video frames of the stimulus span ``[t0, t1)``."""
    a = None if audio is None else audio[int(round(t0 * AUDIO_SR)):
                                         int(round(t1 * AUDIO_SR))]
    f0 = min(int(round(t0 * fps)), max(0, len(frames) - 1))
    f1 = min(max(f0 + 1, int(round(t1 * fps))), len(frames))
    return a, frames[f0:f1]


def slice_span_words(words, onsets, durations, t0, t1):
    """Words whose end time lies in ``(t0, t1]``, onsets made span-relative."""
    ends = [o + max(d, 0.0) for o, d in zip(onsets, durations)]
    keep = [i for i, c in enumerate(ends) if t0 < c <= t1]
    return ([words[i] for i in keep], [onsets[i] - t0 for i in keep],
            [durations[i] for i in keep])


def _chunk_bounds(duration_s, chunk_s):
    edges = list(np.arange(0.0, duration_s, chunk_s)) + [duration_s]
    return [(float(a), float(b)) for a, b in zip(edges, edges[1:]) if b - a > 1e-3]


def build_clip_inputs(proc, audio, frames, fps, words, onsets, durations,
                      duration_s, *, chunk_s=20.0, use_audio=True):
    """Assemble the model input for one span plus per-token stimulus times.

    ``use_audio=False`` removes the audio stream from the input entirely (no
    mel features, no audio placeholders); vision and text keep their times.
    """
    fe, vp = proc.feature_extractor, proc.video_processor
    merge = vp.merge_size
    bounds = _chunk_bounds(duration_s, chunk_s)
    word_end = np.array([o + max(d, 0.0) for o, d in zip(onsets, durations)],
                        dtype=np.float64)

    mels, audio_lens, audio_times = [], [], []
    pix, grids, video_times = [], [], []
    text_pieces, chunk_words = [], []
    # the STFT reflect-pads by n_fft // 2, so a chunk needs more samples than that
    min_audio = fe.n_fft // 2 + 1
    for t0, t1 in bounds:
        if use_audio:
            a = audio[int(round(t0 * AUDIO_SR)):int(round(t1 * AUDIO_SR))]
            if a.size < min_audio:
                a = np.pad(np.asarray(a, dtype=np.float32), (0, min_audio - a.size))
            mel = fe(a, sampling_rate=AUDIO_SR, return_attention_mask=True,
                     padding=False, truncation=False,
                     return_tensors="pt")["input_features"][0]
            mels.append(mel)
            n_a = int(_get_feat_extract_output_lengths(torch.tensor([mel.shape[-1]]))[0])
            audio_lens.append(n_a)
            audio_times.extend(t0 + (np.arange(n_a) + 1) / AUDIO_TOK_HZ)

        i0 = min(int(round(t0 * fps)), max(0, len(frames) - 1))
        i1 = min(max(i0 + 1, int(round(t1 * fps))), len(frames))
        sl = frames[i0:i1] if i1 > i0 else frames[-1:]
        vi = vp(videos=[list(sl)], size=VIDEO_SIZE, return_tensors="pt")
        g = vi["video_grid_thw"][0]
        pix.append(vi["pixel_values_videos"])
        grids.append(g)
        sec_per_grid = vp.temporal_patch_size / fps
        per_step = (int(g[1]) // merge) * (int(g[2]) // merge)
        for s in range(int(g[0])):
            video_times.extend([t0 + (s + 1) * sec_per_grid] * per_step)

        sel = np.where((word_end > t0) & (word_end <= t1))[0]
        text_pieces.append(" ".join(words[i] for i in sel))
        chunk_words.append(sel)

    n_blocks = len(bounds)
    sec_per_grid = vp.temporal_patch_size / fps
    template = "<|im_start|>user\n"
    for piece in text_pieces:
        template += proc.vision_bos_token + proc.video_token + proc.vision_eos_token
        template += piece
    template += "<|im_end|>\n"
    expanded = proc.replace_multimodal_special_tokens(
        [template], iter(audio_lens), iter([]), iter(grids),
        video_second_per_grid=iter([sec_per_grid] * n_blocks),
        use_audio_in_video=use_audio,
        position_id_per_seconds=POSITION_ID_PER_SECONDS,
        seconds_per_chunk=SECONDS_PER_CHUNK)
    tok = proc.tokenizer(expanded, return_tensors="pt", padding=False,
                         return_offsets_mapping=True)
    text_pos, text_times = _locate_words(expanded[0], tok, text_pieces,
                                         chunk_words, words, word_end)

    inputs = {"input_ids": tok["input_ids"],
              "attention_mask": tok["attention_mask"],
              "pixel_values_videos": torch.cat(pix, dim=0),
              "video_grid_thw": torch.stack(grids, dim=0),
              "video_second_per_grid": torch.tensor([sec_per_grid] * n_blocks)}
    if use_audio:
        max_mel = max(m.shape[-1] for m in mels)
        feats = torch.zeros(n_blocks, mels[0].shape[0], max_mel, dtype=mels[0].dtype)
        mask = torch.zeros(n_blocks, max_mel, dtype=torch.long)
        for i, m in enumerate(mels):
            feats[i, :, :m.shape[-1]] = m
            mask[i, :m.shape[-1]] = 1
        inputs["input_features"] = feats
        inputs["feature_attention_mask"] = mask
    meta = {
        "duration_s": duration_s,
        "total_tokens": int(tok["input_ids"].shape[1]),
        "audio_times": np.asarray(audio_times, dtype=np.float64),
        "video_times": np.asarray(video_times, dtype=np.float64),
        "text_pos": np.asarray(text_pos, dtype=np.int64),
        "text_times": np.asarray(text_times, dtype=np.float64),
    }
    return inputs, meta


def _locate_words(prompt, tok, pieces, chunk_words, words, word_end):
    """Token position and end time of every word found in the prompt.

    Each text piece appears verbatim after its media block, so it is found by
    a forward scan and its words are mapped to tokens via the offset mapping
    (special tokens are excluded from the lookup).
    """
    off = tok["offset_mapping"][0].numpy()
    tok_start, tok_stop = off[:, 0].copy(), off[:, 1]
    tok_start[tok_stop <= tok_start] = np.iinfo(np.int64).max
    tok_start = np.minimum.accumulate(tok_start[::-1])[::-1]
    pos, times, cursor = [], [], 0
    for i, piece in enumerate(pieces):
        if not piece:
            continue
        start = prompt.find(piece, cursor)
        if start < 0:
            log.warning("text piece %d not found in the prompt; skipped", i)
            continue
        cursor = start + len(piece)
        local = 0
        for w_idx in chunk_words[i]:
            w = words[w_idx]
            ws = prompt.find(w, start + local, cursor)
            if ws < 0:
                continue
            local = ws - start + len(w)
            j = int(np.searchsorted(tok_start, ws + len(w), side="left")) - 1
            if j >= 0 and tok_stop[j] > ws:
                pos.append(j)
                times.append(float(word_end[w_idx]))
    return pos, times


def stream_positions(proc, input_ids):
    """LM positions of the audio and video placeholder tokens."""
    ids = input_ids[0]
    tk = proc.tokenizer
    return {"audio": (ids == tk.convert_tokens_to_ids(proc.audio_token)).nonzero(as_tuple=True)[0],
            "vision": (ids == tk.convert_tokens_to_ids(proc.video_token)).nonzero(as_tuple=True)[0]}


def grid_readout(pos, meta):
    """For each 2 Hz step, the last token of each stream at or before it.

    ``valid[name]`` is False on steps before the stream's first token; those
    steps are zeroed downstream rather than reading a future token.
    """
    dur = meta["duration_s"]
    n_out = max(1, int(round(dur * STEP_HZ)))
    t_grid = np.minimum((np.arange(n_out) + 1) / STEP_HZ, dur)
    out = {"n_out": n_out, "valid": {}}
    streams = (("audio", meta["audio_times"], pos["audio"]),
               ("vision", meta["video_times"], pos["vision"]),
               ("text", meta["text_times"], torch.from_numpy(meta["text_pos"])))
    for name, times, tok_pos in streams:
        if len(times) == 0:
            out[name] = None
            out["valid"][name] = np.zeros(n_out, dtype=bool)
            continue
        k = np.searchsorted(times, t_grid, side="right") - 1
        out["valid"][name] = k >= 0
        out[name] = tok_pos[torch.from_numpy(np.clip(k, 0, len(times) - 1))]
    return out


def build_span_inputs(proc, audio, frames, fps, words, onsets, durations,
                      t0, t1, *, chunk_s=20.0, use_audio=True):
    """Media and transcript of the span ``[t0, t1)`` -> (inputs, grid, meta)."""
    a, fr = slice_span_media(audio, frames, fps, t0, t1)
    w, o, d = slice_span_words(words, onsets, durations, t0, t1)
    inputs, meta = build_clip_inputs(proc, a, fr, fps, w, o, d, t1 - t0,
                                     chunk_s=chunk_s, use_audio=use_audio)
    grid = grid_readout(stream_positions(proc, inputs["input_ids"]), meta)
    return inputs, grid, meta


# ---------------------------------------------------------------------------
# token index for the cross-attention readout
# ---------------------------------------------------------------------------
def build_token_index(proc, inputs, meta, grid, *, n_keep, rel_lo, n_take, dst_lo,
                      mods=("text", "audio", "vision")):
    """Per-stream token index: ``{name: {"pos", "pool_idx", "n_tok", "phi", "valid"}}``.

    ``pos`` are LM positions to gather, ``pool_idx`` maps each gathered position
    to an output token (vision tokens of one 1 s step are mean-pooled), ``phi``
    is each output token's window-frame coordinate (negative for the context
    lead), and ``valid`` zeroes padding tokens.
    """
    n_out = int(grid["n_out"])
    pos_by_stream = stream_positions(proc, inputs["input_ids"])
    raw = {"audio": (pos_by_stream["audio"], meta["audio_times"], False),
           "vision": (pos_by_stream["vision"], meta["video_times"], True),
           "text": (torch.from_numpy(meta["text_pos"]), meta["text_times"], False)}
    out = {}
    for name in mods:
        positions, times, pool = raw[name]
        if len(times) == 0:
            out[name] = _empty_tokens()
            continue
        positions = np.asarray(positions.cpu(), dtype=np.int64)
        if len(positions) != len(times):
            raise ValueError(f"{name}: {len(positions)} positions vs {len(times)} times")
        # span time (s) -> segment frame (the segment is the last n_keep steps
        # of the span) -> window frame; tokens past the window are dropped
        seg_f = np.asarray(times, dtype=np.float64) * STEP_HZ - (n_out - n_keep + 1)
        sel = np.nonzero(seg_f < rel_lo + n_take)[0]
        if len(sel) == 0:
            out[name] = _empty_tokens()
            continue
        positions, phi = positions[sel], (dst_lo + seg_f - rel_lo)[sel]
        if pool:
            # all tokens of one video step share its time, so times are groups
            _, pool_idx = np.unique(phi, return_inverse=True)
            n_tok = int(pool_idx.max()) + 1
            phi_tok = np.zeros(n_tok, dtype=np.float64)
            phi_tok[pool_idx] = phi
        else:
            pool_idx = np.arange(len(sel), dtype=np.int64)
            n_tok, phi_tok = len(sel), phi
        out[name] = _with_pad_rows({
            "pos": torch.from_numpy(np.ascontiguousarray(positions)).long(),
            "pool_idx": torch.from_numpy(np.ascontiguousarray(pool_idx)).long(),
            "n_tok": int(n_tok),
            "phi": torch.from_numpy(phi_tok.astype(np.float32)),
            "valid": torch.ones(int(n_tok), dtype=torch.float32),
        }, dst_lo, n_take)
    return out


def _empty_tokens():
    return {"pos": torch.zeros(0, dtype=torch.long),
            "pool_idx": torch.zeros(0, dtype=torch.long), "n_tok": 0,
            "phi": torch.zeros(0, dtype=torch.float32),
            "valid": torch.zeros(0, dtype=torch.float32)}


def _with_pad_rows(tok, dst_lo, n_take):
    """Append one zero-valued token per window frame outside the media.

    The linear readout pads such frames with raw zeros, which the lag operator
    averages in; a padding token at the same frame keeps the cross-attention
    readout's lag-equivalent initialisation exact on those windows.
    """
    lo, hi = int(dst_lo), int(dst_lo) + int(n_take or 0)
    missing = np.array([r for r in range(WINDOW_FRAMES) if not lo <= r < hi],
                       dtype=np.int64)
    if not len(missing):
        return tok
    n = int(tok["n_tok"])
    return {
        "pos": torch.cat([tok["pos"], torch.zeros(len(missing), dtype=torch.long)]),
        "pool_idx": torch.cat([tok["pool_idx"],
                               torch.arange(n, n + len(missing), dtype=torch.long)]),
        "n_tok": n + len(missing),
        "phi": torch.cat([tok["phi"], torch.from_numpy(missing.astype(np.float32))]),
        "valid": torch.cat([tok["valid"], torch.zeros(len(missing))]),
    }


def token_keep_positions(tokens):
    """Sorted unique LM positions named by a token index."""
    parts = [t["pos"] for t in tokens.values() if int(t["n_tok"]) > 0]
    return torch.cat(parts).unique().sort().values


def gather_token_rows(h, keep, tokens, n_positions):
    """Tapped hidden states ``h`` (rows aligned with ``keep``) -> per-stream
    tokens ``{name: (n_tok, D)}``, mean-pooled and padding-zeroed in graph."""
    lut = torch.full((int(n_positions),), -1, dtype=torch.long)
    lut[keep.cpu()] = torch.arange(len(keep))
    out = {}
    for name, tok in tokens.items():
        n_tok = int(tok["n_tok"])
        if n_tok == 0:
            out[name] = h.new_zeros((0, h.shape[-1]))
            continue
        rows = lut[tok["pos"]]
        if bool((rows < 0).any()):
            raise RuntimeError(f"{name}: token positions missing from the tap")
        g = h.index_select(0, rows.to(h.device))
        pool = tok["pool_idx"].to(h.device)
        if len(pool) == n_tok and bool(torch.equal(pool, torch.arange(n_tok, device=pool.device))):
            x = g
        else:
            acc = g.new_zeros((n_tok, g.shape[-1])).index_add(0, pool, g)
            cnt = g.new_zeros(n_tok).index_add(0, pool, torch.ones_like(pool, dtype=g.dtype))
            x = acc / cnt.clamp(min=1)[:, None]
        out[name] = x * tok["valid"].to(x.device, x.dtype)[:, None]
    return out
