import argparse
import json

import pandas as pd
import pytest

from brain_lora.cli.train import add_config_args
from brain_lora.constants import window_starts
from brain_lora.data.dataset import build_windows
from brain_lora.training import Config


def parse(argv):
    ap = argparse.ArgumentParser()
    add_config_args(ap)
    return Config(**vars(ap.parse_args(["--data-dir", "d", "--readout-init", "i"] + argv)))


def test_defaults_are_the_paper_recipe():
    cfg = parse([])
    assert (cfg.readout, cfg.tap_layer, cfg.rank, cfg.lora_alpha) == ("linear", 32, 16, None)
    assert (cfg.lora_lr, cfg.readout_lr, cfg.proj_lr) == (6e-4, 3e-4, 6e-4)
    assert (cfg.max_steps, cfg.warmup_steps, cfg.context_lead_s, cfg.chunk_s) == \
        (1429, 30, 60.0, 20.0)
    assert cfg.mods == ("text", "audio", "vision")


def test_flags_reach_the_config(tmp_path):
    cfg = parse(["--no-audio", "--lora-lr", "0", "--readout", "cross_attn", "--lora-alpha", "8"])
    assert not cfg.use_audio and cfg.mods == ("text", "vision")
    assert cfg.lora_lr == 0 and cfg.readout == "cross_attn" and cfg.lora_alpha == 8
    cfg.save(tmp_path)
    assert Config.load(tmp_path) == cfg


def test_unknown_readout_is_rejected():
    with pytest.raises(SystemExit):
        parse(["--readout", "mlp"])


def _manifest(tmp_path, duration, n_subjects=2):
    (tmp_path / "friends").mkdir()
    (tmp_path / "friends" / "clip_meta.json").write_text(json.dumps({"duration_s": duration}))
    rows = [dict(media="friends/clip", segment_idx=k, segment_start_s=s, subject_idx=sub,
                 stimulus_id="friends/clip", segment_id=f"sub/clip@seg{k:03d}",
                 fmri_item_id=f"sub-0{sub + 1}/friends/clip")
            for k, s in enumerate(window_starts(duration + 160)) for sub in range(n_subjects)]
    return pd.DataFrame(rows)


def test_window_geometry(tmp_path):
    duration = 400.0                       # 800 frames; windows start at -4.47 + 149 k
    windows = build_windows(_manifest(tmp_path, duration), tmp_path)
    first, second, tail, past = windows[0], windows[1], windows[2], windows[-1]
    # window 0 starts 4.47 s before the stimulus: 9 empty frames, then 289 real ones
    assert (first.dst_lo, first.n_take, first.rel_lo, first.seg_lo) == (9, 289, 0, 0)
    assert (second.dst_lo, second.n_take, second.seg_lo) == (0, 298, 289)
    # the last segment absorbs the clip tail; its window is cut at the clip end
    assert tail.seg_hi == 800 and tail.n_take == 800 - tail.seg_lo - tail.rel_lo
    # a window entirely after the media still carries targets but no frames
    assert past.n_take == 0
    assert all(w.subject_idx == (0, 1) for w in windows)
