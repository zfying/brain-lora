import numpy as np
import torch

from brain_lora.model.inputs import (
    _with_pad_rows,
    gather_token_rows,
    slice_span_words,
    token_keep_positions,
)


def test_words_are_assigned_by_end_time():
    words, onsets, durs = ["a", "b", "c"], [0.5, 9.8, 12.0], [0.2, 0.4, 0.3]
    w, o, d = slice_span_words(words, onsets, durs, 10.0, 20.0)
    assert w == ["b", "c"]                      # "b" ends at 10.2
    assert np.allclose(o, [-0.2, 2.0])


def test_pad_rows_cover_frames_outside_the_media():
    tok = {"pos": torch.tensor([5, 6]), "pool_idx": torch.tensor([0, 1]), "n_tok": 2,
           "phi": torch.tensor([10.0, 11.0]), "valid": torch.ones(2)}
    out = _with_pad_rows(tok, dst_lo=9, n_take=289)
    assert out["n_tok"] == 2 + 9
    assert torch.equal(out["phi"][2:], torch.arange(9, dtype=torch.float32))
    assert not out["valid"][2:].any()


def test_gather_pools_and_zeroes():
    # LM positions 10..13 hold rows 0..3 of the tapped states
    h = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    tokens = {"vision": {"pos": torch.tensor([10, 11, 12, 13, 10]),
                         "pool_idx": torch.tensor([0, 0, 1, 1, 2]), "n_tok": 3,
                         "phi": torch.zeros(3), "valid": torch.tensor([1.0, 1.0, 0.0])}}
    keep = token_keep_positions(tokens)
    assert torch.equal(keep, torch.tensor([10, 11, 12, 13]))
    out = gather_token_rows(h, keep, tokens, n_positions=20)["vision"]
    assert torch.allclose(out[0], h[:2].mean(0))
    assert torch.allclose(out[1], h[2:].mean(0))
    assert not out[2].any()
