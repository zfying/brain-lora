import numpy as np
import pytest
import torch

from brain_lora.model.readouts import CrossAttnReadout, GatedMLPReadout, LinearReadout, load_init

BLOCKS = [(0.5, 2), (2.5, 2), (4.5, 4), (8.5, 8), (16.5, 16), (32.5, 32), (64.5, 64),
          (128.5, 64)]
D, K = 32, 4


@pytest.fixture
def init_file(tmp_path):
    rng = np.random.default_rng(0)
    n_cols = len(BLOCKS) * 3 * K
    path = tmp_path / "init.npz"
    np.savez(path, blocks=np.array(BLOCKS, np.float32), k_use=K,
             subjects=np.array(["sub-01", "sub-02"]), mods=np.array(["text", "audio", "vision"]),
             pca_mean=rng.normal(size=(3, D)).astype(np.float32),
             pca_comps=rng.normal(size=(3, D, K)).astype(np.float32),
             feat_mean=rng.normal(size=n_cols).astype(np.float32),
             W=rng.normal(size=(2, n_cols, 5)).astype(np.float32) * 0.1,
             b=rng.normal(size=(2, 5)).astype(np.float32), report="{}")
    return path


def frames(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {m: torch.randn(n, D, generator=g) for m in ("text", "audio", "vision")}


def test_linear_matches_explicit_computation(init_file):
    m = LinearReadout.from_init(init_file)
    x = frames(298)
    pred = m(x, torch.tensor([0, 1]))
    X = torch.stack([x[n] for n in m.mods], 1)
    p = torch.einsum("fmd,mdk->fmk", X - m.proj_mean, m.proj).reshape(298, -1)
    feat = (m.lag_op @ p).reshape(100, -1) - m.feat_mean
    assert torch.allclose(pred[1], feat @ m.W[1] + m.b[1], atol=1e-5)


def test_window_past_media_is_zero_input(init_file):
    m = LinearReadout.from_init(init_file)
    zeros = {n: torch.zeros(298, D) for n in m.mods}
    subj = torch.tensor([0])
    assert torch.equal(m(None, subj), m(zeros, subj, rel_lo=0, n_take=298, dst_lo=0))


def test_gated_mlp_refit_design_is_exact(init_file):
    torch.manual_seed(0)
    m = GatedMLPReadout.from_init(init_file, hidden=6, gate_init=0.7)
    x, subj = frames(298), torch.tensor([0, 1])
    X = m.refit_design(x)
    mean = torch.cat([m.feat_mean, torch.zeros(6)])
    W = torch.cat([m.W, m.Wx], 1)
    assert torch.allclose(m(x, subj), torch.einsum("td,sdp->stp", X - mean, W)
                          + m.b[:, None], atol=1e-5)


def test_dropping_a_modality_slices_every_block(init_file):
    z = load_init(init_file, mods=("text", "vision"))
    full = load_init(init_file)
    assert z["W"].shape[1] == len(BLOCKS) * 2 * K
    W = full["W"].reshape(2, len(BLOCKS), 3, K, -1)
    assert np.array_equal(z["W"].reshape(2, len(BLOCKS), 2, K, -1), W[:, :, [0, 2]])


def test_cross_attention_starts_at_the_lag_average(init_file):
    """With one token per frame, the initial attention reproduces the lag
    operator, so the linear readout's solution transfers."""
    torch.manual_seed(0)
    lin = LinearReadout.from_init(init_file)
    att = CrossAttnReadout.from_init(init_file)
    x = frames(298)
    streams = dict(x, __phi__={m: torch.arange(298, dtype=torch.float32) for m in x})
    a, b = att.features(streams), lin.features(x)
    rel = (a - b).norm() / (b + lin.feat_mean).norm()
    assert rel < 1e-3


def test_cross_attention_refit_round_trip(init_file):
    att = CrossAttnReadout.from_init(init_file)
    x = frames(298)
    streams = dict(x, __phi__={m: torch.arange(298, dtype=torch.float32) for m in x})
    subj = torch.tensor([0, 1])
    before = att(streams, subj)
    W = att.W.detach().double()
    mx = torch.randn(2, att.out_dim, dtype=torch.float64)
    b = att.b.detach().double() - torch.einsum("sd,sdp->sp", att.feat_mean.double() - mx, W)
    att.install_refit(W, b, mx)
    assert torch.allclose(att(streams, subj), before, atol=1e-5)
