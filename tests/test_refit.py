import numpy as np
import torch

from brain_lora.metrics import pearson_per_parcel, pearson_report
from brain_lora.model.refit import (
    RidgeAccumulator,
    fit_per_parcel_alpha,
    holdout_indices,
    pearson_columns,
    solve_ridge,
)


def _data(n_items=40, T=50, D=8, P=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(D, P, generator=g, dtype=torch.float64)
    for i in range(n_items):
        X = torch.randn(T, D, generator=g, dtype=torch.float64) + 1.0
        Y = torch.stack([X @ W + torch.randn(T, P, generator=g, dtype=torch.float64)
                         for _ in range(2)])
        yield i, X, Y, torch.tensor([0, 1])


def test_single_penalty_matches_direct_ridge():
    fit, held = RidgeAccumulator(8, 6, 2), RidgeAccumulator(8, 6, 2)
    Xs, Ys = [], []
    for i, X, Y, s in _data():
        (held if i % 4 == 0 else fit).add(X, Y, s)
        Xs.append(X)
        Ys.append(Y[0])
    held_X = torch.cat([X for i, X, _, _ in _data() if i % 4 == 0])
    rows = [torch.arange(len(held_X))] * 2
    held_Y = [torch.cat([Y[s] for i, _, Y, _ in _data() if i % 4 == 0]) for s in range(2)]
    W, b, mx, picks = fit_per_parcel_alpha(fit, held, held_X, rows, held_Y, alphas=(3.0,))
    X, Y = torch.cat(Xs), torch.cat(Ys)
    Xc = X - X.mean(0)
    direct = torch.linalg.solve(Xc.T @ Xc + 3.0 * torch.eye(8, dtype=torch.float64),
                                Xc.T @ (Y - Y.mean(0)))
    assert torch.allclose(W[0], direct, atol=1e-8)
    assert torch.allclose(mx[0], X.mean(0)) and torch.allclose(b[0], Y.mean(0))
    assert (picks == 0).all()


def test_repeated_observations_count_twice():
    acc = RidgeAccumulator(3, 2, 1)
    X, Y = torch.randn(4, 3).double(), torch.randn(2, 4, 2).double()
    acc.add(X, Y, torch.tensor([0, 0]))
    assert acc.rows[0] == 8
    assert torch.allclose(acc.XtX[0], 2 * X.T @ X)


def test_solve_ridge_does_not_mutate():
    S = torch.eye(3, dtype=torch.float64)
    solve_ridge(S, torch.ones(3, 1, dtype=torch.float64), 5.0)
    assert torch.equal(S, torch.eye(3, dtype=torch.float64))


def test_holdout_is_deterministic():
    assert holdout_indices(1429) == holdout_indices(1429)
    assert len(holdout_indices(1429)) == round(1429 * 0.15)


def test_pearson_implementations_agree():
    rng = np.random.default_rng(0)
    p, t = rng.normal(size=(200, 7)), rng.normal(size=(200, 7))
    t[:, 3] = 1.0
    ref = pearson_per_parcel(p, t)
    fast = pearson_columns(torch.from_numpy(p), torch.from_numpy(t)).numpy()
    assert np.isnan(ref[3]) and np.isnan(fast[3])
    assert np.allclose(np.delete(ref, 3), np.delete(fast, 3), atol=1e-6)


def test_subject_mean_is_mean_of_subject_means():
    rng = np.random.default_rng(1)
    p, t = rng.normal(size=(300, 5)), rng.normal(size=(300, 5))
    subj = np.repeat([0, 1, 2], 100)
    rep = pearson_report(p, t, subj)
    assert np.isclose(rep["subject_mean"], np.mean(list(rep["per_subject"].values())))
    assert set(rep["per_subject"]) == {"sub-01", "sub-02", "sub-03"}
