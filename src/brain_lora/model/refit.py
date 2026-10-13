"""Closed-form ridge solve of a readout's per-subject linear map.

Design rows are streamed once into per-subject moments (float64); the penalty
is chosen per parcel on held-out training windows and the winning penalty is
re-solved on all training windows. Pure math, no model: testable on CPU.

Grams are kept per subject because a stimulus may be observed more than once
by the same subject (Movie10 runs 1 and 2), and each observation counts.
"""

import numpy as np
import torch

ALPHAS = (1e3, 1e4, 1e5, 1e6, 1e7)


class RidgeAccumulator:
    """Streaming X'X, X'Y and column sums, one set per subject."""

    def __init__(self, dim, n_parcels, n_subjects, device="cpu"):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)  # noqa: E731
        self.device = torch.device(device)
        self.XtX = [z(dim, dim) for _ in range(n_subjects)]
        self.XtY = [z(dim, n_parcels) for _ in range(n_subjects)]
        self.sx = [z(dim) for _ in range(n_subjects)]
        self.sy = [z(n_parcels) for _ in range(n_subjects)]
        self.rows = [0] * n_subjects

    def add(self, X, targets, subject_idx):
        """X (T, D) design rows; targets (S, T, P) aligned with subject_idx (S,)."""
        x = X.to(self.device, torch.float64)
        gram, col = x.T @ x, x.sum(0)
        for row, s in enumerate(subject_idx.tolist()):
            y = targets[row].to(self.device, torch.float64)
            self.XtX[s] += gram
            self.sx[s] += col
            self.XtY[s] += x.T @ y
            self.sy[s] += y.sum(0)
            self.rows[s] += x.shape[0]


def solve_ridge(Sxx, Sxy, alpha):
    """(Sxx + alpha I)^-1 Sxy for centred moments."""
    A = Sxx.clone()
    A.diagonal().add_(alpha)
    return torch.linalg.solve(A, Sxy)


def pearson_columns(pred, true):
    """Per-column Pearson r of two (T, P) tensors; NaN where a column is flat."""
    p, t = pred - pred.mean(0), true - true.mean(0)
    den = p.norm(dim=0) * t.norm(dim=0)
    return torch.where(den > 0, (p * t).sum(0) / den.clamp(min=1e-12),
                       torch.full_like(den, float("nan")))


def fit_per_parcel_alpha(fit, held, held_X, held_rows, held_Y, alphas=ALPHAS):
    """Ridge with a per-parcel penalty chosen on held-out rows.

    ``fit`` / ``held`` are accumulators over disjoint windows. ``held_X`` stacks
    the held-out design rows; ``held_rows[s]`` indexes the rows subject ``s``
    observed and ``held_Y[s]`` holds the matching targets. Each parcel's penalty
    maximises held-out Pearson r of a fit on ``fit`` alone; the solution is then
    recomputed on ``fit`` and ``held`` together.

    Returns ``W (S, D, P)``, ``b (S, P)``, ``mx (S, D)`` (the design mean the
    solution is centred at) and the chosen penalty index per parcel ``(S, P)``.
    """
    Ws, bs, mxs, picks = [], [], [], []
    for s in range(len(fit.rows)):
        n_f, n_v = fit.rows[s], held.rows[s]
        mx_f, my_f = fit.sx[s] / n_f, fit.sy[s] / n_f
        Sxx_f = fit.XtX[s] - n_f * torch.outer(mx_f, mx_f)
        Sxy_f = fit.XtY[s] - n_f * torch.outer(mx_f, my_f)
        Xv = held_X.index_select(0, held_rows[s].to(held_X.device)).to(
            Sxx_f.device, Sxx_f.dtype) - mx_f
        Yv = held_Y[s].to(Sxx_f.device, Sxx_f.dtype)
        scores = [torch.nan_to_num(pearson_columns(Xv @ solve_ridge(Sxx_f, Sxy_f, a)
                                                   + my_f, Yv), nan=-1.0)
                  for a in alphas]
        pick = torch.stack(scores).argmax(0)
        del scores, Xv, Yv, Sxx_f, Sxy_f

        n = n_f + n_v
        mx_v, my_v = held.sx[s] / n_v, held.sy[s] / n_v
        mx = (mx_f * n_f + mx_v * n_v) / n
        my = (my_f * n_f + my_v * n_v) / n
        Sxx = fit.XtX[s] + held.XtX[s] - n * torch.outer(mx, mx)
        Sxy = fit.XtY[s] + held.XtY[s] - n * torch.outer(mx, my)
        W = torch.zeros_like(Sxy)
        for ai, a in enumerate(alphas):
            cols = (pick == ai).nonzero(as_tuple=True)[0]
            if cols.numel():
                W[:, cols] = solve_ridge(Sxx, Sxy, a)[:, cols]
        del Sxx, Sxy
        Ws.append(W)
        bs.append(my)
        mxs.append(mx)
        picks.append(pick)
    return torch.stack(Ws), torch.stack(bs), torch.stack(mxs), torch.stack(picks)


def holdout_indices(n_items, frac=0.15, seed=0):
    """Deterministic held-out item indices for penalty selection."""
    perm = np.random.RandomState(seed).permutation(n_items)
    return set(int(i) for i in perm[:max(1, int(round(n_items * frac)))])
