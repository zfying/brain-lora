"""Fit the readout initialisation on frozen features.

    python -m brain_lora.cli.fit_readout_init --data-dir data \\
        --features data/features_L32_lead30.h5 --out data/readout_init_L32.npz

1. A PCA basis per modality (2048 -> 512) from a random sample of training
   frames.
2. The closed-form ridge solution of the linear readout on the projected
   features, built with the same windows, padding and lag operator as the
   trained readout, so it initialises ``LinearReadout`` exactly. The penalty is
   chosen per parcel on 15% of the training runs.

The validation (season 6) score of this frozen ridge readout is printed and
stored in the file's ``report``.
"""

import argparse
import json
import logging

import h5py
import numpy as np
import torch

from brain_lora.constants import (
    MODALITIES,
    SUBJECTS,
    WINDOW_FRAMES,
    WINDOW_OFFSET_FRAMES,
    WINDOW_TRS,
)
from brain_lora.data.algonauts import load_manifests
from brain_lora.model.readouts import build_lag_operator
from brain_lora.model.refit import ALPHAS, RidgeAccumulator, fit_per_parcel_alpha, pearson_columns
from brain_lora.training import data_paths

log = logging.getLogger("fit_readout_init")

#: (delay, width) in seconds of the eight log-spaced lag blocks
LAG_BLOCKS = [(0.5, 2), (2.5, 2), (4.5, 4), (8.5, 8), (16.5, 16), (32.5, 32),
              (64.5, 64), (128.5, 64)]


def runs_by_split(manifests):
    """{split: [(stimulus_id, media)]}, Friends before Movie10, each sorted."""
    out = {}
    for split, df in zip(("train", "val"), manifests):
        runs = df[["task", "stimulus_id", "media"]].drop_duplicates()
        out[split] = [(r.stimulus_id, r.media) for r in
                      runs.sort_values(["task", "stimulus_id"]).itertuples()]
    return out


def fit_pca(store, runs, k, n_sample, seed):
    """Per-modality PCA of a frame sample of ``runs`` -> (mean (M, D), comps (M, D, k))."""
    rng = np.random.RandomState(seed)
    per_run = max(1, n_sample // len(runs))
    means, comps = [], []
    for m, name in enumerate(MODALITIES):
        rows = []
        for _, media in runs:
            n = store[media].shape[0]
            idx = np.sort(rng.choice(n, min(per_run, n), replace=False))
            rows.append(store[media][idx, m].astype(np.float32))
        X = torch.from_numpy(np.concatenate(rows))
        mean = X.mean(0)
        torch.manual_seed(seed + m)
        _, S, V = torch.pca_lowrank(X - mean, q=min(k + 32, X.shape[1]), niter=4)
        explained = float((S[:k] ** 2).sum() / ((X - mean) ** 2).sum())
        log.info("%s: PCA of %d frames, %.3f of variance in %d components",
                 name, X.shape[0], explained, k)
        means.append(mean.numpy())
        comps.append(V[:, :k].contiguous().numpy())
    return np.stack(means), np.stack(comps)


def design(reduced, n_tr, lag_op, pad_row):
    """Design rows of a run's TRs: the run is cut into 298-frame windows,
    frames outside the clip take the value of a zero feature, and the lag
    operator averages each window."""
    n_win = int(np.ceil(n_tr / WINDOW_TRS))
    out = np.zeros((n_win * WINDOW_TRS, lag_op.shape[0] // WINDOW_TRS * reduced.shape[1]),
                   dtype=np.float32)
    A = lag_op.numpy()
    for w in range(n_win):
        lo = WINDOW_FRAMES * w - WINDOW_OFFSET_FRAMES
        buf = np.tile(pad_row, (WINDOW_FRAMES, 1))
        s0, s1 = max(0, lo), min(len(reduced), lo + WINDOW_FRAMES)
        if s1 > s0:
            buf[s0 - lo:s1 - lo] = reduced[s0:s1]
        out[w * WINDOW_TRS:(w + 1) * WINDOW_TRS] = (A @ buf).reshape(WINDOW_TRS, -1)
    return out[:n_tr]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--features", required=True, help="brain_lora.cli.extract_features output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--n-sample", type=int, default=120000)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")

    paths = data_paths(args.data_dir)
    runs = runs_by_split(load_manifests(paths["manifests"]))
    store = h5py.File(args.features, "r")
    fmri = h5py.File(paths["fmri_h5"], "r")

    mean, comps = fit_pca(store, runs["train"], args.k, args.n_sample, args.seed)
    pad_row = np.concatenate([-(mean[m] @ comps[m]) for m in range(len(MODALITIES))])
    lag_op = build_lag_operator(LAG_BLOCKS)
    dev = torch.device(args.device)
    proj_mean, proj = torch.from_numpy(mean).to(dev), torch.from_numpy(comps).to(dev)

    def run_rows(stimulus_id, media):
        x = torch.from_numpy(store[media][:].astype(np.float32)).to(dev)
        red = torch.einsum("tmd,mdk->tmk", x - proj_mean, proj)
        red = red.reshape(len(x), -1).cpu().numpy()
        Y = np.stack([fmri[f"{s}/{stimulus_id}"][:].astype(np.float32) for s in SUBJECTS])
        return torch.from_numpy(design(red, Y.shape[1], lag_op, pad_row)), torch.from_numpy(Y)

    train = runs["train"]
    perm = np.random.RandomState(args.seed).permutation(len(train))
    held_set = set(perm[:int(len(train) * args.holdout_frac)].tolist())
    dim, n_sub = lag_op.shape[0] // WINDOW_TRS * pad_row.size, len(SUBJECTS)
    fit = RidgeAccumulator(dim, 1000, n_sub, dev)
    held = RidgeAccumulator(dim, 1000, n_sub, dev)
    held_X, held_Y, everyone = [], [[] for _ in SUBJECTS], torch.arange(n_sub)
    for i, run in enumerate(train):
        X, Y = run_rows(*run)
        if i in held_set:
            held.add(X, Y, everyone)
            held_X.append(X.double())
            for s in range(n_sub):
                held_Y[s].append(Y[s].double())
        else:
            fit.add(X, Y, everyone)
    held_X = torch.cat(held_X)
    rows = [torch.arange(len(held_X))] * n_sub
    W, b, mx, picks = fit_per_parcel_alpha(fit, held, held_X, rows,
                                           [torch.cat(y) for y in held_Y], ALPHAS)

    # frozen ridge readout on season 6
    val = [run_rows(*run) for run in runs["val"]]
    Xv = torch.cat([x for x, _ in val]).to(dev).double()
    report = {}
    for s, subject in enumerate(SUBJECTS):
        Yv = torch.cat([y[s] for _, y in val]).to(dev).double()
        report[subject] = float(torch.nanmean(pearson_columns((Xv - mx[s]) @ W[s] + b[s], Yv)))
    report["subject_mean"] = float(np.mean([report[s] for s in SUBJECTS]))
    log.info("frozen ridge readout, season 6: %s", report)

    np.savez(args.out, layer=store.attrs["tap_layer"],
             blocks=np.array(LAG_BLOCKS, dtype=np.float32), k_use=args.k,
             subjects=np.array(SUBJECTS), mods=np.array(MODALITIES),
             pca_mean=mean, pca_comps=comps, feat_mean=mx[0].float().cpu().numpy(),
             W=W.float().cpu().numpy(), b=b.float().cpu().numpy(),
             report=json.dumps(report))


if __name__ == "__main__":
    main()
