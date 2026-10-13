"""Paired per-parcel comparison of two runs with a functional-block bootstrap.

    python -m brain_lora.analysis.compare runs/control runs/arm --data-dir data

The statistic is the mean over parcels of ``r_arm - r_control`` (per-parcel
accuracy pooled over subjects, from ``<stem>_per_parcel.npy``). Neighbouring
parcels are correlated, so parcels are resampled in 100 blocks obtained by
k-means on the parcels' functional-connectivity profiles over the validation
fMRI (cached next to the manifests).
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def functional_blocks(data_dir, k=100, seed=0):
    """(1000,) block label per parcel, cached in ``<data_dir>/blocks_k<k>.npz``."""
    from sklearn.cluster import KMeans

    from brain_lora.data.algonauts import load_manifests

    cache = Path(data_dir) / f"blocks_k{k}.npz"
    if cache.exists():
        return np.load(cache)["labels"]
    _, val = load_manifests(Path(data_dir) / "manifests")
    cols = []
    with h5py.File(Path(data_dir) / "fmri.h5", "r") as f:
        for key in dict.fromkeys(val["fmri_item_id"]):
            x = f[key][:].astype(np.float32)
            x = x - x.mean(0)
            sd = x.std(0)
            cols.append(x / np.where(sd > 0, sd, 1.0))
    C = np.nan_to_num(np.corrcoef(np.concatenate(cols).T))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(C)
    np.savez(cache, labels=labels)
    return labels


def block_bootstrap(delta, labels, n_boot=1000, seed=0):
    """95% interval of ``delta.mean()`` resampling whole blocks."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == b) for b in np.unique(labels)]
    draws = np.array([delta[np.concatenate([groups[j] for j in
                                            rng.integers(0, len(groups), len(groups))])].mean()
                      for _ in range(n_boot)])
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("control")
    ap.add_argument("arm")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--stem", default="refit_last",
                    help="per-parcel file stem: refit_last (after re-fit) or val_last")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    a = np.load(Path(args.control) / f"{args.stem}_per_parcel.npy").astype(np.float64)
    b = np.load(Path(args.arm) / f"{args.stem}_per_parcel.npy").astype(np.float64)
    delta = b - a
    lo, hi = block_bootstrap(delta, functional_blocks(args.data_dir), args.n_boot)
    rep = {"control": args.control, "arm": args.arm, "paired_delta": float(delta.mean()),
           "ci95": [float(lo), float(hi)], "parcels_improved": int((delta > 0).sum())}
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
