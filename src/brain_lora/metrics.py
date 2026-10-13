"""Encoding accuracy: per-parcel Pearson r, averaged within and then across
subjects. ``subject_mean`` is the number every table reports."""

import numpy as np

from brain_lora.constants import SUBJECTS


def pearson_per_parcel(preds, trues):
    """(rows, parcels) x2 -> (parcels,) Pearson r; NaN where a column is flat."""
    from scipy.stats import pearsonr

    out = np.full(trues.shape[1], np.nan, dtype=np.float32)
    for p in range(trues.shape[1]):
        if np.std(trues[:, p]) > 0 and np.std(preds[:, p]) > 0:
            out[p] = pearsonr(trues[:, p], preds[:, p])[0]
    return out


def pearson_report(preds, trues, subject_idx):
    """Per-subject and pooled encoding accuracy.

    Returns ``{"subject_mean", "per_subject", "pooled", "per_parcel",
    "per_parcel_per_subject"}``; the last two hold arrays.
    """
    per_subject, by_subject = {}, {}
    for s in np.unique(subject_idx):
        m = subject_idx == s
        r = pearson_per_parcel(preds[m], trues[m])
        label = SUBJECTS[int(s)]
        per_subject[label] = float(np.nanmean(r))
        by_subject[label] = r
    pooled = pearson_per_parcel(preds, trues)
    return {
        "subject_mean": float(np.mean(list(per_subject.values()))),
        "per_subject": per_subject,
        "pooled": float(np.nanmean(pooled)),
        "per_parcel": pooled,
        "per_parcel_per_subject": by_subject,
    }


def save_report(rep, out_dir, stem):
    """Write a report's arrays to ``<stem>_per_parcel.npy`` / ``.npz`` and
    return the JSON-serialisable remainder."""
    from pathlib import Path

    out_dir = Path(out_dir)
    rep = dict(rep)
    np.save(out_dir / f"{stem}_per_parcel.npy", rep.pop("per_parcel"))
    np.savez(out_dir / f"{stem}_per_parcel_by_subject.npz",
             **rep.pop("per_parcel_per_subject"))
    return rep
