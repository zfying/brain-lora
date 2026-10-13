"""Prepare the training data from the Algonauts 2025 download.

    python -m brain_lora.cli.prepare_data --algonauts-root <download> --out data

Writes, under ``--out``:

* ``media_cache/``  every labelled clip decoded once (CPU, parallel, resumable);
* ``manifests/train.tsv``, ``manifests/val.tsv``  one row per subject, run and
  149 s window (train: Friends seasons 1-5 and Movie10; val: Friends season 6);
* ``fmri.h5``  every BOLD run z-scored per parcel.
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from brain_lora.data.algonauts import build_manifests, find_runs, media_key, write_fmri_store
from brain_lora.data.media import build_cache_entry

log = logging.getLogger("prepare_data")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--algonauts-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    out = Path(args.out)
    runs = find_runs(args.algonauts_root)
    log.info("%d fMRI runs", len(runs))

    cache = out / "media_cache"
    clips = {media_key(r): r for r in runs}
    log.info("decoding %d clips into %s", len(clips), cache)
    with ProcessPoolExecutor(args.workers) as ex:
        jobs = [ex.submit(build_cache_entry, r["video"], r["transcript"],
                          cache / r["task"]) for r in clips.values()]
        for n, job in enumerate(jobs, 1):
            job.result()
            if n % 25 == 0:
                log.info("%d/%d clips", n, len(jobs))

    (out / "manifests").mkdir(parents=True, exist_ok=True)
    train, val = build_manifests(runs, cache)
    train.to_csv(out / "manifests" / "train.tsv", sep="\t", index=False)
    val.to_csv(out / "manifests" / "val.tsv", sep="\t", index=False)
    write_fmri_store(runs, out / "fmri.h5")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
