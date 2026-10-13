"""Score a run's checkpoint on the validation set (Friends season 6).

    python -m brain_lora.cli.evaluate runs/lora_linear --checkpoint refit_last.pt

Writes ``eval_<checkpoint>.json`` and per-parcel accuracies to the run directory;
``--dump-predictions`` also writes every predicted row.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from brain_lora.data.dataset import make_loader
from brain_lora.metrics import save_report
from brain_lora.training import Config, build_datasets, build_trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--checkpoint", default="refit_last.pt")
    ap.add_argument("--max-items", type=int, default=None, help="score only N windows")
    ap.add_argument("--dump-predictions", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")

    run_dir = Path(args.run_dir)
    cfg = Config.load(run_dir)
    (val_ds,) = build_datasets(cfg, splits=("val",))
    trainer = build_trainer(cfg, args.device)
    trainer.load(run_dir / args.checkpoint)
    collect = {} if args.dump_predictions else None
    rep = trainer.validate(make_loader(val_ds, shuffle=False, num_workers=cfg.num_workers),
                           max_items=args.max_items, collect=collect)

    stem = f"eval_{Path(args.checkpoint).stem}"
    rep = save_report(rep, run_dir, stem)
    if collect is not None:
        collect["segment_ids"] = np.array([w.segment_id for w in val_ds.windows])
        np.savez_compressed(run_dir / f"{stem}_predictions.npz", **collect)
    (run_dir / f"{stem}.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "max_items": args.max_items, **rep}, indent=2))
    print(f"subject mean r = {rep['subject_mean']:.5f}")


if __name__ == "__main__":
    main()
