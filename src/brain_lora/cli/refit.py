"""Re-solve a trained run's readout map in closed form.

    python -m brain_lora.cli.refit runs/lora_linear --checkpoint last.pt

Training moves the features under the readout, so its linear map is no longer
the ridge optimum for them. One forward pass over the training windows
accumulates the design rows of the trained model; the per-subject ridge
solution, with a penalty chosen per parcel on 15% of the training windows,
replaces the map. Season 6 is only scored, never fitted.

Writes ``refit_<checkpoint>.pt`` and ``refit_<checkpoint>.json``.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from brain_lora.data.dataset import make_loader
from brain_lora.metrics import save_report
from brain_lora.model.refit import ALPHAS, RidgeAccumulator, fit_per_parcel_alpha, holdout_indices
from brain_lora.training import Config, build_datasets, build_trainer

log = logging.getLogger("refit")


@torch.no_grad()
def accumulate(trainer, loader, holdout, device):
    """One pass over the training windows -> (fit, held, held_X, held_rows, held_Y)."""
    readout = trainer.readout
    n_subjects, _, n_parcels = readout.W.shape
    fit = RidgeAccumulator(readout.refit_dim, n_parcels, n_subjects, device)
    held = RidgeAccumulator(readout.refit_dim, n_parcels, n_subjects, device)
    held_X, held_rows, held_Y, n_held = [], [[] for _ in range(n_subjects)], \
        [[] for _ in range(n_subjects)], 0
    trainer._set_mode(False)
    t0 = time.perf_counter()
    for i, item in enumerate(loader):
        X = readout.refit_design(trainer.encode_streams(item), rel_lo=item["rel_lo"],
                                 n_take=item["n_take"], dst_lo=item["dst_lo"]).double()
        targets = item["targets"].double()
        if i in holdout:
            held.add(X, targets, item["subject_idx"])
            held_X.append(X.cpu())
            for row, s in enumerate(item["subject_idx"].tolist()):
                held_rows[s].append(torch.arange(n_held, n_held + X.shape[0]))
                held_Y[s].append(targets[row].cpu())
            n_held += X.shape[0]
        else:
            fit.add(X, targets, item["subject_idx"])
        if (i + 1) % 100 == 0:
            log.info("%d windows (%.1f s/window)", i + 1, (time.perf_counter() - t0) / (i + 1))
    return (fit, held, torch.cat(held_X), [torch.cat(r) for r in held_rows],
            [torch.cat(y) for y in held_Y])


def gram_device(trainer, n_rows, device):
    """The GPU if the float64 moments fit beside the model, else the CPU."""
    if not str(device).startswith("cuda"):
        return "cpu"
    d = trainer.readout.refit_dim
    S, _, P = trainer.readout.W.shape
    need = 8 * (2 * S * d * d + 3 * d * d + 2 * S * d * P + n_rows * d)
    return device if need < 0.9 * torch.cuda.mem_get_info(torch.device(device))[0] else "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--checkpoint", default="last.pt")
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--skip-before", action="store_true",
                    help="do not score the checkpoint before the re-fit")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")

    run_dir = Path(args.run_dir)
    cfg = Config.load(run_dir)
    train_ds, val_ds = build_datasets(cfg)
    trainer = build_trainer(cfg, args.device)
    trainer.load(run_dir / args.checkpoint)
    val_loader = make_loader(val_ds, shuffle=False, num_workers=cfg.num_workers)
    before = None
    if not args.skip_before:
        before = save_report(trainer.validate(val_loader, tag="before re-fit"),
                             run_dir, f"before_refit_{Path(args.checkpoint).stem}")

    holdout = holdout_indices(len(train_ds), args.holdout_frac)
    device = gram_device(trainer, len(holdout) * trainer.readout.n_trs, args.device)
    log.info("accumulating %d-column design on %s", trainer.readout.refit_dim, device)
    acc = accumulate(trainer, make_loader(train_ds, shuffle=False,
                                          num_workers=cfg.num_workers), holdout, device)
    W, b, mx, picks = fit_per_parcel_alpha(*acc, alphas=ALPHAS)
    trainer.readout.install_refit(W, b, mx)
    counts = {str(a): int((picks == i).sum()) for i, a in enumerate(ALPHAS)}
    log.info("per-parcel penalties: %s", counts)

    stem = f"refit_{Path(args.checkpoint).stem}"
    after = save_report(trainer.validate(val_loader, tag="after re-fit"), run_dir, stem)
    trainer.save(run_dir / f"{stem}.pt")
    (run_dir / f"{stem}.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "alpha_counts": counts,
         "fit_rows": acc[0].rows, "holdout_rows": acc[1].rows,
         "before": before, "after": after}, indent=2))
    print(f"subject mean r after re-fit = {after['subject_mean']:.5f}"
          + ("" if before is None else f" (before {before['subject_mean']:.5f})"))


if __name__ == "__main__":
    main()
