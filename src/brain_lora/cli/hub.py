"""Export a run to, or fetch a model from, the Hugging Face Hub layout.

    python -m brain_lora.cli.hub export runs/lora_linear hub/lora_linear
    python -m brain_lora.cli.hub download <repo_id> lora_linear \\
        --data-dir data --run-dir runs/lora_linear

``export`` writes a published model directory from the run's re-fitted
checkpoint. ``download`` fetches one model of a Hub repository and makes
``--run-dir`` a run directory, so that, e.g.,

    python -m brain_lora.cli.evaluate runs/lora_linear

scores it on season 6.
"""

import argparse
from pathlib import Path

from brain_lora.hub import export_run, import_model


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="run directory -> published model directory")
    ex.add_argument("run_dir")
    ex.add_argument("out_dir")
    ex.add_argument("--checkpoint", default="refit_last.pt")
    dl = sub.add_parser("download", help="Hub model -> run directory")
    dl.add_argument("repo_id")
    dl.add_argument("model", help="model subfolder of the repository, e.g. lora_linear")
    dl.add_argument("--data-dir", required=True)
    dl.add_argument("--run-dir", required=True)
    dl.add_argument("--revision", default=None)
    args = ap.parse_args()

    if args.cmd == "export":
        export_run(args.run_dir, args.out_dir, args.checkpoint)
        print(f"wrote {args.out_dir}")
    else:
        from huggingface_hub import snapshot_download

        local = snapshot_download(args.repo_id, revision=args.revision,
                                  allow_patterns=[f"{args.model}/*"])
        import_model(Path(local) / args.model, args.run_dir, args.data_dir)
        print(f"wrote {args.run_dir}")


if __name__ == "__main__":
    main()
