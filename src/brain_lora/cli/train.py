"""Train LoRA and a readout jointly on Algonauts 2025.

    python -m brain_lora.cli.train --data-dir data --run-dir runs/lora_linear \\
        --readout-init data/readout_init_L32.npz

Defaults are the paper's recipe (LoRA r16 on q/k/v/o through layer 32, linear
readout, 60 s context lead, 1429 steps). ``--lora-lr 0`` trains the readout on
the frozen backbone (the readout-only control); ``--readout cross_attn`` selects
the cross-attention readout.
"""

import argparse
import dataclasses
import logging

from brain_lora.model.readouts import READOUTS
from brain_lora.training import Config, train

HELP = {
    "data_dir": "directory written by brain_lora.cli.prepare_data",
    "readout_init": "readout initialisation (brain_lora.cli.fit_readout_init)",
    "readout": "readout family",
    "tap_layer": "decoder layer read by the readout; the decoder is truncated there",
    "lora_lr": "learning rate of the adapters; 0 trains the readout only",
    "context_lead_s": "stimulus encoded before each window as context (s)",
    "chunk_s": "audio/video block length of the input layout (s)",
    "use_audio": "drop the audio stream from the input (ablation)",
    "val_subset": "validation windows scored every --val-every steps",
}


def add_config_args(ap):
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.name == "readout_shape":       # written by brain_lora.hub, not a flag
            continue
        if f.name == "use_audio":
            ap.add_argument("--no-audio", dest="use_audio", action="store_false",
                            help=HELP[f.name])
        elif f.default is dataclasses.MISSING:
            ap.add_argument(flag, required=True, help=HELP.get(f.name))
        else:
            kind = int if f.name == "lora_alpha" else type(f.default)
            ap.add_argument(flag, type=kind, default=f.default,
                            choices=sorted(READOUTS) if f.name == "readout" else None,
                            help=HELP.get(f.name))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    add_config_args(ap)
    args = vars(ap.parse_args())
    run_dir, device, resume = args.pop("run_dir"), args.pop("device"), args.pop("resume")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    train(Config(**args), run_dir, device=device, resume=resume)


if __name__ == "__main__":
    main()
