"""Trained models in the Hugging Face Hub layout.

A published model is one directory::

    config.json                  run configuration and readout shape
    adapter_config.json          LoRA on the Qwen3-Omni thinker (PEFT format)
    adapter_model.safetensors
    readout.safetensors

``export_run`` writes it from a run's checkpoint. ``import_model`` turns it
back into a run directory (``config.json`` + ``refit_last.pt``) that
``brain_lora.cli.evaluate``, ``brain_lora.cli.refit`` and the attribution
analysis read like any other run.
"""

import dataclasses
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from brain_lora.constants import MODEL_ID, SUBJECTS
from brain_lora.model.encoder import LORA_TARGETS
from brain_lora.model.readouts import load_init
from brain_lora.training import Config, build_readout

ADAPTER = "default"                     # PEFT's adapter name inside a live model
LOCAL_ONLY = ("data_dir", "readout_init", "readout_shape")


def readout_shape(state, blocks):
    """Constructor arguments that rebuild a readout holding ``state``."""
    _, feat_dim, k = state["proj"].shape
    n_subjects, _, n_parcels = state["W"].shape
    return {"blocks": [[float(d), float(w)] for d, w in blocks], "k": int(k),
            "feat_dim": int(feat_dim), "n_subjects": int(n_subjects),
            "n_parcels": int(n_parcels)}


def export_run(run_dir, out_dir, checkpoint="refit_last.pt"):
    """Write ``run_dir/checkpoint`` as a published model in ``out_dir``."""
    from peft import LoraConfig

    cfg = Config.load(run_dir)
    payload = torch.load(Path(run_dir) / checkpoint, map_location="cpu", weights_only=False)
    shape = readout_shape(payload["readout"], load_init(cfg.readout_init, cfg.mods)["blocks"])
    # the shape must rebuild this exact readout, lag operator included
    rebuilt = build_readout(dataclasses.replace(cfg, readout_shape=shape))
    if not torch.equal(rebuilt.lag_op, payload["readout"]["lag_op"]):
        raise RuntimeError("lag blocks of the init do not match the checkpoint")
    rebuilt.load_state_dict(payload["readout"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter = {k.replace(f".{ADAPTER}.", "."): v.contiguous()
               for k, v in payload["encoder_trainable"].items()}
    save_file(adapter, out / "adapter_model.safetensors")
    LoraConfig(r=cfg.rank, lora_alpha=2 * cfg.rank if cfg.lora_alpha is None else cfg.lora_alpha,
               lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none",
               base_model_name_or_path=MODEL_ID).save_pretrained(out)
    save_file({k: v.contiguous() for k, v in payload["readout"].items()},
              out / "readout.safetensors")
    config = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in LOCAL_ONLY}
    config.update(readout_shape=shape, base_model=MODEL_ID, subjects=list(SUBJECTS))
    (out / "config.json").write_text(json.dumps(config, indent=2))


def import_model(model_dir, run_dir, data_dir):
    """Make ``run_dir`` a run directory holding the published model."""
    model_dir = Path(model_dir)
    raw = json.loads((model_dir / "config.json").read_text())
    if raw["base_model"] != MODEL_ID or raw["subjects"] != list(SUBJECTS):
        raise ValueError(f"{model_dir} was trained on {raw['base_model']} for "
                         f"{raw['subjects']}, this code expects {MODEL_ID} for {list(SUBJECTS)}")
    names = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(data_dir=str(data_dir), readout_init="",
                 **{k: v for k, v in raw.items() if k in names and k not in LOCAL_ONLY[:2]})
    adapter = load_file(model_dir / "adapter_model.safetensors")
    trainable = {k.replace(".lora_A.", f".lora_A.{ADAPTER}.")
                  .replace(".lora_B.", f".lora_B.{ADAPTER}."): v for k, v in adapter.items()}
    cfg.save(run_dir)
    torch.save({"encoder_trainable": trainable,
                "readout": load_file(model_dir / "readout.safetensors"), "step": 0},
               Path(run_dir) / "refit_last.pt")
    return cfg
