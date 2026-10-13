"""Joint training of LoRA and readout, and the shared run reconstruction.

One step is one 149 s window: encode it with the adapted backbone, predict
every subject's 100 x 1000 target block with the readout, take the MSE, and
update LoRA and readout with AdamW under a warm-up + cosine schedule. A run is
stored as ``config.json`` plus trainable-only checkpoints; evaluation, re-fit
and attribution rebuild the model from ``config.json``.
"""

import contextlib
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from brain_lora.metrics import pearson_report, save_report

log = logging.getLogger(__name__)


@dataclass
class Config:
    data_dir: str
    readout_init: str
    readout: str = "linear"            # linear | cross_attn | gated_mlp
    readout_shape: dict | None = None  # set by brain_lora.hub; replaces readout_init
    tap_layer: int = 32
    rank: int = 16
    lora_alpha: int | None = None      # default 2 * rank
    lora_lr: float = 6e-4              # 0: readout-only control (frozen backbone)
    readout_lr: float = 3e-4
    proj_lr: float = 6e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 30
    max_steps: int = 1429
    seed: int = 0
    context_lead_s: float = 60.0
    chunk_s: float = 20.0
    use_audio: bool = True
    mlp_hidden: int = 512
    mlp_gate_init: float = 1.0
    attn_d: int = 128
    attn_bias_floor: float = -12.0
    val_every: int = 200
    val_subset: int = 40
    save_every: int = 200
    num_workers: int = 6

    @classmethod
    def load(cls, run_dir):
        raw = json.loads((Path(run_dir) / "config.json").read_text())
        return cls(**{f.name: raw[f.name] for f in fields(cls) if f.name in raw})

    def save(self, run_dir):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "config.json").write_text(json.dumps(asdict(self), indent=2))

    @property
    def mods(self):
        return ("text", "audio", "vision") if self.use_audio else ("text", "vision")


def data_paths(data_dir):
    d = Path(data_dir)
    return {"manifests": d / "manifests", "media_cache": d / "media_cache",
            "fmri_h5": d / "fmri.h5"}


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_datasets(cfg, splits=("train", "val"), targets=True):
    from brain_lora.data.algonauts import load_manifests
    from brain_lora.data.dataset import WindowDataset
    from brain_lora.model.readouts import READOUTS

    p = data_paths(cfg.data_dir)
    dfs = dict(zip(("train", "val"), load_manifests(p["manifests"])))
    return [WindowDataset(dfs[s], p["media_cache"], p["fmri_h5"] if targets else None,
                          context_lead_s=cfg.context_lead_s, chunk_s=cfg.chunk_s,
                          use_audio=cfg.use_audio,
                          tokens=READOUTS[cfg.readout].NEEDS_TOKENS)
            for s in splits]


def build_readout(cfg):
    from brain_lora.model.readouts import READOUTS

    kw = {"gated_mlp": dict(hidden=cfg.mlp_hidden, gate_init=cfg.mlp_gate_init),
          "cross_attn": dict(d_att=cfg.attn_d, bias_floor=cfg.attn_bias_floor)
          }.get(cfg.readout, {})
    if cfg.readout_shape is not None:      # weights are loaded from a checkpoint
        return READOUTS[cfg.readout](mods=cfg.mods, **cfg.readout_shape, **kw)
    return READOUTS[cfg.readout].from_init(cfg.readout_init, mods=cfg.mods, **kw)


def build_trainer(cfg, device="cuda:0"):
    """Readout, then encoder (this order fixes the random initialisation)."""
    from brain_lora.model.encoder import OmniEncoder

    readout = build_readout(cfg).to(device)
    encoder = OmniEncoder(tap_layer=cfg.tap_layer, rank=cfg.rank,
                          alpha=cfg.lora_alpha, use_audio=cfg.use_audio,
                          device=device)
    if cfg.lora_lr == 0:
        for p in encoder.lora_params:
            p.requires_grad_(False)
    return Trainer(encoder, readout, cfg, device=device)


def lr_lambda(step, warmup, max_steps):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, max_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


class Trainer:
    def __init__(self, encoder, readout, cfg, device="cuda:0"):
        self.encoder, self.readout, self.device = encoder, readout, device
        self.grad_clip = cfg.grad_clip
        groups = []
        if cfg.lora_lr > 0:
            groups.append({"params": list(encoder.lora_params), "lr": cfg.lora_lr})
        groups += readout.param_groups(cfg.readout_lr, cfg.proj_lr)
        self.opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: lr_lambda(s, cfg.warmup_steps, cfg.max_steps))
        self.trainable = [p for g in self.opt.param_groups for p in g["params"]]
        self.backbone_trains = cfg.lora_lr > 0
        self.step = 0

    def _set_mode(self, train):
        for m in (self.encoder, self.readout):
            m.train() if train else m.eval()

    def encode_streams(self, item):
        """The one encode path used by training, evaluation and the re-fit."""
        if item["inputs"] is None:
            return None
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.backbone_trains):
            if not self.readout.NEEDS_TOKENS:
                return self.encoder.encode(item["inputs"], item["grid"], item["n_keep"])
            streams = self.encoder.encode_tokens(item["inputs"], item["tokens"])
        streams[self.readout.PHI_KEY] = {n: t["phi"] for n, t in item["tokens"].items()}
        return streams

    def forward(self, item):
        return self.readout(self.encode_streams(item), item["subject_idx"],
                            rel_lo=item["rel_lo"], n_take=item["n_take"],
                            dst_lo=item["dst_lo"])

    def train_step(self, item):
        self._set_mode(True)
        t0 = time.perf_counter()
        # activations saved for backward go to pinned host memory
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            preds = self.forward(item)
            loss = torch.nn.functional.mse_loss(
                preds.float(), item["targets"].to(self.device).float())
        if loss.requires_grad:
            loss.backward()
        for p in self.trainable:        # a window past the media touches no LoRA
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        self.encoder.tap.release()
        gnorm = torch.nn.utils.clip_grad_norm_(self.trainable, self.grad_clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        self.step += 1
        return {"loss": float(loss.detach()), "grad_norm": float(gnorm),
                "lr": self.sched.get_last_lr()[0], "step_s": time.perf_counter() - t0}

    @torch.no_grad()
    def validate(self, loader, *, max_items=None, collect=None, tag="val"):
        """Encoding accuracy over a loader; ``collect`` receives the raw rows."""
        self._set_mode(False)
        preds, trues, subj, item_idx = [], [], [], []
        t0 = time.perf_counter()
        for n, item in enumerate(loader):
            if max_items is not None and n >= max_items:
                break
            p = self.forward(item)
            S, T, P = p.shape
            preds.append(p.float().cpu().numpy().reshape(S * T, P))
            trues.append(item["targets"].float().numpy().reshape(S * T, P))
            subj.append(np.repeat(item["subject_idx"].numpy(), T))
            item_idx.append(np.full(S * T, n, dtype=np.int32))
        rep = pearson_report(np.concatenate(preds), np.concatenate(trues),
                             np.concatenate(subj))
        rep["n_windows"] = len(preds)
        if collect is not None:
            collect.update(preds=np.concatenate(preds).astype(np.float16),
                           trues=np.concatenate(trues).astype(np.float16),
                           subject_idx=np.concatenate(subj).astype(np.int16),
                           item_idx=np.concatenate(item_idx))
        log.info("[%s] step %d: subject mean r %.5f %s (%d windows, %.0f s)", tag,
                 self.step, rep["subject_mean"],
                 {k: round(v, 4) for k, v in rep["per_subject"].items()},
                 len(preds), time.perf_counter() - t0)
        return rep

    def save(self, path):
        payload = {
            "encoder_trainable": {n: p.detach().cpu() for n, p in
                                  self.encoder.model.named_parameters() if p.requires_grad},
            "readout": {k: v.detach().cpu() for k, v in self.readout.state_dict().items()},
            "step": self.step, "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict()}
        tmp = Path(path).with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.rename(path)

    def load(self, path, resume=False):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        named = dict(self.encoder.model.named_parameters())
        with torch.no_grad():
            for n, v in payload["encoder_trainable"].items():
                named[n].copy_(v.to(named[n].device, named[n].dtype))
        dev = self.readout.W.device
        self.readout.load_state_dict({k: v.to(dev) for k, v in payload["readout"].items()})
        if resume:
            self.opt.load_state_dict(payload["optimizer"])
            self.sched.load_state_dict(payload["scheduler"])
            self.step = int(payload["step"])
        return payload


def selection_subset(dataset, n):
    """``n`` validation windows spread evenly over the set."""
    idx = list(range(0, len(dataset), max(1, len(dataset) // n)))
    return torch.utils.data.Subset(dataset, idx)


def train(cfg, run_dir, device="cuda:0", resume=None):
    """Train one run to ``cfg.max_steps``; write checkpoints and reports."""
    from brain_lora.data.dataset import make_loader

    run_dir = Path(run_dir)
    cfg.save(run_dir)
    seed_everything(cfg.seed)
    train_ds, val_ds = build_datasets(cfg)
    trainer = build_trainer(cfg, device)
    if resume:
        trainer.load(resume, resume=True)
    train_loader = make_loader(train_ds, shuffle=True, seed=cfg.seed,
                               num_workers=cfg.num_workers)
    val_loader = make_loader(val_ds, shuffle=False, num_workers=cfg.num_workers)
    sel_loader = make_loader(selection_subset(val_ds, cfg.val_subset), shuffle=False,
                             num_workers=min(4, cfg.num_workers))
    log.info("%d training windows, %d validation windows", len(train_ds), len(val_ds))

    history, val_history = [], []
    t_start = time.perf_counter()
    while trainer.step < cfg.max_steps:
        for item in train_loader:
            m = trainer.train_step(item)
            history.append({"step": trainer.step, **m})
            if trainer.step % 10 == 0 or trainer.step == 1:
                log.info("step %d/%d loss %.4f grad norm %.3f %.1f s lr %.2e",
                         trainer.step, cfg.max_steps, m["loss"], m["grad_norm"],
                         m["step_s"], m["lr"])
            if cfg.val_every and trainer.step % cfg.val_every == 0:
                rep = trainer.validate(sel_loader, tag="val-subset")
                val_history.append({"step": trainer.step, "subject_mean": rep["subject_mean"],
                                    "per_subject": rep["per_subject"]})
            if cfg.save_every and trainer.step % cfg.save_every == 0:
                trainer.save(run_dir / "last.pt")
            (run_dir / "history.json").write_text(
                json.dumps({"history": history, "val_history": val_history}))
            if trainer.step >= cfg.max_steps:
                break
    wall_s = time.perf_counter() - t_start
    trainer.save(run_dir / "last.pt")
    rep = save_report(trainer.validate(val_loader, tag="val"), run_dir, "val_last")
    summary = {"steps": trainer.step, "wall_s": wall_s,
               "median_step_s": float(np.median([h["step_s"] for h in history])),
               "n_lora_params": int(sum(p.numel() for p in trainer.encoder.lora_params)),
               "n_readout_params": int(sum(p.numel() for p in trainer.readout.parameters())),
               "val": rep}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done: subject mean r %.5f", rep["subject_mean"])
    return summary


@contextlib.contextmanager
def deterministic():
    """Deterministic CUDA kernels (fixes the order of the vision mean-pool's
    atomic adds, so repeated forwards of one input agree exactly)."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)
