"""Input-level modality attribution with exact Shapley values.

    python -m brain_lora.analysis.attribution baseline --run-dir R --out A
    python -m brain_lora.analysis.attribution run      --run-dir R --out A [--shard i --n-shards n]
    python -m brain_lora.analysis.attribution report   --out A

A modality is *absent* when its LM input embeddings are replaced by a
content-free baseline that keeps its token count, positions and time-aligned
rotary ids, so only stimulus content is removed:

* video: the vision tower output (and its deepstack features) -> their means;
* audio: the audio tower output -> its mean;
* text: the transcript-token rows of the embedding output -> their mean.

The means are taken over training windows (``baseline``). ``run`` evaluates
all eight modality subsets on every validation window (tower outputs are
computed once per window and replayed). With ``--baseline donor`` an absent
modality instead receives the real content of a window from another episode.

``report`` forms the Shapley components ``phi_V, phi_A, phi_T`` with
``f(full) = f(empty) + phi_V + phi_A + phi_T`` row by row, and per parcel and
subject the accuracy contribution ``A_c = cov(c, y) / (sd f sd y)`` (which sums
to Pearson r) and the variance share ``S_c = cov(c, f) / var f`` (which sums to
one). Region summaries carry episode block-bootstrap intervals.
"""

import argparse
import contextlib
import csv
import itertools
import json
import logging
import re
from collections import OrderedDict
from importlib import resources
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("attribution")

MODS = ("vision", "audio", "text")          # Shapley players; bit i = MODS[i]
BIT = {m: 1 << i for i, m in enumerate(MODS)}
FULL = 7
FIRST_SPECIAL_ID = 151643                   # Qwen special tokens are >= this


def text_mask(input_ids):
    """Transcript-token positions: ordinary token ids, minus the constant
    ``<|im_start|>user\\n`` head and the trailing newline."""
    ids = input_ids[0]
    m = ids < FIRST_SPECIAL_ID
    m[:3] = False
    m[-1:] = False
    return m


def _tile(src, like):
    reps = -(-like.shape[0] // src.shape[0])
    return src.repeat(reps, *([1] * (src.dim() - 1)))[:like.shape[0]].to(like.dtype)


class ModalityAblator:
    """Replaces absent modalities' LM inputs; ``present`` names the rest."""

    def __init__(self, thinker):
        self.present = set(MODS)
        self.cache, self.base, self.donor, self.collect, self.tmask = {}, None, None, None, None
        get_video, get_audio = thinker.get_video_features, thinker.get_audio_features

        def video(*a, **kw):
            if "video" not in self.cache:
                self.cache["video"] = get_video(*a, **kw)
            emb, ms = self.cache["video"]
            if self.collect is not None:
                self._acc("video", emb)
                for j, t in enumerate(ms):
                    self._acc(f"video_ms{j}", t)
            if "vision" in self.present or self.base is None:
                return emb, ms
            if self.donor is not None:
                return (_tile(self.donor["video"], emb),
                        tuple(_tile(d, t) for d, t in zip(self.donor["video_ms"], ms)))
            return (self.base["video"].to(emb.dtype).expand_as(emb),
                    tuple(self.base[f"video_ms{j}"].to(t.dtype).expand_as(t)
                          for j, t in enumerate(ms)))

        def audio(*a, **kw):
            if "audio" not in self.cache:
                self.cache["audio"] = get_audio(*a, **kw)
            emb = self.cache["audio"]
            if self.collect is not None:
                self._acc("audio", emb)
            if "audio" in self.present or self.base is None:
                return emb
            if self.donor is not None:
                return _tile(self.donor["audio"], emb)
            return self.base["audio"].to(emb.dtype).expand_as(emb)

        thinker.get_video_features = video
        thinker.get_audio_features = audio
        thinker.model.embed_tokens.register_forward_hook(self._text)

    def _acc(self, key, t):
        t = t.reshape(-1, t.shape[-1]).double()
        s, n = self.collect.get(key, (0.0, 0))
        self.collect[key] = (s + t.sum(0), n + t.shape[0])

    def _text(self, mod, inp, out):
        if self.tmask is None:
            return None
        m = self.tmask.to(out.device)
        self.cache["text_rows"] = out[0, m].detach()
        if self.collect is not None:
            self._acc("text", out[0, m])
        if "text" in self.present or self.base is None:
            return None
        out = out.clone()
        if self.donor is not None and self.donor["text"].shape[0] > 0:
            out[0, m] = _tile(self.donor["text"], out[0, m])
        else:
            out[0, m] = self.base["text"].to(out.dtype)
        return out

    def new_window(self, input_ids):
        self.cache, self.tmask = {}, text_mask(input_ids)

    def content(self):
        emb, ms = self.cache["video"]
        return {"video": emb, "video_ms": ms, "audio": self.cache["audio"],
                "text": self.cache["text_rows"]}


def subsets():
    """All eight presence sets, the full set first (it fills the tower cache)."""
    return [frozenset(c) for r in range(len(MODS), -1, -1)
            for c in itertools.combinations(MODS, r)]


def readout_split(readout, streams, item):
    """Prediction split over the readout's text/audio/vision design blocks:
    ``pred = sum_m feat[:, cols_m] @ W[:, cols_m] + b``."""
    feat = readout.features(streams, rel_lo=item["rel_lo"], n_take=item["n_take"],
                            dst_lo=item["dst_lo"])
    T, M, k = feat.shape[0], readout.n_mods, readout.k
    B = readout.out_dim // (M * k)
    idx = item["subject_idx"].to(readout.W.device).long()
    W = readout.W[idx].reshape(len(idx), B, M, k, -1)
    parts = torch.einsum("tbmk,sbmkp->mstp", feat.reshape(T, B, M, k).float(), W.float())
    by_name = dict(zip(readout.mods, parts))
    return torch.stack([by_name[m] for m in MODS]), readout.b[idx].float()


def _load(run_dir, checkpoint, device):
    from brain_lora.model.readouts import GatedMLPReadout
    from brain_lora.training import Config, build_trainer

    cfg = Config.load(run_dir)
    trainer = build_trainer(cfg, device)
    if isinstance(trainer.readout, GatedMLPReadout):
        raise SystemExit("attribution covers readouts of the form features @ W + b")
    trainer.load(Path(run_dir) / checkpoint)
    trainer._set_mode(False)
    return cfg, trainer


@torch.no_grad()
def stage_baseline(args):
    from brain_lora.training import build_datasets

    cfg, trainer = _load(args.run_dir, args.checkpoint, args.device)
    ab = ModalityAblator(trainer.encoder.thinker)
    (ds,) = build_datasets(cfg, splits=("train",), targets=False)
    ab.collect = {}
    for i in np.linspace(0, len(ds) - 1, args.n_baseline_windows).round().astype(int):
        item = ds[int(i)]
        if item["inputs"] is None:
            continue
        ab.new_window(item["inputs"]["input_ids"])
        trainer.encode_streams(item)
        trainer.encoder.tap.release()
    means = {k: (s / n).float().cpu() for k, (s, n) in ab.collect.items()}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save(means, Path(args.out) / "baseline_means.pt")


@torch.no_grad()
def stage_run(args):
    from brain_lora.data.dataset import make_loader
    from brain_lora.training import build_datasets, deterministic

    out = Path(args.out)
    (out / "windows").mkdir(parents=True, exist_ok=True)
    cfg, trainer = _load(args.run_dir, args.checkpoint, args.device)
    enc, readout = trainer.encoder, trainer.readout
    ab = ModalityAblator(enc.thinker)
    ab.base = {k: v.to(args.device) for k, v in
               torch.load(out / "baseline_means.pt").items()}
    (ds,) = build_datasets(cfg, splits=("val",))
    shard = np.array_split(np.arange(0, len(ds), args.stride), args.n_shards)[args.shard]
    pending = [int(i) for i in shard if not (out / "windows" / f"w{int(i):04d}.npz").exists()]
    loader = make_loader(torch.utils.data.Subset(ds, pending), shuffle=False,
                         num_workers=args.num_workers)

    def predict(item):
        streams = trainer.encode_streams(item)
        p = readout(streams, item["subject_idx"], rel_lo=item["rel_lo"],
                    n_take=item["n_take"], dst_lo=item["dst_lo"])
        parts, bias = readout_split(readout, streams, item)
        enc.tap.release()
        return p.float(), parts, bias

    # token readouts mean-pool vision with atomic adds; fix their order so the
    # replayed tower outputs reproduce the full forward exactly
    ctx = deterministic() if readout.NEEDS_TOKENS else contextlib.nullcontext()
    with ctx:
        for n, (di, item) in enumerate(zip(pending, loader)):
            if item["inputs"] is None:
                continue
            if args.baseline == "donor":    # a window from another episode
                donor = ds[(di + len(ds) // 2) % len(ds)]
                ab.donor, ab.present = None, set(MODS)
                ab.new_window(donor["inputs"]["input_ids"])
                trainer.encode_streams(donor)
                enc.tap.release()
                ab.donor = ab.content()
            ab.new_window(item["inputs"]["input_ids"])
            subs = subsets()
            shape = tuple(item["targets"].shape)
            preds = np.zeros((len(subs),) + shape, np.float32)
            parts_all = np.zeros((len(subs), len(MODS)) + shape, np.float16)
            masks = np.array([sum(BIT[m] for m in s) for s in subs], np.int8)
            for j, present in enumerate(subs):
                ab.present = set(present)
                p, parts, bias = predict(item)
                preds[j] = p.cpu().numpy()
                parts_all[j] = parts.cpu().numpy().astype(np.float16)
                if j == 0:
                    err = float((parts.sum(0) + bias[:, None] - p).abs().max())
                    if err > 1e-3:
                        raise RuntimeError(f"readout split does not add up ({err})")
            if n == 0:
                # replaying the cached towers reproduces the full forward, and
                # removing any one modality moves the prediction
                ab.present = set(MODS)
                replay = float(np.abs(predict(item)[0].cpu().numpy() - preds[0]).max())
                j_of = {int(b): j for j, b in enumerate(masks)}
                effect = {m: float(np.abs(preds[0] - preds[j_of[FULL & ~BIT[m]]]).mean())
                          for m in MODS}
                log.info("self-test: replay max |d| %.1e, removal effects %s", replay, effect)
                if replay > 1e-3 or min(effect.values()) <= 0:
                    raise RuntimeError("attribution self-test failed")
            np.savez(out / "windows" / f"w{di:04d}.npz",
                     preds=preds.astype(np.float16), masks=masks,
                     readout_parts_subsets=parts_all,
                     trues=item["targets"].numpy().astype(np.float16),
                     subject_idx=item["subject_idx"].numpy(),
                     segment_id=np.asarray(item["segment_id"]))
            log.info("window %d (%d/%d)", di, n + 1, len(pending))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
SERIES = (["shap_" + m for m in MODS] + ["layout"]
          + ["solo_" + m for m in MODS] + ["drop_" + m for m in MODS]
          + [f"flow_{i}_{j}" for i in MODS for j in MODS]
          + [f"flowlay_{j}" for j in MODS])

#: functional regions over Schaefer 17-network sub-region labels
ROIS = OrderedDict([
    ("Early visual (V1/V2)", r"VisCent_Striate|VisPeri_StriCal"),
    ("Extrastriate visual", r"VisCent_ExStr|VisPeri_ExStr"),
    ("MT+ / temporo-occipital", r"DorsAttn[AB]_TempOcc"),
    ("Superior parietal", r"DorsAttnA_(SPL|ParOcc)"),
    ("Auditory cortex", r"SomMotB_Aud"),
    ("Lateral temporal (STS/MTG)", r"TempPar|DefaultB_Temp"),
    ("Inferior frontal", r"DefaultB_PFC[vl]"),
    ("Temporal pole", r"LimbicA_TempPole"),
    ("Somatomotor", r"SomMotA|SomMotB_(Cent|S2|Ins)"),
    ("Default medial (PCC/mPFC)", r"DefaultA_(pCunPCC|PFCm|PFCd)"),
    ("Frontoparietal control", r"^Cont"),
    ("Salience / ventral attention", r"^SalVentAttn"),
    ("Orbitofrontal", r"LimbicB_OFC"),
])
NETWORKS = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")


def shapley(pred, masks):
    """(8, ...) subset predictions -> ({modality: phi}, f(bits))."""
    j = {int(b): i for i, b in enumerate(masks)}
    f = lambda bits: pred[j[bits]]                              # noqa: E731
    weight = {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}
    phi = {}
    for m in MODS:
        others = [o for o in MODS if o != m]
        coalitions = [()] + [(o,) for o in others] + [tuple(others)]
        phi[m] = sum(weight[len(c)] * (f(sum(BIT[o] for o in c) | BIT[m])
                                       - f(sum(BIT[o] for o in c)))
                     for c in coalitions)
    return phi, f


def window_stats(z):
    """Per-window sufficient statistics (sums over TRs, per subject and parcel)."""
    pred, masks = z["preds"].astype(np.float64), z["masks"]
    y = z["trues"].astype(np.float64)
    phi, f = shapley(pred, masks)
    full, empty = f(FULL), f(0)
    series = {"layout": empty}
    for m in MODS:
        series["shap_" + m] = phi[m]
        series["solo_" + m] = f(BIT[m])
        series["drop_" + m] = f(FULL & ~BIT[m])
    parts = z["readout_parts_subsets"].astype(np.float64)       # (8, 3, S, T, P)
    for jj, j in enumerate(MODS):
        phj, fj = shapley(parts[:, jj], masks)
        for i in MODS:
            series[f"flow_{i}_{j}"] = phj[i]
        series[f"flowlay_{j}"] = fj(0)
    X = np.stack([series[k] for k in SERIES])                   # (K, S, T, P)
    st = {"n": np.full(y.shape[0], y.shape[1], np.float64),
          "sy": y.sum(1), "syy": (y * y).sum(1), "sf": full.sum(1),
          "sff": (full * full).sum(1), "sfy": (full * y).sum(1),
          "sx": X.sum(2), "sxx": (X * X).sum(2), "sxf": (X * full).sum(2),
          "sxy": (X * y).sum(2)}
    resid = float(np.abs(empty + sum(phi.values()) - full).max())
    return st, z["subject_idx"].astype(int), resid


def derive(tot, series=SERIES):
    """Summed statistics -> per (subject, parcel) maps."""
    n = tot["n"][:, None]
    cov = lambda sab, sa, sb: sab / n - sa * sb / n ** 2         # noqa: E731
    vf = cov(tot["sff"], tot["sf"], tot["sf"])
    vy = cov(tot["syy"], tot["sy"], tot["sy"])
    out = {"r": cov(tot["sfy"], tot["sf"], tot["sy"]) / np.sqrt(vf * vy)}
    cxf = cov(tot["sxf"], tot["sx"], tot["sf"][None])
    cxy = cov(tot["sxy"], tot["sx"], tot["sy"][None])
    vx = cov(tot["sxx"], tot["sx"], tot["sx"])
    for i, s in enumerate(series):
        out["S_" + s] = cxf[i] / vf
        out["A_" + s] = cxy[i] / np.sqrt(vf * vy)
        out["r_" + s] = cxy[i] / np.sqrt(vx[i] * vy)
    return out


def parcel_groups():
    with resources.files("brain_lora.resources").joinpath(
            "schaefer1000_labels.tsv").open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    sub17 = ["_".join(r["name17"].split("_")[2:-1]) for r in rows]
    net7 = [r["name7"].split("_")[2] for r in rows]
    groups = OrderedDict((roi, np.array([bool(re.search(p, s)) for s in sub17]))
                         for roi, p in ROIS.items())
    groups.update((n, np.array([x == n for x in net7])) for n in NETWORKS)
    groups["Whole cortex"] = np.ones(len(rows), bool)
    return groups, rows


def report(args):
    out = Path(args.out)
    files = sorted((out / "windows").glob("w*.npz"))
    stats, segs, worst = [], [], 0.0
    for fp in files:
        z = np.load(fp)
        st, sidx, resid = window_stats(z)
        stats.append((st, sidx))
        segs.append(str(z["segment_id"]))
        worst = max(worst, resid)
    log.info("%d windows; Shapley additivity residual %.1e", len(files), worst)

    stack = {}                       # per window, re-indexed onto the four subjects
    for key, v in stats[0][0].items():
        shape = ((v.shape[0], 4) + v.shape[2:] if key.startswith("sx")
                 else (4,) + v.shape[1:])
        stack[key] = np.zeros((len(stats),) + shape)
    for w, (st, sidx) in enumerate(stats):
        for key, v in st.items():
            if key.startswith("sx"):
                stack[key][w][:, sidx] = v
            else:
                stack[key][w][sidx] = v
    maps = derive({k: v.sum(0) for k, v in stack.items()})

    groups, labels = parcel_groups()
    episodes = np.array([re.search(r"s\d\de\d\d", s).group(0) if "friends" in s
                         else s.split("@")[0] for s in segs])
    eps = np.unique(episodes)
    ep_stats = {k: np.stack([v[episodes == e].sum(0) for e in eps])
                for k, v in stack.items()}
    keys = ([f"A_shap_{m}" for m in MODS] + ["A_layout", "r"]
            + [f"S_shap_{m}" for m in MODS] + ["S_layout"]
            + [f"r_drop_{m}" for m in MODS] + [f"S_flow_{i}_{j}" for i in MODS for j in MODS])
    point = np.array([[np.nanmean(maps[k][:, g]) for k in keys] for g in groups.values()])
    rng = np.random.default_rng(0)
    boot = np.zeros((args.n_boot,) + point.shape)
    for b in range(args.n_boot):
        cnt = np.bincount(rng.integers(0, len(eps), len(eps)), minlength=len(eps))
        mp = derive({k: np.tensordot(cnt.astype(float), v, axes=1)
                     for k, v in ep_stats.items()})
        boot[b] = [[np.nanmean(mp[k][:, g]) for k in keys] for g in groups.values()]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    table = [{"group": g, "n_parcels": int(m.sum()),
              **{k: float(point[gi, ki]) for ki, k in enumerate(keys)},
              **{k + "_ci": [float(lo[gi, ki]), float(hi[gi, ki])]
                 for ki, k in enumerate(keys)}}
             for gi, (g, m) in enumerate(groups.items())]
    np.savez(out / "attribution_maps.npz", **{k: v.astype(np.float32) for k, v in maps.items()})
    (out / "attribution_regions.json").write_text(json.dumps(
        {"n_windows": len(files), "n_episodes": len(eps), "n_boot": args.n_boot,
         "shapley_residual": worst, "groups": table}, indent=2))
    with open(out / "attribution_parcels.tsv", "w") as fh:
        fh.write("\t".join(["parcel", "name7", "name17"] + keys) + "\n")
        for p, row in enumerate(labels):
            fh.write("\t".join([row["parcel"], row["name7"], row["name17"]]
                               + [f"{np.nanmean(maps[k][:, p]):.5f}" for k in keys]) + "\n")
    print(f"{'region':32s} {'r':>6s} {'video':>6s} {'audio':>6s} {'text':>6s} {'layout':>6s}"
          "   (share of the prediction)")
    for row in table:
        print(f"{row['group']:32s} {row['r']:.3f}  " + "  ".join(
            f"{row[k]:.3f}" for k in ["S_shap_vision", "S_shap_audio", "S_shap_text",
                                      "S_layout"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["baseline", "run", "report"])
    ap.add_argument("--run-dir")
    ap.add_argument("--checkpoint", default="refit_last.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", choices=["mean", "donor"], default="mean")
    ap.add_argument("--n-baseline-windows", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1, help="score every k-th window")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    {"baseline": stage_baseline, "run": stage_run, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
