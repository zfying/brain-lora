"""Readouts from tapped hidden states to 1000 parcels x 100 TRs per window.

``LinearReadout`` (52.3M parameters)::

    per-frame features, 3 modalities          (298, 3, 2048)
      -> per-modality projection P_m          (298, 3, 512)
      -> lag-average into 8 log-spaced blocks (100, 8 * 3 * 512)
      -> (x - feat_mean) @ W_s + b_s          (S, 100, 1000)  one map per subject

Every step is linear, so it is initialised exactly at the closed-form ridge
solution fitted on frozen features (``from_init``) and re-solved in closed form
after training (``brain_lora.model.refit``).

``CrossAttnReadout`` replaces the fixed lag average with per-TR cross-attention
over the LM token sequence (audio at 13 Hz, every word, one pooled vision
vector per second): one learned query per lag block and modality, with a lag
bias that makes the attention equal the lag average at initialisation, so the
same ridge solution transfers.

``GatedMLPReadout`` adds a gated non-linear branch to the linear readout's own
design rows (an ablation).

The design column order is ``[block][modality][k]`` throughout.
"""

import numpy as np
import torch
import torch.nn as nn

from brain_lora.constants import (
    FRAMES_PER_TR,
    MODALITIES,
    STEP_HZ,
    WINDOW_FRAMES,
    WINDOW_OFFSET_FRAMES,
    WINDOW_TRS,
)


def build_lag_operator(blocks, n_frames=WINDOW_FRAMES, n_trs=WINDOW_TRS):
    """-> (n_trs * n_blocks, n_frames) averaging matrix.

    Block ``(delay, width)`` (seconds) of TR ``j`` averages the frames in
    ``[c_j - (delay + width), c_j - delay)`` seconds before the TR's centre
    ``c_j``, clipped to the window; a block lying entirely before the window is
    an all-zero row.
    """
    A = torch.zeros(n_trs * len(blocks), n_frames, dtype=torch.float32)
    centre = WINDOW_OFFSET_FRAMES + FRAMES_PER_TR * np.arange(n_trs)
    for b, (delay, width) in enumerate(blocks):
        lo = np.maximum(np.ceil(centre - (delay + width) * STEP_HZ).astype(np.int64), 0)
        hi = np.minimum(np.ceil(centre - delay * STEP_HZ).astype(np.int64), n_frames)
        for j in range(n_trs):
            if hi[j] > lo[j]:
                A[j * len(blocks) + b, lo[j]:hi[j]] = 1.0 / (hi[j] - lo[j])
    return A


def load_init(path, mods=MODALITIES):
    """Read a readout initialisation npz, restricted to ``mods``.

    Dropping a modality keeps its PCA basis rows and removes its columns from
    every lag block of ``W`` and ``feat_mean``; the closed-form re-fit then
    re-solves the map for the remaining columns.
    """
    z = dict(np.load(path, allow_pickle=True))
    have = [str(m) for m in z["mods"]]
    keep = [i for i, m in enumerate(have) if m in mods]
    if [have[i] for i in keep] != list(mods):
        raise ValueError(f"init holds {have}, cannot provide {list(mods)}")
    if len(keep) < len(have):
        k, n_blocks = int(z["k_use"]), len(z["blocks"])
        cols = np.concatenate([np.arange(k) + m * k + b * len(have) * k
                               for b in range(n_blocks) for m in keep])
        z["pca_mean"], z["pca_comps"] = z["pca_mean"][keep], z["pca_comps"][keep]
        z["W"], z["feat_mean"] = z["W"][:, cols], z["feat_mean"][cols]
        z["mods"] = np.array(mods)
    return z


class LinearReadout(nn.Module):
    """Lag-average -> per-modality projection -> per-subject linear map."""

    NEEDS_TOKENS = False

    def __init__(self, blocks, k, mods=MODALITIES, n_subjects=4, feat_dim=2048,
                 n_parcels=1000):
        super().__init__()
        self.blocks = [tuple(map(float, b)) for b in blocks]
        self.mods = tuple(mods)
        self.n_mods, self.k = len(self.mods), int(k)
        self.n_frames, self.n_trs = WINDOW_FRAMES, WINDOW_TRS
        self.out_dim = len(self.blocks) * self.n_mods * self.k
        self.register_buffer("lag_op", build_lag_operator(self.blocks))
        self.proj = nn.Parameter(torch.zeros(self.n_mods, feat_dim, self.k))
        self.register_buffer("proj_mean", torch.zeros(self.n_mods, feat_dim))
        self.register_buffer("feat_mean", torch.zeros(self.out_dim))
        self.W = nn.Parameter(torch.zeros(n_subjects, self.out_dim, n_parcels))
        self.b = nn.Parameter(torch.zeros(n_subjects, n_parcels))

    @classmethod
    def from_init(cls, path, mods=MODALITIES, **kw):
        """Build at the closed-form ridge solution stored in ``path``."""
        z = load_init(path, mods)
        n_subjects, _, n_parcels = z["W"].shape
        m = cls(z["blocks"], int(z["k_use"]), mods=mods, n_subjects=n_subjects,
                feat_dim=z["pca_comps"].shape[1], n_parcels=n_parcels, **kw)
        m._load_init(z)
        return m

    def _load_init(self, z):
        with torch.no_grad():
            self.proj.copy_(torch.from_numpy(z["pca_comps"]))
            self.proj_mean.copy_(torch.from_numpy(z["pca_mean"]))
            self.feat_mean.copy_(torch.from_numpy(z["feat_mean"]))
            self.W.copy_(torch.from_numpy(z["W"]))
            self.b.copy_(torch.from_numpy(z["b"]))

    def features(self, streams, *, rel_lo=0, n_take=None, dst_lo=0):
        """-> (n_trs, out_dim) centred design rows.

        ``streams`` maps modality -> (n_frames_of_segment, D). The window reads
        segment frames ``[rel_lo, rel_lo + n_take)`` into window frames starting
        at ``dst_lo``; other window frames are zero. ``None`` is a window past
        the end of the media (all frames zero).
        """
        if streams is None:
            x = self.proj.new_zeros(self.n_frames, self.n_mods, self.proj.shape[1])
        else:
            cols = []
            for name in self.mods:
                a = streams[name]
                a = a[rel_lo:rel_lo + (a.shape[0] if n_take is None else n_take)]
                if dst_lo or a.shape[0] < self.n_frames:
                    pad = a.new_zeros((self.n_frames, a.shape[1]))
                    pad[dst_lo:dst_lo + a.shape[0]] = a
                    a = pad
                cols.append(a)
            x = torch.stack(cols, dim=1).to(self.proj.dtype)
        # project each frame, then average: an empty lag block stays exactly zero
        p = torch.einsum("fmd,mdk->fmk", x - self.proj_mean, self.proj)
        lagged = self.lag_op @ p.reshape(self.n_frames, -1)
        return lagged.reshape(self.n_trs, -1) - self.feat_mean

    def forward(self, streams, subject_idx, **window):
        feat = self.features(streams, **window)
        idx = subject_idx.to(self.W.device).long()
        return torch.einsum("td,sdp->stp", feat, self.W[idx]) + self.b[idx][:, None, :]

    # -- closed-form re-fit: the prediction is linear in W over these rows ---
    def refit_design(self, streams, **window):
        return self.features(streams, **window) + self.feat_mean

    @property
    def refit_dim(self):
        return self.out_dim

    def install_refit(self, W, b, mx):
        """Install a ridge solution fitted on raw design rows centred at ``mx``."""
        b_eff = b + torch.einsum("sd,sdp->sp",
                                 self.feat_mean.to(W.device, W.dtype)[None] - mx, W)
        with torch.no_grad():
            self.W.copy_(W.to(self.W.device, self.W.dtype))
            self.b.copy_(b_eff.to(self.b.device, self.b.dtype))

    def param_groups(self, lr_readout, lr_proj):
        """No weight decay: W starts at a ridge optimum whose penalty is
        already chosen per parcel."""
        return [{"params": [self.W, self.b], "lr": lr_readout, "weight_decay": 0.0},
                {"params": [self.proj], "lr": lr_proj, "weight_decay": 0.0}]


class GatedMLPReadout(LinearReadout):
    """Linear readout plus ``gate * gelu(feat @ U + u_b) @ Wx_s``."""

    def __init__(self, blocks, k, hidden=512, gate_init=1.0, **kw):
        super().__init__(blocks, k, **kw)
        n_subjects, _, n_parcels = self.W.shape
        self.U = nn.Parameter(torch.randn(self.out_dim, hidden) / self.out_dim ** 0.5)
        self.u_b = nn.Parameter(torch.zeros(hidden))
        self.Wx = nn.Parameter(torch.randn(n_subjects, hidden, n_parcels) * 0.01)
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))

    def _branch(self, feat):
        return torch.nn.functional.gelu(feat @ self.U + self.u_b)

    def forward(self, streams, subject_idx, **window):
        feat = self.features(streams, **window)
        idx = subject_idx.to(self.W.device).long()
        base = torch.einsum("td,sdp->stp", feat, self.W[idx]) + self.b[idx][:, None, :]
        extra = torch.einsum("th,shp->stp", self._branch(feat), self.Wx[idx])
        return base + self.gate * extra

    # with the gate held fixed the prediction is linear in [W; Wx]
    def refit_design(self, streams, **window):
        feat = self.features(streams, **window)
        return torch.cat([feat + self.feat_mean, self.gate * self._branch(feat)], dim=1)

    @property
    def refit_dim(self):
        return self.out_dim + self.Wx.shape[1]

    def install_refit(self, W, b, mx):
        d = self.out_dim
        mean = torch.cat([self.feat_mean, self.feat_mean.new_zeros(self.Wx.shape[1])])
        b_eff = b + torch.einsum("sd,sdp->sp", mean.to(W.device, W.dtype)[None] - mx, W)
        with torch.no_grad():
            self.W.copy_(W[:, :d].to(self.W.device, self.W.dtype))
            self.Wx.copy_(W[:, d:].to(self.Wx.device, self.Wx.dtype))
            self.b.copy_(b_eff.to(self.b.device, self.b.dtype))

    def param_groups(self, lr_readout, lr_proj):
        return super().param_groups(lr_readout, lr_proj) + [
            {"params": [self.U, self.u_b, self.Wx, self.gate], "lr": lr_proj,
             "weight_decay": 0.0}]


class CrossAttnReadout(LinearReadout):
    """Per-TR cross-attention over the LM token sequence, in place of the lag
    average.

    Query ``q`` of modality ``m`` scores each token by content
    (``q_emb[m, q] . (v Wk[m])``) plus a learned bias over which lag block of
    the current TR the token falls in. At initialisation the bias is 0 inside
    the query's own block and ``bias_floor`` outside it, and the content logits
    are near zero, so each query averages exactly the tokens of its lag block
    and the linear readout's ridge solution applies unchanged. Attention is
    causal at each TR's centre; a query whose block holds no token outputs
    zero, as the lag operator does.
    """

    NEEDS_TOKENS = True
    PHI_KEY = "__phi__"

    def __init__(self, blocks, k, d_att=128, bias_floor=-12.0, **kw):
        super().__init__(blocks, k, **kw)
        n_b = len(self.blocks)
        self.n_q, self.d_att = n_b, int(d_att)
        self.register_buffer("blk_lo", torch.tensor([b[0] for b in self.blocks]))
        self.register_buffer("blk_hi", torch.tensor([b[0] + b[1] for b in self.blocks]))
        self.register_buffer("q_block", torch.arange(n_b))
        self.q_emb = nn.Parameter(torch.randn(self.n_mods, n_b, self.d_att)
                                  * (1e-2 / self.d_att ** 0.5))
        self.Wk = nn.Parameter(torch.randn(self.n_mods, self.k, self.d_att)
                               * (1e-2 / self.k ** 0.5))
        bias = torch.full((n_b, n_b + 1), float(bias_floor))   # last column: no block
        bias[torch.arange(n_b), torch.arange(n_b)] = 0.0
        self.bias = nn.Parameter(bias)

    def _lag_bounds(self):
        """Per (TR, block) frame bounds, identical to ``build_lag_operator``."""
        dev = self.W.device
        centre = WINDOW_OFFSET_FRAMES + FRAMES_PER_TR * torch.arange(
            self.n_trs, device=dev, dtype=torch.float32)
        lo = torch.ceil(centre[:, None] - self.blk_hi[None] * STEP_HZ)
        hi = torch.ceil(centre[:, None] - self.blk_lo[None] * STEP_HZ)
        return centre, lo.clamp(min=0), hi.clamp(max=float(self.n_frames))

    def _attend(self, v, phi, mi):
        """(n_tok, k) values at window frames ``phi`` -> (n_trs, n_q, k)."""
        centre, lo, hi = self._lag_bounds()
        inb = (phi[None, None] >= lo[..., None]) & (phi[None, None] < hi[..., None])
        occupied = (inb.any(-1) & (hi > lo))[:, self.q_block]       # (T, Q)
        n_b = len(self.blocks)
        which = torch.full((self.n_trs, phi.shape[0]), n_b, dtype=torch.long,
                           device=phi.device)
        for c in range(n_b):
            which = torch.where(inb[:, c], c, which)                 # (T, n_tok)
        logit = (self.q_emb[mi] @ (v @ self.Wk[mi]).T) / self.d_att ** 0.5
        logit = logit[None] + self.bias[None].expand(self.n_trs, -1, -1).gather(
            2, which[:, None, :].expand(-1, self.n_q, -1))
        causal = phi[None, :] <= centre[:, None]
        # finite: a TR may have no causal token of a stream (occupied is 0 there)
        logit = logit.masked_fill(~causal[:, None, :], -1e9)
        return torch.einsum("tqi,ik->tqk", logit.softmax(-1), v) * occupied[..., None]

    def features(self, streams, **window):
        """``streams`` maps modality -> (n_tok, D) and ``PHI_KEY`` -> the
        tokens' window-frame coordinates; the window placement is already
        folded into those coordinates."""
        if streams is None:
            return self.W.new_zeros(self.n_trs, self.out_dim) - self.feat_mean
        cols = []
        for mi, name in enumerate(self.mods):
            x, phi = streams[name], streams[self.PHI_KEY][name]
            if x.shape[0] == 0:
                cols.append(self.W.new_zeros(self.n_trs, self.n_q, self.k))
                continue
            v = (x.to(self.proj.dtype) - self.proj_mean[mi]) @ self.proj[mi]
            cols.append(self._attend(v, phi.to(v.device).float(), mi))
        out = torch.stack(cols, dim=2)                # (T, n_q, n_mods, k)
        return out.reshape(self.n_trs, self.out_dim) - self.feat_mean

    def param_groups(self, lr_readout, lr_proj):
        return super().param_groups(lr_readout, lr_proj) + [
            {"params": [self.q_emb, self.Wk, self.bias], "lr": lr_proj,
             "weight_decay": 0.0}]


READOUTS = {"linear": LinearReadout, "cross_attn": CrossAttnReadout,
            "gated_mlp": GatedMLPReadout}
