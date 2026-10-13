"""Qwen3-Omni thinker with LoRA, read at one decoder layer.

The decoder is truncated at the tapped layer (layers above it neither run nor
receive adapters). LoRA (rank r, alpha 2r) is attached to q/k/v/o of every kept
decoder layer; the audio and vision towers are frozen and run without grad.
Decoder layers use non-reentrant gradient checkpointing, which keeps the tap's
gathered hidden states in the autograd graph.

Layer indexing follows the HF ``hidden_states`` convention: layer ``i`` is the
output of decoder block ``i - 1`` (0 would be the embeddings).
"""

import logging

import torch

from brain_lora.constants import MODEL_ID
from brain_lora.model.inputs import gather_token_rows, token_keep_positions

log = logging.getLogger(__name__)

LORA_TARGETS = r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"


def patch_vision_conv_fp32():
    """Run the vision patch-embed Conv3d in float32.

    cuDNN 9 executes this bf16 Conv3d shape several orders of magnitude slower
    than float32; the result is cast back to the model dtype.
    """
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeVisionPatchEmbed as PatchEmbed,
    )

    if getattr(PatchEmbed, "_fp32_conv", False):
        return

    def forward(self, hidden_states):
        dtype = self.proj.weight.dtype
        x = hidden_states.view(-1, self.in_channels, self.temporal_patch_size,
                               self.patch_size, self.patch_size)
        p = self.proj
        out = torch.nn.functional.conv3d(
            x.float(), p.weight.float(), None if p.bias is None else p.bias.float(),
            stride=p.stride, padding=p.padding, dilation=p.dilation, groups=p.groups)
        return out.to(dtype).view(-1, self.embed_dim)

    PatchEmbed.forward = forward
    PatchEmbed._fp32_conv = True


class _NoLMHead(torch.nn.Module):
    """Stands in for the vocabulary projection, whose logits are never read."""

    def forward(self, hidden_states):
        return hidden_states[:, :1, :1]


def _one_video_block_at_a_time(model):
    """Run the vision tower one media block at a time (bounds peak memory;
    blocks are independent, so the result is unchanged)."""
    orig = model.get_video_features

    def chunked(pixel_values_videos, video_grid_thw=None):
        if video_grid_thw is None or len(video_grid_thw) <= 1:
            return orig(pixel_values_videos, video_grid_thw)
        embeds, deepstack, off = [], None, 0
        for g in video_grid_thw.split(1):
            n = int(g.prod(-1).sum())
            e, ms = orig(pixel_values_videos[off:off + n], g)
            off += n
            embeds.append(e)
            deepstack = [[t] for t in ms] if deepstack is None else \
                [acc + [t] for acc, t in zip(deepstack, ms)]
        return torch.cat(embeds), tuple(torch.cat(p) for p in deepstack)

    model.get_video_features = chunked


def _freeze_towers(model):
    for name in ("get_video_features", "get_audio_features"):
        fn = getattr(model, name)

        def no_grad(*a, _fn=fn, **kw):
            with torch.no_grad():
                out = _fn(*a, **kw)
            if isinstance(out, tuple):
                return tuple(o.detach() if torch.is_tensor(o) else o for o in out)
            return out.detach()

        setattr(model, name, no_grad)


class _LayerTap:
    """Captures one layer's output at selected positions, inside the graph.

    The hook fires again during checkpoint recomputation in backward; only the
    first capture after ``arm`` is used, and ``release`` clears it after the
    optimiser step.
    """

    def __init__(self, layer):
        self.keep, self.out, self.armed = None, [], False
        layer.register_forward_hook(self._hook)

    def arm(self, keep):
        self.keep, self.out, self.armed = keep, [], True

    def _hook(self, mod, inp, out):
        if self.armed:
            h = out[0] if isinstance(out, tuple) else out
            self.out.append(h[0, self.keep.to(h.device), :])

    def get(self):
        if not self.out:
            raise RuntimeError("the tap captured nothing")
        return self.out[0]

    def release(self):
        self.armed, self.out = False, []


class OmniEncoder:
    """Encodes one span of stimulus and returns the tapped hidden states.

    ``rank=None`` builds the frozen backbone without adapters (feature
    extraction).
    """

    def __init__(self, *, tap_layer=32, rank=16, alpha=None, use_audio=True,
                 device="cuda:0", model_id=MODEL_ID):
        import transformers

        self.device = device
        self.tap_layer = int(tap_layer)
        self.use_audio = bool(use_audio)
        self.proc = transformers.AutoProcessor.from_pretrained(model_id)
        patch_vision_conv_fp32()
        model = transformers.Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to(device)
        model.lm_head = _NoLMHead()
        _one_video_block_at_a_time(model)
        _freeze_towers(model)
        model.requires_grad_(False)

        n_layers = len(model.model.layers)
        if not 1 <= self.tap_layer <= n_layers:
            raise ValueError(f"tap_layer must be in [1, {n_layers}]")
        model.model.layers = model.model.layers[:self.tap_layer]
        self.thinker = model
        if rank is not None:
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(model, LoraConfig(
                r=rank, lora_alpha=2 * rank if alpha is None else alpha,
                lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none"))
            n_adapted = sum("lora_" in n for n, _ in model.named_parameters())
            if n_adapted != self.tap_layer * 4 * 2:
                raise RuntimeError(f"LoRA attached to {n_adapted} tensors")
            self.thinker.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        self.model = model
        self.tap = _LayerTap(self.thinker.model.layers[-1])
        self.lora_params = [p for p in model.parameters() if p.requires_grad]
        log.info("Qwen3-Omni truncated at layer %d; %.2fM adapter parameters",
                 self.tap_layer, sum(p.numel() for p in self.lora_params) / 1e6)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def _forward(self, inputs, keep):
        gpu = {k: v.to(self.device) if torch.is_tensor(v) else v
               for k, v in inputs.items()}
        has_audio = "input_features" in gpu
        if has_audio != self.use_audio:
            raise RuntimeError(f"encoder built with use_audio={self.use_audio} "
                               f"but the inputs {'carry' if has_audio else 'lack'} audio")
        if has_audio:
            gpu["input_features"] = gpu["input_features"].to(torch.bfloat16)
        gpu["pixel_values_videos"] = gpu["pixel_values_videos"].to(torch.bfloat16)
        self.tap.arm(keep)
        self.model(**gpu, use_audio_in_video=has_audio, use_cache=False,
                   return_dict=True)
        return self.tap.get()

    def encode(self, inputs, grid, n_keep):
        """-> {modality: (n_keep, D)} on the 2 Hz grid of the last ``n_keep``
        steps of the span; steps before a stream's first token are zero."""
        parts = [grid[m] for m in ("audio", "vision", "text") if grid[m] is not None]
        keep = torch.cat(parts).unique().sort().values
        h = self._forward(inputs, keep)
        lut = torch.full((int(inputs["input_ids"].shape[1]),), -1, dtype=torch.long)
        lut[keep] = torch.arange(len(keep))
        n_span = grid["n_out"]
        out = {}
        for name in ("audio", "vision", "text"):
            if grid[name] is None:
                out[name] = h.new_zeros((n_keep, h.shape[-1]))
                continue
            valid = torch.from_numpy(grid["valid"][name][-n_keep:].copy()).to(h.device)
            rows = h.index_select(0, lut[grid[name].cpu()].to(h.device))
            out[name] = rows[n_span - n_keep:] * valid[:, None]
        return out

    def encode_tokens(self, inputs, tokens):
        """-> {modality: (n_tok, D)} for a ``build_token_index`` index."""
        keep = token_keep_positions(tokens)
        h = self._forward(inputs, keep)
        return gather_token_rows(h, keep, tokens, int(inputs["input_ids"].shape[1]))
