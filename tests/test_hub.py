import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from brain_lora.hub import export_run, import_model
from brain_lora.training import Config, build_readout

BLOCKS = [(0.5, 2), (2.5, 2), (4.5, 4), (8.5, 8), (16.5, 16), (32.5, 32), (64.5, 64),
          (128.5, 64)]
D, K = 32, 4
LORA = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"]


@pytest.fixture(params=["linear", "cross_attn"])
def run_dir(request, tmp_path):
    """A trained run: config, init, and a checkpoint with moved weights."""
    rng = np.random.default_rng(0)
    n_cols = len(BLOCKS) * 3 * K
    init = tmp_path / "init.npz"
    np.savez(init, blocks=np.array(BLOCKS, np.float32), k_use=K,
             subjects=np.array(["sub-01", "sub-02"]), mods=np.array(["text", "audio", "vision"]),
             pca_mean=rng.normal(size=(3, D)).astype(np.float32),
             pca_comps=rng.normal(size=(3, D, K)).astype(np.float32),
             feat_mean=rng.normal(size=n_cols).astype(np.float32),
             W=rng.normal(size=(2, n_cols, 5)).astype(np.float32),
             b=rng.normal(size=(2, 5)).astype(np.float32), report="{}")
    run = tmp_path / "run"
    cfg = Config(data_dir="data", readout_init=str(init), readout=request.param)
    cfg.save(run)
    readout = build_readout(cfg)
    with torch.no_grad():
        for p in readout.parameters():
            p.add_(torch.randn_like(p))
    torch.save({"encoder_trainable": {k: torch.randn(16, 8) for k in LORA},
                "readout": readout.state_dict(), "step": 1429,
                "optimizer": {"state": {}}, "scheduler": {}}, run / "refit_last.pt")
    return run


def test_export_then_import_restores_the_run(run_dir, tmp_path):
    export_run(run_dir, tmp_path / "hub")
    cfg = import_model(tmp_path / "hub", tmp_path / "back", "elsewhere")
    old = torch.load(run_dir / "refit_last.pt", weights_only=False)
    new = torch.load(tmp_path / "back" / "refit_last.pt", weights_only=False)
    assert set(new) == {"encoder_trainable", "readout", "step"}      # no optimizer state
    for group in ("encoder_trainable", "readout"):
        assert new[group].keys() == old[group].keys()
        assert all(torch.equal(new[group][k], old[group][k]) for k in old[group])
    # the imported config rebuilds the readout without the init file
    assert cfg.data_dir == "elsewhere" and cfg.readout_init == ""
    assert Config.load(tmp_path / "back") == cfg
    build_readout(cfg).load_state_dict(new["readout"])
    trained = Config.load(run_dir)
    for f in ("readout", "rank", "tap_layer", "context_lead_s", "use_audio", "max_steps"):
        assert getattr(cfg, f) == getattr(trained, f)


def test_adapter_is_in_peft_format(run_dir, tmp_path):
    export_run(run_dir, tmp_path / "hub")
    keys = load_file(tmp_path / "hub" / "adapter_model.safetensors").keys()
    assert sorted(keys) == sorted(k.replace(".default.", ".") for k in LORA)
    adapter = json.loads((tmp_path / "hub" / "adapter_config.json").read_text())
    assert (adapter["r"], adapter["lora_alpha"], adapter["peft_type"]) == (16, 32, "LORA")
    assert adapter["base_model_name_or_path"] == "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    config = json.loads((tmp_path / "hub" / "config.json").read_text())
    assert "data_dir" not in config and "readout_init" not in config


def test_wrong_lag_blocks_are_rejected(run_dir, tmp_path):
    ckpt = torch.load(run_dir / "refit_last.pt", weights_only=False)
    ckpt["readout"]["lag_op"] = ckpt["readout"]["lag_op"].flip(0)
    torch.save(ckpt, run_dir / "refit_last.pt")
    with pytest.raises(RuntimeError, match="lag blocks"):
        export_run(run_dir, tmp_path / "hub")
