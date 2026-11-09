"""Hugging Face Hub-compatible export: save -> from_pretrained gives identical logits; card is filled; no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from conftest import tiny_cfg
from morrowglass.hub import MorrowglassHubLM, export_hf, publish_command
from morrowglass.model.transformer import ModelConfig
from morrowglass.train.trainer import load_model_from_checkpoint, train_from_config


@torch.no_grad()
def test_save_pretrained_roundtrip_identical_logits(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, d_model=32, pos_encoding="sinusoidal")
    m = MorrowglassHubLM(cfg).eval()
    m.save_pretrained(tmp_path / "m")
    assert {p.name for p in (tmp_path / "m").iterdir()} >= {"model.safetensors", "config.json", "README.md"}
    assert json.loads((tmp_path / "m" / "config.json").read_text())["pos_encoding"] == "sinusoidal"
    r = MorrowglassHubLM.from_pretrained(tmp_path / "m")
    assert r.cfg == cfg
    x = torch.randint(0, 256, (2, 32))
    assert torch.equal(r(x).logits, m(x).logits)
    assert r.lm_head.weight.data_ptr() == r.tok_emb.weight.data_ptr()  # tied weights survive safetensors


def test_export_hf_from_a_real_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    def no_network(*a: object, **k: object) -> None:
        raise AssertionError("export must not touch the network")

    monkeypatch.setattr(huggingface_hub.HfApi, "upload_folder", no_network)
    monkeypatch.setattr(huggingface_hub.HfApi, "create_repo", no_network)
    runs, ck = tmp_path / "runs", tmp_path / "ckpt"
    s = train_from_config(tiny_cfg(max_steps=10, experiment="unit"), quiet=True, runs_dir=runs, ckpt_root=ck)
    rid = s["run_id"]
    info = export_hf(rid, out_root=tmp_path / "hf", card_root=tmp_path / "cards", runs_dir=runs, ckpt_root=ck)
    assert info["roundtrip_max_abs_logit_diff"] == 0.0 and info["published"] is False
    assert "hf upload" in info["publish_command_after_owner_approval"] and "--private" in publish_command(rid)
    base, _ = load_model_from_checkpoint(ck / rid / "best.pt")
    re = MorrowglassHubLM.from_pretrained(tmp_path / "hf" / rid)
    x = torch.randint(0, 256, (1, 32))
    with torch.no_grad():
        assert torch.equal(re(x).logits, base(x).logits)
    card = (tmp_path / "cards" / rid / "README.md").read_text()
    assert card.startswith("---\nlicense: mit")
    for section in ("## Intended use", "## Limitations", "## Data", "## Evaluation", "Not instruction-tuned"):
        assert section in card
    assert f"{s['final']['val']['bits_per_byte']:.3f}" in card  # metric copied from the run record
    assert "model.safetensors" in info["files"]
