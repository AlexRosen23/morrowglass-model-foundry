"""Tests of the code the kit provides (config, checkpoints, data helpers, CLI). Always pass."""

from __future__ import annotations

import json

import pytest
import torch

from scratchlm.checkpoint import IntegrityError, load_checkpoint, save_checkpoint
from scratchlm.cli import build_parser, main
from scratchlm.data import load_text, split_text
from scratchlm.model import LanguageModel, ModelConfig, count_parameters
from tests.conftest import tiny_model


def test_kit_config_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        ModelConfig(d_model=30, n_head=4)
    with pytest.raises(ValueError):
        ModelConfig(n_layer=0)
    with pytest.raises(ValueError):
        ModelConfig(pos_encoding="rotary")  # type: ignore[arg-type]
    cfg = ModelConfig(d_model=64)
    assert cfg.ff == 256
    assert ModelConfig.from_dict({**cfg.to_dict(), "unknown": 1}) == cfg


def test_kit_tied_weights_are_one_tensor_counted_once() -> None:
    tied = LanguageModel(ModelConfig(block_size=8, n_layer=1, n_head=2, d_model=16))
    untied = LanguageModel(ModelConfig(block_size=8, n_layer=1, n_head=2, d_model=16, tie_weights=False))
    assert tied.lm_head.weight is tied.tok_emb.weight
    assert count_parameters(untied) - count_parameters(tied) == 256 * 16


def test_kit_checkpoint_roundtrip(tmp_path) -> None:
    m = tiny_model(seed=3)
    sha = save_checkpoint(tmp_path / "m.pt", m, {"step": 7})
    assert len(sha) == 64
    loaded, extra = load_checkpoint(tmp_path / "m.pt")
    assert extra == {"step": 7}
    assert loaded.cfg == m.cfg
    assert not loaded.training
    for (k1, v1), (k2, v2) in zip(m.state_dict().items(), loaded.state_dict().items(), strict=True):
        assert k1 == k2 and torch.equal(v1, v2)
    assert loaded.lm_head.weight is loaded.tok_emb.weight


def test_kit_checkpoint_refuses_a_changed_or_unhashed_file(tmp_path) -> None:
    path = tmp_path / "m.pt"
    save_checkpoint(path, tiny_model())
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(IntegrityError):
        load_checkpoint(path)
    (tmp_path / "m.pt.sha256.json").unlink()
    with pytest.raises(IntegrityError):
        load_checkpoint(path)


def test_kit_split_text_cuts_at_a_line_and_keeps_everything(tmp_path) -> None:
    text = "".join(f"line {i}\n" for i in range(100))
    train, val = split_text(text, 0.1)
    assert train + val == text
    assert train.endswith("\n") and 0 < len(val) < len(text) * 0.2
    with pytest.raises(ValueError):
        split_text(text, 0.0)
    p = tmp_path / "t.txt"
    p.write_bytes(b"a\r\nb\r\n")
    assert load_text(p) == "a\nb\n"


def test_kit_cli_info_and_parser(capsys) -> None:
    assert main(["info", "--n-layer", "1", "--d-model", "32", "--n-head", "2", "--block-size", "16"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["parameters"] > 0 and info["config"]["d_model"] == 32
    a = build_parser().parse_args(["sample", "--ckpt", "x.pt", "--kv-cache"])
    assert a.kv_cache and a.device == "cpu" and a.top_k == 40
