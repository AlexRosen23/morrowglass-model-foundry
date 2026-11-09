"""LoRA plumbing on a locally constructed random tiny GPT-2 (no download). Skipped without the hf extra."""

from __future__ import annotations

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from morrowglass.finetune.lora import eval_bpb, lora_wrap, run_lora, train_lora, trainable_fraction  # noqa: E402


def _tiny_gpt2() -> torch.nn.Module:
    cfg = transformers.GPT2Config(vocab_size=256, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(cfg)


def test_lora_trains_only_adapters_and_reduces_loss() -> None:
    model = lora_wrap(_tiny_gpt2(), r=4, alpha=8, dropout=0.0)
    trainable, total = trainable_fraction(model)
    assert 0 < trainable < 0.2 * total
    before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    ids = np.frombuffer(b"abcabcabcabc" * 200, dtype=np.uint8).astype(np.int64)
    losses = train_lora(model, ids, block=32, steps=40, batch=4, lr=5e-3, device="cpu", seed=0)
    assert losses[-1] < losses[0]
    for n, p in model.named_parameters():
        if not p.requires_grad:
            torch.testing.assert_close(p, before[n], rtol=0, atol=0, msg=n)


def test_eval_bpb_is_finite_and_uses_byte_lengths() -> None:
    model = _tiny_gpt2()
    ids = np.arange(200) % 256
    bpb = eval_bpb(model, ids, np.ones(256, dtype=np.int64), block=32, n_windows=3, device="cpu")
    assert 0 < bpb < 20


def test_real_run_requires_approval() -> None:
    with pytest.raises(PermissionError, match="owner agrees"):
        run_lora("distilgpt2", approved=False)
