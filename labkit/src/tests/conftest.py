"""Shared test helpers. Every test runs on the CPU with tiny models and finishes in seconds."""

from __future__ import annotations

import pytest
import torch

from scratchlm.model import LanguageModel, ModelConfig

torch.set_num_threads(2)  # stay polite on a shared machine

PATTERN = "the cat sat on the mat. " * 40  # a tiny, very learnable "corpus"


def tiny_model(seed: int = 0, **kw: object) -> LanguageModel:
    base: dict[str, object] = dict(vocab_size=256, block_size=16, n_layer=2, n_head=2, d_model=32)
    base.update(kw)
    torch.manual_seed(seed)
    return LanguageModel(ModelConfig(**base))  # type: ignore[arg-type]


def pattern_ids(text: str = PATTERN) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


@pytest.fixture
def gen() -> torch.Generator:
    return torch.Generator().manual_seed(1234)
