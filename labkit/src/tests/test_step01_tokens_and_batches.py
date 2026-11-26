"""Step 1: the byte tokenizer and training batches."""

from __future__ import annotations

import pytest
import torch

from scratchlm.data import sample_batch
from scratchlm.tokenizer import ByteTokenizer


def test_step01_encode_gives_utf8_bytes() -> None:
    tok = ByteTokenizer()
    assert tok.encode("Hi!") == [72, 105, 33]
    assert tok.encode("é") == [0xC3, 0xA9]  # one character, two bytes
    assert tok.encode("") == []
    assert all(0 <= i < tok.vocab_size for i in tok.encode("To be, or not to be ✓ 🙂"))


def test_step01_decode_roundtrips_and_tolerates_broken_bytes() -> None:
    tok = ByteTokenizer()
    for text in ["ROMEO:\nBut, soft!", "naïve café", "🙂 emoji", "tabs\tand\r\nline ends", ""]:
        assert tok.decode(tok.encode(text)) == text
    assert tok.decode([0xC3]) == "�"  # half a character: replaced, not an exception


def test_step01_targets_are_the_inputs_shifted_by_one(gen) -> None:
    ids = torch.arange(100, dtype=torch.long)
    x, y = sample_batch(ids, block_size=8, batch_size=4, gen=gen)
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == y.dtype == torch.long
    assert torch.equal(y, x + 1)  # with ids = 0..99, the next token is always one more
    assert torch.equal(x[:, 1:], x[:, :-1] + 1)  # each row is a contiguous window
    assert int(y.max()) <= 99


def test_step01_batches_are_random_but_reproducible() -> None:
    ids = torch.arange(1000, dtype=torch.long)
    a = sample_batch(ids, 16, 8, torch.Generator().manual_seed(0))[0]
    b = sample_batch(ids, 16, 8, torch.Generator().manual_seed(0))[0]
    c = sample_batch(ids, 16, 8, torch.Generator().manual_seed(1))[0]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert len(set(a[:, 0].tolist())) > 1  # not every window starts at the same place


def test_step01_every_start_position_is_reachable() -> None:
    ids = torch.arange(10, dtype=torch.long)
    x, y = sample_batch(ids, block_size=8, batch_size=200, gen=torch.Generator().manual_seed(0))
    assert set(x[:, 0].tolist()) == {0, 1}  # 10 tokens hold two windows of 8 plus a next token
    with pytest.raises(ValueError):
        sample_batch(ids[:8], block_size=8, batch_size=1, gen=torch.Generator())
