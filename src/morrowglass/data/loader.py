"""Token streams and batch sampling for language-model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from morrowglass.data.corpus import load_split
from morrowglass.repro import REPO_ROOT, sha256_bytes, write_verified
from morrowglass.tokenizer.tokenizers import Tokenizer

TOKEN_CACHE = REPO_ROOT / "artifacts" / "tokens"


@dataclass
class TokenSplit:
    ids: np.ndarray  # (N,) int64 token ids
    byte_len: np.ndarray  # (vocab,) UTF-8 byte length of each token id
    n_bytes: int  # bytes of the underlying text
    sha256: str  # hash of the token id array

    def __len__(self) -> int:
        return int(self.ids.shape[0])


def token_byte_lengths(tok: Tokenizer) -> np.ndarray:
    return np.array([len(tok.token_bytes(i)) for i in range(tok.vocab_size)], dtype=np.int64)


def encode_split(dataset: str, split: str, tok: Tokenizer, cache: Path = TOKEN_CACHE) -> TokenSplit:
    text = load_split(dataset, split)
    key = f"{dataset}/{tok.name}-{tok.fingerprint()[:12]}/{split}-{sha256_bytes(text.encode())[:12]}.npy"
    path = cache / key
    if path.exists():
        ids = np.load(path)
    else:
        ids = np.asarray(tok.encode(text), dtype=np.int64)
        import io

        buf = io.BytesIO()
        np.save(buf, ids)
        write_verified(path, buf.getvalue())
    blen = token_byte_lengths(tok)
    n_bytes = len(text.encode("utf-8"))
    if int(blen[ids].sum()) != n_bytes:
        raise AssertionError(f"{split}: token bytes {int(blen[ids].sum())} != text bytes {n_bytes}")
    return TokenSplit(ids=ids, byte_len=blen, n_bytes=n_bytes, sha256=sha256_bytes(ids.tobytes()))


def sample_batch(
    ids: np.ndarray, block_size: int, batch_size: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random contiguous windows. Returns x, y of shape (B, T); y is x shifted by one."""
    n = len(ids) - block_size - 1
    if n <= 0:
        raise ValueError(f"split of {len(ids)} tokens is too short for block_size {block_size}")
    starts = torch.randint(0, n, (batch_size,), generator=gen).numpy()
    x = np.stack([ids[s : s + block_size] for s in starts])
    y = np.stack([ids[s + 1 : s + 1 + block_size] for s in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


def eval_windows(n_tokens: int, block_size: int, max_windows: int | None = None) -> list[tuple[int, int]]:
    """Non-overlapping (start, length) windows that predict every token after the first exactly once.

    With ``max_windows`` the windows are an evenly spaced deterministic subset (for fast
    in-training evaluation). Final metrics always use every window.
    """
    wins = []
    s = 0
    while s < n_tokens - 1:
        length = min(block_size, n_tokens - 1 - s)
        wins.append((s, length))
        s += length
    if max_windows is not None and len(wins) > max_windows:
        idx = np.linspace(0, len(wins) - 1, max_windows).round().astype(int)
        wins = [wins[i] for i in sorted(set(idx.tolist()))]
    return wins
