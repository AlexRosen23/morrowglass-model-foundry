"""Byte-level and byte-level-BPE tokenizers behind one small interface.

Both are lossless over arbitrary UTF-8 (no <unk>), which makes bits-per-byte an exact,
tokenizer-independent metric: total negative log-likelihood (in bits) divided by the
number of UTF-8 bytes of the text being predicted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

from morrowglass.repro import REPO_ROOT, sha256_bytes, write_verified

TOKENIZER_DIR = REPO_ROOT / "artifacts" / "tokenizers"


class Tokenizer(Protocol):
    name: str
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def token_bytes(self, token_id: int) -> bytes: ...

    def fingerprint(self) -> str: ...


class ByteTokenizer:
    """One token per UTF-8 byte (vocab 256)."""

    name = "byte"
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        return bytes([token_id])

    def fingerprint(self) -> str:
        return sha256_bytes(b"byte-tokenizer-v1")


class BPETokenizer:
    """Byte-level BPE (GPT-2 style) trained with Hugging Face ``tokenizers`` on train text only."""

    def __init__(self, hf: HFTokenizer, name: str) -> None:
        self._tok = hf
        self.name = name
        self.vocab_size = hf.get_vocab_size()
        # GPT-2 byte<->unicode table used by the ByteLevel pre-tokenizer.
        self._u2b = {v: k for k, v in _bytes_to_unicode().items()}
        self._id_bytes: list[bytes] = []
        for i in range(self.vocab_size):
            tok = hf.id_to_token(i) or ""
            self._id_bytes.append(bytes(self._u2b[c] for c in tok))

    @classmethod
    def train(cls, text: str, vocab_size: int, name: str | None = None) -> BPETokenizer:
        tok = HFTokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],
            show_progress=False,
        )
        tok.train_from_iterator(_chunks(text), trainer=trainer, length=None)
        return cls(tok, name or f"bpe{vocab_size}")

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        return cls(HFTokenizer.from_file(str(path)), path.stem)

    def save(self, path: Path) -> str:
        return write_verified(path, self._tok.to_str(pretty=False).encode())

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text).ids)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._id_bytes[i] for i in ids).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        return self._id_bytes[token_id]

    def fingerprint(self) -> str:
        return sha256_bytes(self._tok.to_str(pretty=False).encode())


def _chunks(text: str, size: int = 1 << 16) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs], strict=True))


def get_tokenizer(spec: str, dataset: str, train_text: str | None = None) -> Tokenizer:
    """``spec`` is 'byte' or 'bpe<N>'. BPE files are cached per dataset and trained on train only."""
    if spec == "byte":
        return ByteTokenizer()
    if spec.startswith("bpe"):
        vocab = int(spec[3:])
        path = TOKENIZER_DIR / dataset / f"{spec}.json"
        if path.exists():
            return BPETokenizer.load(path)
        if train_text is None:
            from morrowglass.data.corpus import load_split

            train_text = load_split(dataset, "train")
        tok = BPETokenizer.train(train_text, vocab, name=spec)
        tok.save(path)
        meta = {"spec": spec, "dataset": dataset, "trained_on": "train split only",
                "vocab_size": tok.vocab_size, "sha256": tok.fingerprint()}
        write_verified(path.with_suffix(".meta.json"), json.dumps(meta, indent=2).encode())
        return tok
    raise ValueError(f"unknown tokenizer spec {spec!r}")
