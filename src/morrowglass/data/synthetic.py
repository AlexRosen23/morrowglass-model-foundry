"""Seeded synthetic algorithmic task with length-shift splits: digit-sum modulo 10.

Example: ``"3+9+4=6"`` (3 + 9 + 4 = 16, and 16 mod 10 = 6). The model reads the
left side and must predict the single answer digit after ``=``.

Splits:
- ``train``  : 3-6 terms
- ``val_iid``: 3-6 terms, strings never seen in train (in-distribution check)
- ``test_ood_near``: 8-10 terms (longer than anything trained)
- ``test_ood_far`` : 12-16 terms

Every expression string is unique across all splits (deduplicated against a global set),
so no template can leak from train to evaluation. Success on this task says nothing
about general reasoning. It only tests whether a tiny model learned this one rule, and
whether that rule still works on longer inputs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch

from morrowglass.repro import sha256_bytes

SPLITS: dict[str, tuple[int, int, int]] = {
    # name: (min_terms, max_terms, n_examples)
    "train": (3, 6, 60_000),
    "val_iid": (3, 6, 2_000),
    "test_ood_near": (8, 10, 2_000),
    "test_ood_far": (12, 16, 2_000),
}
MOD = 10
SEED = 4242
VOCAB = "0123456789+=\n"  # encoded as bytes by the byte tokenizer


@dataclass(frozen=True)
class Example:
    expr: str  # "3+9+4"
    answer: str  # "6"

    @property
    def text(self) -> str:
        return f"{self.expr}={self.answer}\n"


def _make(rng: random.Random, n_terms: int) -> Example:
    digits = [rng.randrange(10) for _ in range(n_terms)]
    return Example("+".join(map(str, digits)), str(sum(digits) % MOD))


def generate(seed: int = SEED, splits: dict[str, tuple[int, int, int]] | None = None) -> dict[str, list[Example]]:
    splits = splits or SPLITS
    rng = random.Random(seed)
    seen: set[str] = set()
    out: dict[str, list[Example]] = {}
    # Evaluation splits are drawn first so train can never contain their strings.
    order = [s for s in splits if s != "train"] + ["train"]
    for name in order:
        lo, hi, n = splits[name]
        exs: list[Example] = []
        attempts = 0
        while len(exs) < n:
            attempts += 1
            if attempts > n * 50:
                raise RuntimeError(f"cannot draw {n} unique examples for {name}")
            ex = _make(rng, rng.randint(lo, hi))
            if ex.expr in seen:
                continue
            seen.add(ex.expr)
            exs.append(ex)
        out[name] = exs
    return {k: out[k] for k in splits}


def split_hash(examples: list[Example]) -> str:
    return sha256_bytes("".join(e.text for e in examples).encode())


def encode_examples(examples: list[Example], block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Byte-encode and right-pad; targets are -100 everywhere except the answer digit."""
    n = len(examples)
    x = torch.full((n, block_size), ord("\n"), dtype=torch.long)
    y = torch.full((n, block_size), -100, dtype=torch.long)
    for i, ex in enumerate(examples):
        b = list(ex.text.encode())
        if len(b) > block_size + 1:
            raise ValueError(f"example longer than block_size: {ex.text!r}")
        inp = b[:-1]
        x[i, : len(inp)] = torch.tensor(inp)
        ans_pos = len(ex.expr)  # position of '=' predicts the answer digit
        y[i, ans_pos] = b[ans_pos + 1]
    return x, y


def describe() -> dict[str, Any]:
    return {"task": "digit-sum mod 10", "mod": MOD, "seed": SEED,
            "splits": {k: {"min_terms": v[0], "max_terms": v[1], "n": v[2]} for k, v in SPLITS.items()}}
