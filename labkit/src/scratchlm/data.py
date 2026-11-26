"""Text files, train/validation splits and training batches (step 1)."""

from __future__ import annotations

from pathlib import Path

import torch

from scratchlm.tokenizer import ByteTokenizer


def load_text(path: str | Path) -> str:
    """Read a UTF-8 text file, with Windows line endings normalised to "\\n"."""
    return Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")


def split_text(text: str, val_fraction: float = 0.1) -> tuple[str, str]:
    """Split one text into (train, validation) at a line break near ``1 - val_fraction``.

    The validation part is the *end* of the text, never a random sample of lines: the model must be
    scored on text it has not seen, and neighbouring lines are too similar to count as unseen.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    cut = int(len(text) * (1.0 - val_fraction))
    nl = text.rfind("\n", 0, cut)
    if nl > 0:
        cut = nl + 1
    return text[:cut], text[cut:]


def encode_file(path: str | Path, tok: ByteTokenizer | None = None) -> torch.Tensor:
    """The whole file as a 1-D tensor of token ids (int64)."""
    tok = tok or ByteTokenizer()
    return torch.tensor(tok.encode(load_text(path)), dtype=torch.long)


def sample_batch(
    ids: torch.Tensor, block_size: int, batch_size: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick ``batch_size`` random windows of ``block_size`` tokens from the 1-D tensor ``ids``.

    Returns ``(x, y)``, both int64 of shape (batch_size, block_size), where ``y`` is ``x`` shifted left
    by one: ``y[b, t]`` is the token that follows ``x[b, t]`` in the text. Raises ValueError if ``ids``
    is too short to hold one window plus its next token.
    """
    #lab:begin 1 random windows and their next-token targets
    n_starts = len(ids) - block_size
    if n_starts <= 0:
        raise ValueError(f"{len(ids)} tokens are too few for block_size {block_size}")
    starts = torch.randint(0, n_starts, (batch_size,), generator=gen)
    offsets = torch.arange(block_size)
    x = ids[starts[:, None] + offsets]
    y = ids[starts[:, None] + offsets + 1]
    return x, y
    #lab:stub
    #> # TODO(step 1): choose `batch_size` random start positions with torch.randint(..., generator=gen),
    #> # cut a window of `block_size` tokens at each start for x, and the same window moved one token to
    #> # the right for y. Raise ValueError when `ids` cannot hold block_size + 1 tokens.
    #> raise NotImplementedError("TODO(step 1): sample_batch")
    #lab:end
