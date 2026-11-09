"""Reproducibility helpers: seeds, hashes, git provenance and verified atomic writes.

The development Mac has a known memory fault that occasionally flips bits in buffered
writes, so every artifact that matters (checkpoints, manifests) is written through
``write_verified`` which hashes the in-memory bytes, writes + fsyncs a temp file, reads it
back, and only renames it into place when the hashes match.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


class IntegrityError(RuntimeError):
    """Raised when an artifact's bytes do not match the recorded hash."""


def set_seed(seed: int) -> None:
    """Seed every RNG the project uses (Python, NumPy, torch CPU/MPS)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(obj: Any) -> str:
    """Stable short-form hash of a JSON-serialisable config."""
    return sha256_bytes(canonical_json(obj).encode())


def write_verified(path: str | Path, data: bytes, retries: int = 2) -> str:
    """Write ``data`` atomically and verify it by reading it back. Returns the sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256_bytes(data)
    tmp = path.with_name(path.name + ".tmp")
    for _attempt in range(retries + 1):
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if sha256_file(tmp) == expected:
            os.replace(tmp, path)
            return expected
    tmp.unlink(missing_ok=True)
    raise IntegrityError(f"read-back hash mismatch writing {path} after {retries + 1} attempts")


def write_json(path: str | Path, obj: Any) -> str:
    return write_verified(path, (json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n").encode())


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify_file(path: str | Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise IntegrityError(f"{path}: sha256 {actual[:12]}… does not match recorded {expected_sha256[:12]}…")


def git_provenance(cwd: Path = REPO_ROOT) -> dict[str, Any]:
    """Current commit and whether the working tree had uncommitted changes."""

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False, timeout=10
        ).stdout.strip()

    commit = _git("rev-parse", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    return {"commit": commit, "dirty": dirty, "branch": branch}


def now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
