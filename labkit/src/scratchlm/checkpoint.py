"""Saving and loading models (given; not an exercise).

A checkpoint is one ``torch.save`` file plus a small ``.sha256.json`` sidecar. Loading checks the hash
first, so a truncated or edited file is refused instead of producing a quietly broken model.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import torch

from scratchlm.model import LanguageModel, ModelConfig


class IntegrityError(Exception):
    """The checkpoint does not match its recorded sha256."""


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256.json")


def save_checkpoint(path: str | Path, model: LanguageModel, extra: dict[str, Any] | None = None) -> str:
    """Write the model's weights and config to ``path``; returns the file's sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model_config": model.cfg.to_dict(), "model": model.state_dict(), "extra": extra or {}}
    buf = io.BytesIO()
    torch.save(state, buf)
    data = buf.getvalue()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    sha = hashlib.sha256(data).hexdigest()
    _sidecar(path).write_text(json.dumps({"sha256": sha, "bytes": len(data)}, indent=2) + "\n")
    return sha


def load_checkpoint(path: str | Path) -> tuple[LanguageModel, dict[str, Any]]:
    """Return ``(model in eval mode, extra)`` after checking the file against its sidecar hash."""
    path = Path(path)
    side = _sidecar(path)
    if not side.exists():
        raise IntegrityError(f"{path}: no {side.name} next to it, so its hash cannot be checked")
    data = path.read_bytes()
    want = json.loads(side.read_text())["sha256"]
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise IntegrityError(f"{path}: sha256 {got[:12]}... does not match the recorded {want[:12]}...")
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    model = LanguageModel(ModelConfig.from_dict(state["model_config"]))
    model.load_state_dict(state["model"])
    model.eval()
    return model, state.get("extra", {})
