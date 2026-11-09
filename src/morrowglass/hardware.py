"""Detect the hardware we are actually running on (no CUDA assumptions)."""

from __future__ import annotations

import os
import platform
import resource
import subprocess
import sys
from typing import Any, Literal

import torch

Device = Literal["cpu", "mps"]


def _sysctl(name: str) -> str | None:
    if sys.platform != "darwin":
        return None
    out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False)
    return out.stdout.strip() or None


def total_ram_bytes() -> int | None:
    v = _sysctl("hw.memsize")
    if v and v.isdigit():
        return int(v)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def mps_usable() -> bool:
    """True only if MPS is built, available, and a tiny op actually runs."""
    if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        return False
    try:
        x = torch.ones(4, device="mps")
        return bool((x * 2).sum().item() == 8.0)
    except Exception:  # pragma: no cover - depends on the machine
        return False


def resolve_device(pref: str = "auto") -> Device:
    """Map 'auto' | 'mps' | 'cpu' to a device that works here (MPS falls back to CPU)."""
    if pref == "cpu":
        return "cpu"
    if pref in ("auto", "mps") and mps_usable():
        return "mps"
    if pref == "cuda":
        raise ValueError("CUDA is not supported by this project; use 'auto', 'mps' or 'cpu'")
    return "cpu"


def memory_in_use_mb(device: str) -> float:
    """Allocated tensor memory on MPS, else the process' peak RSS."""
    if device == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def hardware_report() -> dict[str, Any]:
    ram = total_ram_bytes()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "ram_gb": round(ram / 2**30, 1) if ram else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_usable": mps_usable(),
        "cuda_available": torch.cuda.is_available(),
        "default_device": resolve_device("auto"),
        "torch_threads": torch.get_num_threads(),
    }
