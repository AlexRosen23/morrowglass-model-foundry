"""Models the OpenAI-compatible API can serve, behind one small interface.

- ``MorrowglassBackend``: a scratch checkpoint from this repo (byte or BPE tokenizer), decoded with
  the KV cache (``inference.generate.CachedDecoder``).
- ``HFBackend``: a Hugging Face causal LM loaded from the local cache only (``local_files_only=True``;
  this module never downloads). It exists so the standard harness can score distilgpt2 through the
  very same endpoint as the scratch models. It is enabled only with ``MORROWGLASS_ENABLE_HF=1``.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol

import torch

from morrowglass.inference.generate import CachedDecoder
from morrowglass.model.transformer import MorrowglassLM
from morrowglass.tokenizer.tokenizers import Tokenizer

HF_ALLOWED = {"hf:distilgpt2": "distilgpt2"}


class Stepper(Protocol):
    def next_logits(self, ids: list[int]) -> torch.Tensor: ...


class Backend(Protocol):
    model_id: str
    kind: str
    block_size: int
    vocab_size: int
    lock: threading.Lock

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def logprobs(self, ids: list[int]) -> torch.Tensor: ...

    def stepper(self) -> Stepper: ...

    def tokenizer_info(self) -> dict[str, Any]: ...

    def describe(self) -> dict[str, Any]: ...


class MorrowglassBackend:
    kind = "morrowglass"

    def __init__(self, run_id: str, model: MorrowglassLM, tok: Tokenizer, provenance: str,
                 use_cache: bool = True) -> None:
        self.model_id = run_id
        self.model = model.eval()
        self.tok = tok
        self.provenance = provenance
        self.use_cache = use_cache
        self.block_size = model.cfg.block_size
        self.vocab_size = model.cfg.vocab_size
        self.lock = threading.Lock()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    @torch.no_grad()
    def logprobs(self, ids: list[int]) -> torch.Tensor:
        """(T, V) log-probabilities; row i is the distribution over the token after ids[i]. T <= block_size."""
        logits = self.model(torch.tensor([ids])).logits[0].float()
        return torch.log_softmax(logits, dim=-1)

    def stepper(self) -> Stepper:
        return CachedDecoder(self.model, "cpu", self.use_cache)

    def tokenizer_info(self) -> dict[str, Any]:
        return {"eos_token": None, "bos_token": None, "pad_token": None, "tokenizer": self.tok.name,
                "vocab_size": self.vocab_size, "max_model_len": self.block_size,
                "note": "No special tokens: the scratch tokenizers are lossless byte / byte-level BPE. "
                        "Use a newline as the rolling-window prefix token."}

    def describe(self) -> dict[str, Any]:
        return {"provenance": self.provenance, "tokenizer": self.tok.name, "context_tokens": self.block_size,
                "kv_cache": self.use_cache, "instruction_tuned": False,
                "trained_on": "Shakespeare (Project Gutenberg #100, train plays only)"}


class _HFStepper:
    def __init__(self, model: Any, block: int) -> None:
        self.model, self.block = model, block

    @torch.no_grad()
    def next_logits(self, ids: list[int]) -> torch.Tensor:
        return self.model(torch.tensor([ids[-self.block:]])).logits[0, -1]


class HFBackend:
    kind = "hf"

    def __init__(self, model_id: str) -> None:
        if model_id not in HF_ALLOWED:
            raise KeyError(model_id)
        try:
            import transformers
        except ImportError as e:  # pragma: no cover - dev group has it
            raise RuntimeError("transformers is not installed (uv sync --extra hf)") from e
        name = HF_ALLOWED[model_id]
        self.model_id = model_id
        self.tok = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(name, local_files_only=True).eval()
        self.block_size = int(self.model.config.n_positions)
        self.vocab_size = int(self.model.config.vocab_size)
        self.lock = threading.Lock()

    def encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return str(self.tok.decode(ids))

    @torch.no_grad()
    def logprobs(self, ids: list[int]) -> torch.Tensor:
        return torch.log_softmax(self.model(torch.tensor([ids])).logits[0].float(), dim=-1)

    def stepper(self) -> Stepper:
        return _HFStepper(self.model, self.block_size)  # no KV cache for the reference model

    def tokenizer_info(self) -> dict[str, Any]:
        return {"eos_token": self.tok.eos_token, "bos_token": self.tok.bos_token, "pad_token": None,
                "tokenizer": HF_ALLOWED[self.model_id], "vocab_size": self.vocab_size,
                "max_model_len": self.block_size}

    def describe(self) -> dict[str, Any]:
        return {"provenance": "pretrained reference (Hugging Face cache, not trained here)",
                "tokenizer": "gpt2 BPE", "context_tokens": self.block_size, "kv_cache": False,
                "instruction_tuned": False, "trained_on": "OpenWebText (distilled from GPT-2)"}
