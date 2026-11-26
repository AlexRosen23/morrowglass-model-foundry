"""Steps 3 and 4: the Transformer block and the whole decoder-only language model.

    idx (B, T) token ids
    x = tok_emb(idx) + pos(T)                      (B, T, C)
    for each block (pre-LayerNorm, residual):
        x = x + attn(LN1(x))                       step 2's causal self-attention
        x = x + mlp(LN2(x))                        C -> 4C -> GELU -> C
    logits = LN_f(x) @ tok_emb.weight^T            (B, T, V), weights tied to the embedding
    loss = cross-entropy(logits, targets)          mean over positions, in nats; targets of -100 ignored
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scratchlm.attention import CausalSelfAttention, LayerCache

PosEncoding = Literal["learned", "sinusoidal"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 128  # longest context the model can read
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int | None = None  # MLP hidden width; None means 4 * d_model
    pos_encoding: PosEncoding = "learned"
    tie_weights: bool = True  # lm_head shares its weight matrix with tok_emb

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.d_model) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_head:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_head={self.n_head}")
        if self.pos_encoding not in ("learned", "sinusoidal"):
            raise ValueError(f"unknown pos_encoding {self.pos_encoding!r}")
        if self.pos_encoding == "sinusoidal" and self.d_model % 2:
            raise ValueError("sinusoidal positions need an even d_model")

    @property
    def ff(self) -> int:
        return self.d_ff or 4 * self.d_model

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelOutput:
    logits: Tensor  # (B, T, V)
    loss: Tensor | None = None  # scalar, only when targets were given
    weights: list[Tensor] = field(default_factory=list)  # attention weights per layer, if asked for
    past_kv: list[LayerCache] | None = None  # per-layer cache, only with use_cache=True (step 8)


class MLP(nn.Module):
    """Position-wise feed-forward network: Linear(C, F) -> GELU -> Linear(F, C)."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff)
        self.out = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        #lab:begin 3 the feed-forward network
        return self.out(F.gelu(self.fc(x)))
        #lab:stub
        #> # TODO(step 3): expand with self.fc, apply GELU (torch.nn.functional.gelu), project back with self.out.
        #> raise NotImplementedError("TODO(step 3): MLP.forward")
        #lab:end


class Block(nn.Module):
    """One pre-LayerNorm Transformer block."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.ff)

    def forward(
        self, x: Tensor, past: LayerCache | None = None, use_cache: bool = False, return_weights: bool = False
    ) -> tuple[Tensor, Tensor | None, LayerCache | None]:
        """Returns ``(x, attention weights or None, present cache or None)``. Pass ``past``,
        ``use_cache`` and ``return_weights`` straight through to ``self.attn``."""
        #lab:begin 3 two residual branches, each after a LayerNorm
        a, weights, present = self.attn(self.ln1(x), past, use_cache, return_weights)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, weights, present
        #lab:stub
        #> # TODO(step 3): x + attn(ln1(x)), then x + mlp(ln2(x)). self.attn returns three values: keep the
        #> # attention output, and return its weights and cache unchanged alongside the new x.
        #> raise NotImplementedError("TODO(step 3): Block.forward")
        #lab:end


def sinusoidal_table(n_pos: int, d_model: int) -> Tensor:
    """Fixed position signals from "Attention Is All You Need" (Vaswani et al., 2017), shape (n_pos, d_model).

    For position p and pair index i (0 <= i < d_model / 2), with angle = p / 10000^(2i / d_model):
    column 2i is sin(angle) and column 2i + 1 is cos(angle).
    """
    #lab:begin 4 sinusoidal position table
    pos = torch.arange(n_pos, dtype=torch.float32)[:, None]
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)[None, :]
    angle = pos / (10000 ** (two_i / d_model))
    table = torch.zeros(n_pos, d_model)
    table[:, 0::2] = torch.sin(angle)
    table[:, 1::2] = torch.cos(angle)
    return table
    #lab:stub
    #> # TODO(step 4): build `angle` with shape (n_pos, d_model // 2) by broadcasting a column of positions
    #> # against a row of 2i values, then fill the even columns with sin(angle) and the odd ones with cos(angle).
    #> raise NotImplementedError("TODO(step 4): sinusoidal_table")
    #lab:end


class LanguageModel(nn.Module):
    """Decoder-only Transformer language model (GPT-style). See the module docstring."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model) if cfg.pos_encoding == "learned" else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self._pos_table: Tensor | None = None  # sinusoidal table, built on first use
        self.apply(self._init_weights)
        # Scale down the projections that write into the residual stream, so its variance does not grow
        # with depth (as in GPT-2).
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.out.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def positions(self, T: int, offset: int = 0) -> Tensor:
        """Position vectors (T, d_model) for absolute positions offset .. offset + T - 1.

        Raises ValueError if offset + T > block_size (the model has no vectors for those positions).
        """
        #lab:begin 4 learned or sinusoidal positions
        if offset + T > self.cfg.block_size:
            raise ValueError(f"positions {offset}..{offset + T - 1} exceed block_size {self.cfg.block_size}")
        if self.pos_emb is not None:
            return self.pos_emb.weight[offset : offset + T]
        if self._pos_table is None:
            self._pos_table = sinusoidal_table(self.cfg.block_size, self.cfg.d_model)
        return self._pos_table[offset : offset + T].to(self.tok_emb.weight.device)
        #lab:stub
        #> # TODO(step 4): check offset + T against block_size, then return rows offset .. offset + T - 1 of
        #> # self.pos_emb.weight (learned) or of sinusoidal_table(...) (built once, kept in self._pos_table).
        #> raise NotImplementedError("TODO(step 4): LanguageModel.positions")
        #lab:end

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        past_kv: list[LayerCache] | None = None,
        use_cache: bool = False,
        return_weights: bool = False,
    ) -> ModelOutput:
        """Logits for every position of ``idx`` (B, T), plus the loss when ``targets`` (B, T) is given.

        With ``past_kv`` (step 8), ``idx`` holds only the new tokens and their positions continue after
        the cached ones.
        """
        offset = past_kv[0][0].size(2) if past_kv else 0
        weights: list[Tensor] = []
        presents: list[LayerCache] = []
        #lab:begin 4 embeddings, blocks, logits and loss
        T = idx.size(1)
        x = self.tok_emb(idx) + self.positions(T, offset)
        for i, block in enumerate(self.blocks):
            x, w, present = block(x, past_kv[i] if past_kv else None, use_cache, return_weights)
            if w is not None:
                weights.append(w)
            if present is not None:
                presents.append(present)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        #lab:stub
        #> # TODO(step 4): token embeddings + positions(T, offset); run every block (pass it past_kv[i] when
        #> # past_kv is given, and use_cache / return_weights), collecting the weights and presents it
        #> # returns; final LayerNorm; lm_head; and, when targets is given, the mean cross-entropy over all
        #> # positions (F.cross_entropy with ignore_index=-100 on flattened logits and targets).
        #> raise NotImplementedError("TODO(step 4): LanguageModel.forward")
        #lab:end
        return ModelOutput(logits=logits, loss=loss, weights=weights, past_kv=presents if use_cache else None)


def count_parameters(model: nn.Module) -> int:
    """Trainable parameters, with tied weights counted once."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def analytic_param_count(cfg: ModelConfig) -> int:
    """The parameter count worked out from the config alone; must equal count_parameters(LanguageModel(cfg)).

    Every nn.Linear here has a bias except lm_head; every nn.LayerNorm has a weight and a bias.
    """
    #lab:begin 4 count the parameters by hand
    C, F_, V, L = cfg.d_model, cfg.ff, cfg.vocab_size, cfg.n_layer
    layer_norm = 2 * C
    attn = (C * 3 * C + 3 * C) + (C * C + C)
    mlp = (C * F_ + F_) + (F_ * C + C)
    block = 2 * layer_norm + attn + mlp
    embeddings = V * C + (cfg.block_size * C if cfg.pos_encoding == "learned" else 0)
    head = 0 if cfg.tie_weights else V * C
    return embeddings + L * block + layer_norm + head
    #lab:stub
    #> # TODO(step 4): add up embeddings (tokens, plus positions if learned), n_layer blocks (two LayerNorms,
    #> # qkv and proj, fc and out, each Linear with its bias), the final LayerNorm and, if the weights are
    #> # not tied, lm_head.
    #> raise NotImplementedError("TODO(step 4): analytic_param_count")
    #lab:end
