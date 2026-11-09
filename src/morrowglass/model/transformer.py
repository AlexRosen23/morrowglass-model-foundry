"""A small decoder-only causal Transformer written directly in PyTorch.

Tensor-shape walkthrough (B = batch, T = sequence length, C = d_model, H = heads,
D = C // H head size, V = vocab):

    idx            (B, T)        int64 token ids
    tok_emb(idx)   (B, T, C)     learned token embedding
    + pos          (T, C)        learned table or fixed sinusoids, broadcast over B
    for each Block (pre-LayerNorm residual):
        h = LN1(x)                          (B, T, C)
        q, k, v = split(W_qkv h)            each (B, H, T, D)
        scores = q @ k^T / sqrt(D)          (B, H, T, T)
        scores[j > i] = -inf                causal mask: position i sees 0..i only
        A = softmax(scores, -1)             (B, H, T, T)  rows sum to 1
        y = A @ v -> merge heads -> W_o     (B, T, C)
        x = x + y                           residual
        x = x + W_2 GELU(W_1 LN2(x))        MLP, hidden 4C by default
    LN_f(x) @ W_tok^T                       (B, T, V) logits (weights tied to tok_emb)

The loss is the mean next-token cross-entropy in nats; targets of -100 are ignored.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

PosEncoding = Literal["learned", "sinusoidal", "none"]
# Per layer: (keys, values), each (B, H, T_past, D). Built by ``forward(..., use_cache=True)``.
KVCache = list[tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int | None = None  # defaults to 4 * d_model
    dropout: float = 0.0
    pos_encoding: PosEncoding = "learned"
    tie_weights: bool = True
    bias: bool = False
    causal: bool = True  # False exists ONLY for the "Break the Model" leakage demo

    def __post_init__(self) -> None:
        if self.d_model % self.n_head:
            raise ValueError(f"d_model={self.d_model} not divisible by n_head={self.n_head}")
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.d_model) <= 0:
            raise ValueError("model dimensions must be positive")

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
    logits: Tensor
    loss: Tensor | None = None
    attentions: list[Tensor] = field(default_factory=list)  # each (B, H, T, T)
    past_kv: KVCache | None = None  # only when forward(..., use_cache=True)


def sinusoidal_table(n_pos: int, d_model: int) -> Tensor:
    pos = torch.arange(n_pos, dtype=torch.float32)[:, None]
    i = torch.arange(0, d_model, 2, dtype=torch.float32)[None, :]
    angle = pos / (10000 ** (i / d_model))
    table = torch.zeros(n_pos, d_model)
    table[:, 0::2] = torch.sin(angle)
    table[:, 1::2] = torch.cos(angle[:, : d_model // 2])
    return table


class CausalSelfAttention(nn.Module):
    mask: Tensor  # registered buffer (1, 1, block_size, block_size), True where attention is allowed

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.causal = cfg.causal
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(
        self, x: Tensor, return_attn: bool = False, past: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None, tuple[Tensor, Tensor] | None]:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, H, T, D)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if past is not None:
            # KV cache: keys/values of earlier positions are reused, only the new queries are computed.
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v) if use_cache else None
        t_past = k.size(2) - T
        if not return_attn and self.causal:
            # Same maths as below via the fused kernel (verified equal in tests).
            if t_past == 0:
                y = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=True
                )
            else:
                # Query i sits at absolute position t_past + i and may read keys 0..t_past + i.
                # (is_causal=True would align the mask top-left, which is wrong when T_q < T_k.)
                allowed = self.mask[:, :, t_past : t_past + T, : t_past + T]
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=allowed, dropout_p=self.attn_drop.p if self.training else 0.0
                )
            att = None
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T_past + T)
            if self.causal:
                scores = scores.masked_fill(~self.mask[:, :, t_past : t_past + T, : t_past + T], float("-inf"))
            att = F.softmax(scores, dim=-1)
            y = self.attn_drop(att) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # merge heads
        return self.resid_drop(self.proj(y)), att, present


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.ff, bias=cfg.bias)
        self.out = nn.Linear(cfg.ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.out(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(
        self, x: Tensor, return_attn: bool = False, past: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None, tuple[Tensor, Tensor] | None]:
        a, att, present = self.attn(self.ln1(x), return_attn, past, use_cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, att, present


class MorrowglassLM(nn.Module):
    """Decoder-only language model. See module docstring for the shape walkthrough."""

    pos_table: Tensor  # registered buffer: fixed sinusoidal table (block_size, d_model)

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_encoding == "learned":
            self.pos_emb: nn.Embedding | None = nn.Embedding(cfg.block_size, cfg.d_model)
        else:
            self.pos_emb = None
        self.register_buffer("pos_table", sinusoidal_table(cfg.block_size, cfg.d_model), persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.out.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def positions(self, T: int, offset: int = 0) -> Tensor:
        if offset + T > self.cfg.block_size:
            raise ValueError(f"positions {offset}+{T} exceed block_size {self.cfg.block_size}")
        if self.cfg.pos_encoding == "learned":
            assert self.pos_emb is not None
            return self.pos_emb.weight[offset : offset + T]
        if self.cfg.pos_encoding == "sinusoidal":
            return self.pos_table[offset : offset + T]
        return torch.zeros(T, self.cfg.d_model, device=self.tok_emb.weight.device)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        return_attn: bool = False,
        pos_offset: int | Tensor = 0,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        """Logits for ``idx``. With ``past_kv`` (from an earlier ``use_cache=True`` call), ``idx`` holds only
        the new tokens; their positions continue after the cached ones and must stay within block_size."""
        B, T = idx.shape
        if self.cfg.block_size < T:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")
        if past_kv is not None:
            if isinstance(pos_offset, Tensor) or pos_offset != 0:
                raise ValueError("pos_offset is implied by the cache length when past_kv is given")
            pos_offset = past_kv[0][0].size(2)
        x = self.tok_emb(idx)
        if isinstance(pos_offset, Tensor):  # per-example offsets (position augmentation)
            if self.cfg.pos_encoding == "none":
                pos = torch.zeros_like(x)
            else:
                table = self.pos_emb.weight if self.pos_emb is not None else self.pos_table
                pos_ids = pos_offset.view(B, 1) + torch.arange(T, device=idx.device).view(1, T)
                pos = table[pos_ids]
            x = x + pos
        else:
            x = x + self.positions(T, pos_offset)
        x = self.drop(x)
        attentions: list[Tensor] = []
        presents: KVCache = []
        for i, block in enumerate(self.blocks):
            x, att, present = block(x, return_attn, past_kv[i] if past_kv is not None else None, use_cache)
            if att is not None:
                attentions.append(att)
            if present is not None:
                presents.append(present)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return ModelOutput(logits=logits, loss=loss, attentions=attentions, past_kv=presents if use_cache else None)


def count_parameters(model: nn.Module) -> int:
    """Unique trainable parameters (tied weights counted once)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def analytic_param_count(cfg: ModelConfig) -> int:
    """Closed-form parameter count; tests assert it equals ``count_parameters``."""
    C, F_, V, L = cfg.d_model, cfg.ff, cfg.vocab_size, cfg.n_layer
    b = 1 if cfg.bias else 0
    ln = C + b * C
    attn = C * 3 * C + b * 3 * C + C * C + b * C
    mlp = C * F_ + b * F_ + F_ * C + b * C
    per_block = 2 * ln + attn + mlp
    emb = V * C + (cfg.block_size * C if cfg.pos_encoding == "learned" else 0)
    head = 0 if cfg.tie_weights else V * C
    return emb + L * per_block + ln + head


def compute_budget(cfg: ModelConfig, tokens: int) -> dict[str, Any]:
    """Approximate training compute: 6 * N_nonembed * tokens + attention term (Kaplan et al. 2020)."""
    n_total = analytic_param_count(cfg)
    n_emb = cfg.vocab_size * cfg.d_model + (cfg.block_size * cfg.d_model if cfg.pos_encoding == "learned" else 0)
    n_nonemb = n_total - n_emb
    attn_flops_per_token = 12 * cfg.n_layer * cfg.block_size * cfg.d_model  # fwd+bwd, ~6 * 2 L T C
    head_flops_per_token = 6 * cfg.vocab_size * cfg.d_model
    flops_per_token = 6 * n_nonemb + attn_flops_per_token + head_flops_per_token
    return {
        "params_total": n_total,
        "params_non_embedding": n_nonemb,
        "train_tokens": tokens,
        "approx_train_flops": float(flops_per_token * tokens),
        "approx_flops_per_token": float(flops_per_token),
        "weights_mb_fp32": round(n_total * 4 / 2**20, 2),
        "checkpoint_mb_with_adam": round(n_total * 4 * 3 / 2**20, 2),
    }


def blueprint(cfg: ModelConfig) -> list[dict[str, Any]]:
    """Layer-by-layer description with tensor shapes, used by the UI's left panel."""
    C, T, V, H = cfg.d_model, cfg.block_size, cfg.vocab_size, cfg.n_head
    D = C // H
    rows: list[dict[str, Any]] = [
        {"name": "tok_emb", "kind": "embedding", "shape_in": "(B, T)", "shape_out": f"(B, T, {C})",
         "params": V * C, "note": f"{V} x {C} table" + (", tied to lm_head" if cfg.tie_weights else "")},
        {"name": "pos", "kind": f"position:{cfg.pos_encoding}", "shape_in": "(T,)", "shape_out": f"(T, {C})",
         "params": T * C if cfg.pos_encoding == "learned" else 0,
         "note": {"learned": f"learned {T} x {C}", "sinusoidal": "fixed sin/cos, no params",
                  "none": "no position signal"}[cfg.pos_encoding]},
    ]
    ln = C * (2 if cfg.bias else 1)
    b = 1 if cfg.bias else 0
    for i in range(cfg.n_layer):
        rows += [
            {"name": f"block{i}.ln1", "kind": "layernorm", "shape_in": f"(B, T, {C})", "shape_out": f"(B, T, {C})",
             "params": ln, "note": "pre-norm"},
            {"name": f"block{i}.attn", "kind": "attention", "shape_in": f"(B, T, {C})", "shape_out": f"(B, T, {C})",
             "params": 4 * C * C + b * 4 * C, "heads": H, "head_dim": D,
             "note": f"{H} heads x {D}; scores (B, {H}, T, T), causal={cfg.causal}"},
            {"name": f"block{i}.resid1", "kind": "residual", "shape_in": "x + attn", "shape_out": f"(B, T, {C})",
             "params": 0, "note": "skip connection"},
            {"name": f"block{i}.ln2", "kind": "layernorm", "shape_in": f"(B, T, {C})", "shape_out": f"(B, T, {C})",
             "params": ln, "note": "pre-norm"},
            {"name": f"block{i}.mlp", "kind": "mlp", "shape_in": f"(B, T, {C})", "shape_out": f"(B, T, {C})",
             "params": 2 * C * cfg.ff + b * (cfg.ff + C), "note": f"{C} -> {cfg.ff} GELU -> {C}"},
            {"name": f"block{i}.resid2", "kind": "residual", "shape_in": "x + mlp", "shape_out": f"(B, T, {C})",
             "params": 0, "note": "skip connection"},
        ]
    rows += [
        {"name": "ln_f", "kind": "layernorm", "shape_in": f"(B, T, {C})", "shape_out": f"(B, T, {C})",
         "params": ln, "note": "final norm"},
        {"name": "lm_head", "kind": "linear", "shape_in": f"(B, T, {C})", "shape_out": f"(B, T, {V})",
         "params": 0 if cfg.tie_weights else V * C, "note": "logits; tied" if cfg.tie_weights else "logits"},
    ]
    return rows
