"""Step 3: the MLP and the pre-LayerNorm residual block."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scratchlm.model import MLP, Block, ModelConfig


def test_step03_mlp_is_linear_gelu_linear() -> None:
    torch.manual_seed(0)
    mlp = MLP(8, 32)
    x = torch.randn(2, 5, 8)
    y = mlp(x)
    assert y.shape == x.shape
    manual = F.linear(F.gelu(F.linear(x, mlp.fc.weight, mlp.fc.bias)), mlp.out.weight, mlp.out.bias)
    assert torch.allclose(y, manual, atol=1e-6)


def test_step03_block_matches_the_pre_norm_formula() -> None:
    torch.manual_seed(1)
    blk = Block(ModelConfig(block_size=8, n_layer=1, n_head=2, d_model=16))
    x = torch.randn(3, 8, 16)
    y, w, present = blk(x, return_weights=True)
    a, _, _ = blk.attn(blk.ln1(x))
    h = x + a
    assert torch.allclose(y, h + blk.mlp(blk.ln2(h)), atol=1e-6)
    assert w is not None and w.shape == (3, 2, 8, 8)
    assert present is None
    _, w_none, _ = blk(x)
    assert w_none is None


def test_step03_block_with_silent_branches_is_the_identity() -> None:
    blk = Block(ModelConfig(block_size=8, n_layer=1, n_head=2, d_model=16))
    with torch.no_grad():
        for lin in (blk.attn.proj, blk.mlp.out):
            lin.weight.zero_()
            lin.bias.zero_()
    x = torch.randn(2, 8, 16)
    assert torch.equal(blk(x)[0], x)  # the residual path carries x through untouched


def test_step03_block_is_causal() -> None:
    torch.manual_seed(2)
    blk = Block(ModelConfig(block_size=12, n_layer=1, n_head=4, d_model=16))
    x = torch.randn(1, 12, 16)
    x2 = x.clone()
    x2[:, 9:] += 1.0
    assert torch.allclose(blk(x)[0][:, :9], blk(x2)[0][:, :9], atol=1e-6)
