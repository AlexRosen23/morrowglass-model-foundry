"""Dynamic int8 quantization: runs on CPU, leaves the fp32 model alone, stays close to it, works with the KV cache."""

from __future__ import annotations

import torch

from morrowglass.inference.generate import CachedDecoder
from morrowglass.inference.quant import quantize_int8, state_dict_mb
from morrowglass.model.transformer import ModelConfig, MorrowglassLM


@torch.no_grad()
def test_int8_copy_is_close_smaller_and_cache_compatible() -> None:
    torch.manual_seed(0)
    m = MorrowglassLM(ModelConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, d_model=64)).eval()
    q = quantize_int8(m)
    assert type(m.blocks[0].attn.qkv) is torch.nn.Linear  # original untouched
    assert "quantized" in type(q.blocks[0].attn.qkv).__module__
    x = torch.randint(0, 256, (2, 32))
    p, pq = torch.softmax(m(x).logits, -1), torch.softmax(q(x).logits, -1)
    assert float((p - pq).abs().max()) < 0.05
    assert state_dict_mb(q) < state_dict_mb(m)
    ids = list(range(40, 50))
    a, b = CachedDecoder(q, use_cache=True), CachedDecoder(q, use_cache=False)
    for _ in range(10):
        la, lb = a.next_logits(ids), b.next_logits(ids)
        # Not bit-equal: dynamic int8 picks activation scales per call from the tensor it sees, and a one-token
        # step sees a different range than the full window. fp32 is equal to 1e-5 (test_kv_cache.py).
        assert float((la - lb).abs().max()) < 0.05
        ids.append(int(la.argmax()))
