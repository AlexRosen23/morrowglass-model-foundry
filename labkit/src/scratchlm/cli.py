"""Command line (given; not an exercise). Run ``python -m scratchlm --help``.

    python -m scratchlm info  [model options]
    python -m scratchlm train --data TRAIN.txt [--val VAL.txt] --out runs/NAME [model and training options]
    python -m scratchlm eval  --ckpt runs/NAME/best.pt --data VAL.txt
    python -m scratchlm sample --ckpt runs/NAME/best.pt --prompt "ROMEO:" [--tokens 300] [--kv-cache]

Training runs on the CPU. eval and sample take --device, but only the CPU is tested by the kit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from scratchlm.model import LanguageModel, ModelConfig, count_parameters


def _model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--block-size", type=int, default=128, help="context length in tokens (default 128)")
    g.add_argument("--n-layer", type=int, default=4)
    g.add_argument("--n-head", type=int, default=4)
    g.add_argument("--d-model", type=int, default=128)
    g.add_argument("--pos", choices=["learned", "sinusoidal"], default="learned", help="position encoding")


def _model_cfg(a: argparse.Namespace) -> ModelConfig:
    return ModelConfig(block_size=a.block_size, n_layer=a.n_layer, n_head=a.n_head, d_model=a.d_model,
                       pos_encoding=a.pos)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m scratchlm", description="A small Transformer language model.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="print a model's size for the given options")
    _model_args(p)

    p = sub.add_parser("train", help="train a model on a text file")
    p.add_argument("--data", required=True, help="training text (UTF-8)")
    p.add_argument("--val", help="validation text; default: the last 10%% of --data")
    p.add_argument("--out", required=True, help="output folder for best.pt, last.pt and summary.json")
    _model_args(p)
    g = p.add_argument_group("training")
    g.add_argument("--steps", type=int, default=1000)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--lr", type=float, default=3e-3)
    g.add_argument("--warmup", type=int, default=50)
    g.add_argument("--eval-interval", type=int, default=100)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--threads", type=int, default=2, help="CPU threads (default 2, to leave room for other work)")

    p = sub.add_parser("eval", help="score a checkpoint on a text file (all windows)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)

    p = sub.add_parser("sample", help="generate text from a checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40, help="0 disables top-k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kv-cache", action="store_true", help="use the key/value cache (step 8)")

    for name in ("eval", "sample"):
        sub.choices[name].add_argument("--device", default="cpu", help="cpu (default; the only tested device)")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "info":
        cfg = _model_cfg(a)
        n = count_parameters(LanguageModel(cfg))
        print(json.dumps({"parameters": n, "weights_mb_fp32": round(n * 4 / 2**20, 2), "config": cfg.to_dict()},
                         indent=2))
        return 0

    from scratchlm.checkpoint import load_checkpoint
    from scratchlm.data import encode_file, load_text, split_text
    from scratchlm.evaluate import evaluate
    from scratchlm.sample import generate
    from scratchlm.tokenizer import ByteTokenizer
    from scratchlm.train import TrainConfig, train

    if a.cmd == "train":
        tok = ByteTokenizer()
        if a.val:
            train_ids, val_ids = encode_file(a.data, tok), encode_file(a.val, tok)
        else:
            tr, va = split_text(load_text(a.data))
            train_ids = torch.tensor(tok.encode(tr), dtype=torch.long)
            val_ids = torch.tensor(tok.encode(va), dtype=torch.long)
        tcfg = TrainConfig(steps=a.steps, batch_size=a.batch_size, lr=a.lr, warmup_steps=a.warmup,
                           eval_interval=a.eval_interval, seed=a.seed, threads=a.threads)
        _, summary = train(_model_cfg(a), tcfg, train_ids, val_ids, out_dir=a.out)
        print(f"{summary['status']}: best validation bits/byte {summary['best_val_bits_per_byte']:.3f} "
              f"at step {summary['best_step']}, {summary['wall_time_s']}s; checkpoints in {a.out}")
        return 0 if summary["status"] == "completed" else 1

    model, _ = load_checkpoint(a.ckpt)
    model.to(a.device)
    if a.cmd == "eval":
        ids = encode_file(a.data).to(a.device)
        print(json.dumps(evaluate(model, ids), indent=2))
        return 0
    t0 = time.time()
    text = generate(model, a.prompt, max_new_tokens=a.tokens, temperature=a.temperature,
                    top_k=a.top_k or None, seed=a.seed, use_cache=a.kv_cache)
    dt = time.time() - t0
    print(a.prompt + text)
    print(f"\n[{a.tokens} tokens in {dt:.2f}s, kv-cache {'on' if a.kv_cache else 'off'}]", file=sys.stderr)
    return 0
