"""Hugging Face Hub-compatible export (local save only; publishing needs the owner's approval).

``MorrowglassHubLM`` is the same network with ``huggingface_hub.PyTorchModelHubMixin`` mixed in, so a folder
written by ``save_pretrained`` holds ``model.safetensors`` + ``config.json`` (the ``ModelConfig`` dataclass)
and loads back with ``MorrowglassHubLM.from_pretrained(path_or_repo_id)``. ``export_hf`` adds:

- ``README.md``: a model card filled from the saved run records (metrics with CIs, the pre-registered
  verdicts that involve the run, serving measurements, intended use, limitations, data);
- ``morrowglass.json``: tokenizer, hashes, provenance and a round-trip check (logits after reload);
- ``tokenizer.json`` for BPE runs (the byte tokenizer needs no file).

Nothing here talks to the network. The publish command is printed, never run.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import PyTorchModelHubMixin

from morrowglass.model.transformer import ModelConfig, MorrowglassLM
from morrowglass.repro import REPO_ROOT, git_provenance, now_iso, read_json, sha256_file, write_json, write_verified
from morrowglass.train.trainer import CHECKPOINT_ROOT, RUNS_DIR, load_model_from_checkpoint

EXPORT_ROOT = REPO_ROOT / "artifacts" / "hf_export"
CARD_ROOT = REPO_ROOT / "reports" / "hf_export"
REPO_URL = "https://github.com/AlexRosen23/morrowglass-model-foundry (private)"


class MorrowglassHubLM(
    MorrowglassLM,
    PyTorchModelHubMixin,
    library_name="morrowglass",
    license="mit",
    pipeline_tag="text-generation",
    tags=["from-scratch", "decoder-only", "shakespeare", "character-level", "research"],
):
    """``MorrowglassLM`` + ``save_pretrained`` / ``from_pretrained`` (safetensors)."""

    def __init__(self, config: ModelConfig | dict[str, Any]) -> None:
        # from_pretrained passes config.json as a dict (annotations are strings under `from __future__`).
        super().__init__(config if isinstance(config, ModelConfig) else ModelConfig.from_dict(config))


def hub_repo_name(run_id: str) -> str:
    return "morrowglass-" + run_id.rsplit("-", 2)[0]


def publish_command(run_id: str, owner: str = "AlexRosen23") -> str:
    folder = EXPORT_ROOT / run_id
    rel = folder.relative_to(REPO_ROOT) if folder.is_relative_to(REPO_ROOT) else folder
    return f"uv run hf upload {owner}/{hub_repo_name(run_id)} {rel} . --repo-type model --private"


def _fmt_ci(v: dict[str, Any]) -> str:
    ci = v.get("bits_per_byte_ci95") or v.get("ci95")
    return f"{v['bits_per_byte']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{v['bits_per_byte']:.3f}"


def _verdicts(run_id: str) -> list[dict[str, str]]:
    out = []
    for p in sorted((REPO_ROOT / "reports" / "experiments").glob("*/result.json")):
        r = read_json(p)
        if run_id in r.get("arms", []):
            out.append({"experiment": r["experiment"], "title": r["title"], "verdict": r["verdict"],
                        "detail": str(r.get("verdict_detail", ""))})
    return out


def _serving(run_id: str) -> dict[str, Any]:
    s: dict[str, Any] = {}
    kv = REPO_ROOT / "reports" / "serving" / "kv-cache" / f"{run_id}.json"
    if kv.exists():
        s["kv"] = read_json(kv)
    q = REPO_ROOT / "reports" / "serving" / "int8" / f"{run_id}.json"
    if q.exists():
        s["int8"] = read_json(q)
    le = REPO_ROOT / "reports" / "lm_eval" / "summary.json"
    if le.exists() and run_id in read_json(le)["models"]:
        s["lm_eval"] = read_json(le)["models"][run_id]
    return s


def model_card(run_id: str, runs_dir: Path = RUNS_DIR) -> str:
    rd = runs_dir / run_id
    summ, meta, cfg = read_json(rd / "summary.json"), read_json(rd / "metadata.json"), read_json(rd / "config.json")
    tr, mc = cfg["train"], cfg["model"]
    val = summ["final"]["val"]
    test = read_json(rd / "test_eval.json") if (rd / "test_eval.json").exists() else None
    name = hub_repo_name(run_id)
    tok = tr["tokenizer"]
    lines = [
        "---",
        "license: mit",
        "language: en",
        "library_name: morrowglass",
        "pipeline_tag: text-generation",
        "tags:", "  - pytorch_model_hub_mixin", "  - model_hub_mixin", "  - from-scratch", "  - shakespeare",
        "  - decoder-only",
        "---",
        "",
        f"# {name}",
        "",
        f"A {summ['params'] / 1e6:.2f}M-parameter decoder-only Transformer trained **from scratch** on public-domain "
        f"Shakespeare by [Morrowglass Model Foundry]({REPO_URL.split(' ')[0]}), a reproducible miniature "
        "language-model lab. Every number below is copied by code from the saved run records; nothing is typed "
        "by hand.",
        "",
        "> **Not instruction-tuned and not a general model.** It continues text in a Shakespeare-like style. "
        "Use it to study small-model training, evaluation and serving, not to generate text anyone relies on.",
        "",
        "## Model",
        "",
        "| | |",
        "|---|---|",
        f"| run id | `{run_id}` (experiment `{tr['experiment']}`, profile `{tr['profile']}`, seed {tr['seed']}) |",
        f"| architecture | pre-LayerNorm decoder, {mc['n_layer']} layers x {mc['n_head']} heads x d={mc['d_model']}, "
        f"context {mc['block_size']} tokens, {mc.get('pos_encoding', 'learned')} positions, tied embeddings |",
        f"| tokenizer | `{tok}` ({'UTF-8 bytes, vocab 256, no file needed' if tok == 'byte' else 'byte-level BPE fitted on the train split; tokenizer.json'}) |",
        f"| parameters | {summ['params']:,} |",
        f"| training | {summ['steps_completed']:,} steps, {summ['tokens_seen']:,} tokens, AdamW lr {tr['lr']}, "
        f"{meta['device'].upper()} on {meta['hardware'].get('chip', 'unknown')}, {summ['wall_time_s']:.0f} s |",
        f"| code commit | `{meta['git']['commit'][:12]}` |",
        "",
        "## Data",
        "",
        "Project Gutenberg eBook #100 (complete works of Shakespeare, public domain, boilerplate removed), split "
        "**by play** with a fixed seed: validation = *As You Like It*, *Hamlet*, *Titus Andronicus*; test = "
        "*Macbeth*, *Pericles*, *Cymbeline*; everything else (including the poems) is train. Model selection used "
        "validation only.",
        "",
        "## Evaluation (held-out plays)",
        "",
        "Bits per UTF-8 byte (lower is better), 95% bootstrap CI over evaluation windows.",
        "",
        "| split | bits/byte | note |",
        "|---|---|---|",
        f"| validation | {_fmt_ci(val)} | top-1 ECE {val.get('calibration', {}).get('ece', float('nan')):.3f} |",
    ]
    if test and "test" in test["results"]:
        lines.append(f"| test | {_fmt_ci(test['results']['test'])} | opened once on {test['evaluated'][:10]}, "
                     "logged in reports/test_access_log.jsonl |")
    else:
        lines.append("| test | not opened | the test split stays locked until `morrowglass final-test` |")
    verdicts = _verdicts(run_id)
    if verdicts:
        lines += ["", "## Pre-registered experiments this run belongs to", "",
                  "Hypotheses and criteria were written before the runs (`src/morrowglass/experiments/registry.py`).",
                  "", "| experiment | verdict | detail |", "|---|---|---|"]
        lines += [f"| {v['title']} | **{v['verdict']}** | {v['detail'][:160]} |" for v in verdicts]
    sv = _serving(run_id)
    if sv:
        lines += ["", "## Serving measurements", ""]
        if "kv" in sv:
            k = sv["kv"]
            lines += [f"KV cache (`{k['command']}`, {k['measured'][:10]}, {k['hardware']['chip']}, "
                      f"{k['threads']} threads, batch 1, greedy, {k['protocol']['new_tokens']} new tokens). "
                      f"Registered verdict: **{k['verdict']}**.", "",
                      "| device | prompt tokens | tokens/s, no cache | tokens/s, KV cache | speed-up |", "|---|---|---|---|---|"]
            for dev in ("cpu", "mps"):
                for L in k["protocol"]["prompt_tokens"]:
                    rr = [r for r in k["rows"] if r["device"] == dev and r["prompt_tokens"] == L]
                    if len(rr) == 2:
                        off, on = sorted(rr, key=lambda r: r["kv_cache"])
                        lines.append(f"| {dev} | {L} | {off['tokens_per_s']} | {on['tokens_per_s']} | "
                                     f"{on['tokens_per_s'] / off['tokens_per_s']:.2f}x |")
        if "int8" in sv:
            q = sv["int8"]
            lines += ["", f"Dynamic int8 (CPU, `{q['command']}`): val bpb {q['fp32']['val_bpb']:.4f} -> "
                          f"{q['int8']['val_bpb']:.4f} ({q['delta_val_bpb']:+.4f}); decode {q['fp32']['tokens_per_s']} -> "
                          f"{q['int8']['tokens_per_s']} tokens/s; weights {q['fp32']['weights_mb']} -> "
                          f"{q['int8']['weights_mb']} MB. Registered verdict: **{q['verdict']}**."]
        if "lm_eval" in sv:
            e = sv["lm_eval"]
            lines += ["", "EleutherAI lm-evaluation-harness (0-shot, through the repo's OpenAI-compatible endpoint; "
                          "expected to be weak, see limitations):", "",
                      "| task | metric | value |", "|---|---|---|",
                      f"| lambada_openai | acc | {e['lambada_acc']:.4f} |",
                      f"| wikitext (wikitext-2) | bits/byte | {e['wikitext_bits_per_byte']:.3f} |",
                      f"| our Shakespeare validation split | bits/byte | {e['shakespeare_val_bits_per_byte']:.3f} "
                      f"(own evaluator: {e['own_val_bpb']:.3f}) |"]
    lines += [
        "",
        "## Intended use",
        "",
        "- Teaching and measurement: attention inspection, training-choice ablations, serving experiments "
        "(KV cache, quantization, OpenAI-compatible API) on a model small enough to run on a laptop CPU.",
        "- Out of scope: any user-facing generation, factual answers, chat, or text in modern English.",
        "",
        "## Limitations",
        "",
        "- About 2M parameters trained on ~4.5 MB of one author's text: output is pastiche with frequent "
        "nonsense; it loses the thread within a sentence.",
        f"- Context is {mc['block_size']} tokens; longer prompts are cut to the last {mc['block_size']}.",
        "- Probabilities are model scores, not calibrated confidence for individual tokens.",
        "- No safety tuning. Shakespeare's plays contain period violence and prejudice, which samples can reproduce.",
        "- One training seed for this run; MPS training is not bit-reproducible.",
        "",
        "## How to load",
        "",
        "```python",
        "from morrowglass.hub import MorrowglassHubLM   # pip install from the project repo",
        f'model = MorrowglassHubLM.from_pretrained("path/to/{name}")  # or the Hub repo id once published',
        "```",
        "",
        "Or serve it with an OpenAI-compatible API: `uv run morrowglass serve --model " + run_id + "`.",
        "",
        "## Licence",
        "",
        "Code and weights: MIT. Training text: public domain.",
        "",
    ]
    return "\n".join(lines)


@torch.no_grad()
def export_hf(run_id: str, out_root: Path = EXPORT_ROOT, card_root: Path | None = CARD_ROOT,
              runs_dir: Path = RUNS_DIR, ckpt_root: Path = CHECKPOINT_ROOT) -> dict[str, Any]:
    ck = ckpt_root / run_id / "best.pt"
    base, state = load_model_from_checkpoint(ck)
    hub = MorrowglassHubLM(base.cfg)
    hub.load_state_dict(base.state_dict())
    hub.eval()
    out = out_root / run_id
    if out.exists():
        shutil.rmtree(out)
    hub.save_pretrained(out)
    write_verified(out / "README.md", model_card(run_id, runs_dir).encode())
    tok_name = state["tokenizer"]
    if tok_name.startswith("bpe"):
        dataset = read_json(runs_dir / run_id / "config.json")["train"]["dataset"]
        shutil.copyfile(REPO_ROOT / "artifacts" / "tokenizers" / dataset / f"{tok_name}.json", out / "tokenizer.json")
    # Round trip: reload from disk and compare logits on a fixed input.
    reloaded = MorrowglassHubLM.from_pretrained(out)
    x = torch.arange(min(64, base.cfg.block_size)).remainder(base.cfg.vocab_size).unsqueeze(0)
    diff = float((reloaded(x).logits - base(x).logits).abs().max())
    info = {
        "run_id": run_id, "exported": now_iso(), "export_git": git_provenance(),
        "format": "huggingface_hub PyTorchModelHubMixin (model.safetensors + config.json)",
        "load": "morrowglass.hub.MorrowglassHubLM.from_pretrained(<folder or repo id>)",
        "tokenizer": {"name": tok_name, "sha256": state["tokenizer_sha256"],
                      "file": "tokenizer.json" if tok_name.startswith("bpe") else None},
        "source_checkpoint_sha256": json.loads(ck.with_suffix(".sha256.json").read_text())["sha256"],
        "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                  for p in sorted(out.iterdir()) if p.is_file()},
        "roundtrip_max_abs_logit_diff": diff,
        "published": False,
        "publish_command_after_owner_approval": publish_command(run_id),
    }
    write_json(out / "morrowglass.json", info)
    if card_root is not None:  # small text files only (weights stay out of git)
        cd = card_root / run_id
        cd.mkdir(parents=True, exist_ok=True)
        for f in ("README.md", "config.json", "morrowglass.json"):
            shutil.copyfile(out / f, cd / f)
    return info
