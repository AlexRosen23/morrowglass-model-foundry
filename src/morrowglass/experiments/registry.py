"""Pre-registered experiments: hypothesis, arms (training configs), and success criterion.

Each experiment is defined *before* it runs. ``analyze.py`` computes the verdict from the
saved run records using the criterion written here, so the conclusion cannot be tuned
after seeing the numbers. Profiles scale budgets: ``smoke`` (seconds, offline sample),
``standard`` (minutes on an Apple-silicon laptop; what the reports use) and ``extended``
(longer; requires explicit approval because of runtime on a shared machine).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from morrowglass.train.trainer import TrainConfig

PROFILES = ("smoke", "standard", "extended")

MODEL_SIZES: dict[str, dict[str, Any]] = {
    "nano": {"n_layer": 1, "n_head": 2, "d_model": 32, "block_size": 64},
    "tiny": {"n_layer": 2, "n_head": 4, "d_model": 96, "block_size": 256},
    "small": {"n_layer": 4, "n_head": 4, "d_model": 192, "block_size": 256},
    "medium": {"n_layer": 6, "n_head": 8, "d_model": 256, "block_size": 256},
}


@dataclass
class Experiment:
    name: str
    title: str
    hypothesis: str
    criterion: str  # human-readable, mirrored by the code in analyze.py
    arms: list[TrainConfig] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    track: str = "language-model"


def _lm(name: str, experiment: str, size: str, profile: str, **kw: Any) -> TrainConfig:
    steps = {"smoke": 40, "standard": 2500, "extended": 10000}[profile]
    base = TrainConfig(
        name=name, experiment=experiment, task="lm",
        dataset="shakespeare-sample" if profile == "smoke" else "shakespeare",
        tokenizer="byte", model=dict(MODEL_SIZES["nano" if profile == "smoke" else size]),
        batch_size=32, max_steps=steps, lr=3e-3, warmup_steps=min(100, steps // 10),
        eval_interval=max(10, steps // 25), checkpoint_interval=max(20, steps // 5),
        log_interval=max(5, steps // 125), profile=profile,
    )
    return replace(base, **kw)


def build(name: str, profile: str = "standard") -> Experiment:
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    p = profile
    if name == "scratch":
        return Experiment(
            name, "Scratch Transformer vs count-based baselines",
            "A 1.9M-parameter byte-level Transformer trained from scratch reaches lower held-out "
            "bits-per-byte than the best interpolated Kneser-Ney n-gram (order chosen on validation).",
            "Transformer val bpb upper 95% CI < best n-gram val bpb lower 95% CI.",
            [_lm("lm-byte-small", name, "small", p)],
            ["Single seed for the flagship run (seed variation is measured in the optimization experiment).",
             "Validation plays are other Shakespeare plays: same author and era, so this is in-domain."],
        )
    if name == "tokenization":
        return Experiment(
            name, "Byte vs BPE-1024 tokenization",
            "At an equal optimizer-step and tokens-per-step budget, BPE-1024 reaches lower val "
            "bits-per-byte than byte-level, because each step covers more text.",
            "BPE val bpb upper CI < byte val bpb lower CI.",
            [_lm("lm-byte-small", name, "small", p), _lm("lm-bpe1024-small", name, "small", p, tokenizer="bpe1024")],
            ["Not apples-to-apples in text seen: equal tokens per step means BPE sees ~3x more bytes.",
             "BPE adds (1024-256) x 192 embedding parameters; the comparison is per step, not per parameter.",
             "BPE merges are learned on the train split only."],
        )
    if name == "capacity":
        arms = [_lm(f"lm-byte-{s}", name, s, p) for s in ("tiny", "small", "medium")]
        arms += [_lm(f"scarce-byte-{s}", name, s, p, dataset="shakespeare-sample",
                     max_steps={"smoke": 40, "standard": 1500, "extended": 3000}[p],
                     eval_interval={"smoke": 10, "standard": 50, "extended": 100}[p])
                 for s in ("tiny", "medium")]
        return Experiment(
            name, "Model capacity at equal token budget (full data vs 240 KB scarce data)",
            "On the full 4.5 MB corpus, larger models reach lower val bpb at an equal token budget; "
            "on the 240 KB sample the medium model overfits (val loss rises after its minimum while "
            "train loss keeps falling).",
            "Full data: bpb(medium) < bpb(small) < bpb(tiny). Scarce data: medium's final val loss "
            "exceeds its best val loss by > 0.05 nats while its train loss keeps decreasing.",
            arms,
            ["Equal tokens, not equal FLOPs: the medium model spends ~4x the compute per token.",
             "One seed per arm."],
        )
    if name == "optimization":
        arms = []
        steps = {"smoke": 30, "standard": 1000, "extended": 3000}[p]
        for seed in (0, 1, 2):
            arms += [
                _lm("opt-lr1e-3-cos", name, "tiny", p, lr=1e-3, max_steps=steps, seed=seed),
                _lm("opt-lr3e-3-cos", name, "tiny", p, lr=3e-3, max_steps=steps, seed=seed),
                _lm("opt-lr3e-3-const-nowarm", name, "tiny", p, lr=3e-3, max_steps=steps, seed=seed,
                    schedule="constant", warmup_steps=0),
            ]
        for a in arms:
            a.eval_interval = max(10, steps // 20)
            a.checkpoint_interval = steps
        return Experiment(
            name, "Learning rate and schedule, 3 seeds each",
            "For the tiny model at 1000 steps, lr 3e-3 with warmup+cosine beats lr 1e-3 cosine and "
            "beats lr 3e-3 constant without warmup, by more than the seed-to-seed range.",
            "Mean val bpb ordered as hypothesised and the [min, max] ranges across seeds do not overlap.",
            arms,
            ["Three seeds give a range, not a confidence interval.",
             "Seeds change both initialisation and batch order."],
        )
    if name == "digitsum":
        steps = {"smoke": 40, "standard": 3000, "extended": 8000}[p]
        model = {"n_layer": 2, "n_head": 4, "d_model": 128, "block_size": 48} if p != "smoke" else \
            {"n_layer": 1, "n_head": 2, "d_model": 32, "block_size": 48}
        arms = []
        for seed in (0, 1):
            for aug in (False, True):
                arms.append(TrainConfig(
                    name=f"digitsum-{'posaug' if aug else 'plain'}", experiment=name, task="digitsum",
                    dataset="synthetic", tokenizer="byte", model=dict(model), batch_size=128,
                    max_steps=steps, lr=1e-3, warmup_steps=min(200, steps // 10), weight_decay=0.1,
                    eval_interval=max(10, steps // 30), checkpoint_interval=steps, log_interval=max(5, steps // 100),
                    seed=seed, pos_augment=aug, profile=p))
        return Experiment(
            name, "Algorithmic transfer: digit-sum mod 10 under length shift",
            "A 2-layer model learns the rule in-distribution (3-6 terms, > 95% val accuracy) but fails "
            "to transfer to longer inputs (8-10 and 12-16 terms); random position offsets during "
            "training reduce but do not remove the gap.",
            "Plain arm: val_iid acc > 0.95 and test_ood_far acc < 0.5. Pos-aug arm: OOD accuracy higher "
            "than plain arm for both seeds.",
            arms,
            ["Chance accuracy is 10%.", "Synthetic success is not evidence of general reasoning."],
            track="algorithmic",
        )
    if name == "digitsum-scratchpad":
        steps = {"smoke": 40, "standard": 3000, "extended": 8000}[p]
        model = {"n_layer": 2, "n_head": 4, "d_model": 128, "block_size": 72} if p != "smoke" else \
            {"n_layer": 1, "n_head": 2, "d_model": 32, "block_size": 72}
        arms = []
        for seed in (0, 1):
            for aug in (False, True):
                arms.append(TrainConfig(
                    name=f"digitsum-sp-{'posaug' if aug else 'plain'}", experiment=name,
                    task="digitsum-scratchpad", dataset="synthetic", tokenizer="byte", model=dict(model),
                    batch_size=128, max_steps=steps, lr=1e-3, warmup_steps=min(200, steps // 10),
                    weight_decay=0.1, eval_interval=max(10, steps // 15), checkpoint_interval=steps,
                    log_interval=max(5, steps // 100), seed=seed, pos_augment=aug, profile=p))
        return Experiment(
            name, "Algorithmic transfer, follow-up: running-sum scratchpad under length shift",
            "Registered 2026-11-08 AFTER the plain digit-sum runs stayed at chance (loss = ln 10). With an "
            "interleaved running-sum scratchpad ('3,3;9,2;4,6;'), every step is local, so the 2-layer model "
            "learns the rule in-distribution (> 95% final-answer accuracy, autoregressive, on 3-6 terms). "
            "With learned absolute positions it still fails on longer inputs (12-16 terms < 50%) because "
            "those positions were never trained; random position offsets during training raise OOD "
            "accuracy for both seeds.",
            "Plain-position arms: val_iid acc > 0.95 and test_ood_far acc < 0.5. Pos-aug arms: OOD near and "
            "far accuracy higher than the plain arm with the same seed.",
            arms,
            ["Follow-up designed after seeing a failure; the original digit-sum experiment is kept and reported.",
             "Evaluation is autoregressive: the model's own running sums are fed back; read-out restricted "
             "to digit tokens.", "Chance accuracy is 10%.",
             "Synthetic success is not evidence of general reasoning."],
            track="algorithmic",
        )
    if name == "break-lr":
        steps = {"smoke": 20, "standard": 400, "extended": 400}[p]
        common = dict(max_steps=steps, eval_interval=max(10, steps // 10), checkpoint_interval=steps)
        return Experiment(
            name, "Break the Model: learning rate far too high",
            "lr=0.1 with no warmup and no clipping diverges or stalls far above the baseline; adding "
            "warmup and clipping alone does not rescue it; returning to lr=3e-3 does.",
            "Broken arm: diverged or best val bpb > 3.0. Fix-1: reported as measured. Fix-2: best val bpb "
            "< broken arm's by > 0.5.",
            [_lm("break-lr-broken", name, "tiny", p, lr=0.1, warmup_steps=0, grad_clip=0.0,
                 schedule="constant", nan_policy="abort", **common),
             _lm("break-lr-fix1-warmup-clip", name, "tiny", p, lr=0.1, warmup_steps=steps // 4,
                 grad_clip=1.0, **common),
             _lm("break-lr-fix2-lr3e-3", name, "tiny", p, lr=3e-3, warmup_steps=steps // 4, **common)],
            ["Short runs (400 steps) by design."],
        )
    if name == "break-mask":
        steps = {"smoke": 20, "standard": 400, "extended": 400}[p]
        common = dict(max_steps=steps, eval_interval=max(10, steps // 10), checkpoint_interval=steps)
        tiny_nc = {**MODEL_SIZES["nano" if p == "smoke" else "tiny"], "causal": False}
        return Experiment(
            name, "Break the Model: remove the causal mask",
            "Without the mask the training loss collapses (the model reads the answer from the "
            "future), but on leak-free last-position scoring it is far worse than the causal model.",
            "Non-causal train loss < causal train loss AND non-causal leak-free bpb > causal leak-free bpb.",
            [_lm("break-mask-noncausal", name, "tiny", p, model=tiny_nc, **common),
             _lm("break-mask-causal", name, "tiny", p, **common)],
            ["The usual windowed val metric is itself leaky for a non-causal model; analysis uses "
             "last-position-only scoring, which cannot see the future for either model."],
        )
    if name == "break-mask-long":
        steps = {"smoke": 30, "standard": 2000, "extended": 4000}[p]
        common = dict(max_steps=steps, eval_interval=max(10, steps // 20), checkpoint_interval=steps)
        tiny_nc = {**MODEL_SIZES["nano" if p == "smoke" else "tiny"], "causal": False}
        return Experiment(
            name, "Break the Model, follow-up: no causal mask for 2,000 steps",
            "Registered 2026-11-08 AFTER break-mask (400 steps) showed no collapse. Given 5x more steps, the "
            "mask-less model finds the copy-from-the-future shortcut: its training loss falls well below the "
            "causal model's, while leak-free last-position bpb is worse than the causal model's.",
            "Non-causal train loss < causal train loss AND non-causal leak-free bpb > causal leak-free bpb.",
            [_lm("break-mask-long-noncausal", name, "tiny", p, model=tiny_nc, **common),
             _lm("break-mask-long-causal", name, "tiny", p, **common)],
            ["Follow-up designed after seeing break-mask; that result is kept and reported.",
             "Leak-free scoring uses only the final position of each window."],
        )
    if name == "break-overfit":
        steps = {"smoke": 40, "standard": 1500, "extended": 3000}[p]
        return Experiment(
            name, "Break the Model: memorising a 240 KB corpus, then a regularisation fix",
            "Medium model on the scarce sample overfits; dropout 0.2 + early stopping (patience 4) "
            "gives a lower best val loss and stops before the overfit regime.",
            "Fix arm best val loss < broken arm best val loss, and fix arm stops early.",
            [_lm("scarce-byte-medium", name, "medium", p, dataset="shakespeare-sample", max_steps=steps,
                 eval_interval={"smoke": 10, "standard": 50, "extended": 100}[p]),
             _lm("scarce-byte-medium-dropout-es", name, "medium", p, dataset="shakespeare-sample",
                 max_steps=steps, eval_interval={"smoke": 10, "standard": 50, "extended": 100}[p],
                 model={**MODEL_SIZES["nano" if p == "smoke" else "medium"], "dropout": 0.2},
                 early_stopping_patience=4)],
            ["The broken arm is shared with the capacity experiment (same config hash)."],
        )
    raise KeyError(f"unknown experiment {name!r}; known: {', '.join(ALL)}")


ALL = ["scratch", "tokenization", "capacity", "optimization", "digitsum", "digitsum-scratchpad", "break-lr",
       "break-mask", "break-mask-long", "break-overfit"]


# ---------------------------------------------------------------------------------------------------------------
# Serving and evaluation studies on already-trained checkpoints (no training). Each is registered here, with its
# hypothesis and success criterion, and committed BEFORE its first measurement. The code that measures it computes
# the verdict from the criterion below; it never reads the test split (validation text only).

FLAGSHIP_RUN = "lm-byte-small-2822813d-s0"  # pre-registered flagship, standard budget
EXTENDED_RUN = "lm-byte-small-240f8c28-s0"  # same model, extended budget (owner-approved); best scratch val bpb
BEST_BPE_RUN = "lm-bpe1024-small-eb29b8fb-s0"  # best scratch model on validation at the standard budget


@dataclass(frozen=True)
class Evaluation:
    name: str
    title: str
    registered: str
    question: str
    hypothesis: str
    criterion: str
    runs: tuple[str, ...]
    protocol: dict[str, Any]
    caveats: tuple[str, ...]
    output: str


EVALUATIONS: dict[str, Evaluation] = {
    "kv-cache": Evaluation(
        "kv-cache", "KV cache: equivalence and decode speed", "2026-11-09",
        "Does caching keys/values change what the model computes, and how much faster is autoregressive decoding?",
        "Cached greedy decoding of a real checkpoint produces the same tokens as uncached decoding with logits "
        "within 1e-4 on CPU. On CPU, cached decoding has higher tokens/s than uncached at every prompt length, "
        "and the speed-up at the longest prompt is larger than at the shortest.",
        "CPU: identical greedy tokens AND max |logit difference| < 1e-4 AND speedup > 1 at every prompt length AND "
        "speedup(longest) > speedup(shortest). MPS numbers are reported without a criterion.",
        (FLAGSHIP_RUN, BEST_BPE_RUN),
        {"devices": ["cpu", "mps"], "prompt_tokens": [16, 64, 192], "new_tokens": 64, "decoding": "greedy",
         "batch": 1, "repeats": 5, "statistic": "median over repeats", "torch_threads": 2,
         "prompt_source": "start of the validation split (never test)"},
        ("Shared machine with other workloads; numbers are indicative, not a controlled benchmark.",
         "prompt + new tokens stay within block_size (256): beyond it the window slides and every step recomputes "
         "(the cache cannot be reused with learned absolute positions)."),
        "reports/serving/kv-cache/<run_id>.json",
    ),
    "lm-eval-harness": Evaluation(
        "lm-eval-harness", "Standard harness (EleutherAI lm-evaluation-harness) through the OpenAI-compatible API",
        "2026-11-09",
        "Do standard harness numbers come out of our own /v1/completions endpoint, do they agree with the "
        "project's own evaluator, and where does a 2M-parameter Shakespeare model sit on general-English tasks?",
        "(a) Plumbing: for each scratch model, the harness's bits_per_byte on our validation split "
        "(custom task morrowglass_shakespeare_val) is within 0.03 of that run's own val bpb in summary.json. "
        "(b) Expected weakness: every scratch model scores lambada_openai accuracy < 0.05 and below distilgpt2, "
        "and a higher wikitext bits_per_byte than distilgpt2.",
        "(a) |harness bpb - summary val bpb| < 0.03 for every scratch run. (b) lambada acc < 0.05 and "
        "< distilgpt2's, wikitext bpb > distilgpt2's, for every scratch run. If distilgpt2 cannot run, (b) is "
        "judged on the 0.05 bound alone and says so.",
        (FLAGSHIP_RUN, EXTENDED_RUN, BEST_BPE_RUN, "hf:distilgpt2"),
        {"harness": "lm-eval==0.4.13 (eval/lm-eval-requirements.txt, fully pinned)",
         "model_type": "local-completions -> POST /v1/completions on 127.0.0.1 (echo + logprobs)",
         "tokenizer": "remote (/tokenize, /detokenize, /tokenizer_info served by the same API)",
         "tasks": {"lambada_openai": "all 5,153 docs", "wikitext": "wikitext-2 document-level test, all 62 docs",
                   "morrowglass_shakespeare_val": "our validation split as one document (eval/tasks/)"},
         "max_length": {"scratch": 256, "hf:distilgpt2": 1024}, "batch_size": 1, "num_concurrent": 1,
         "prefix_token": "newline (the scratch tokenizers have no BOS/EOS)", "fewshot": 0},
        ("lambada needs the exact last word of an English news/fiction passage; the scratch models saw only "
         "Shakespeare and, at byte level, 256 bytes of context, so near-zero accuracy is expected.",
         "distilgpt2 sees 1,024 BPE tokens of context; the scratch byte models see 256 bytes. Not a fair fight, "
         "and not meant to be: the point is comparable plumbing.",
         "distilgpt2's pretraining data (OpenWebText) may overlap with Wikipedia-derived and Shakespeare text."),
        "reports/lm_eval/<model>/ (raw harness output + samples) and reports/lm_eval/summary.json",
    ),
    "int8-dynamic": Evaluation(
        "int8-dynamic", "Dynamic int8 quantization on CPU: quality vs latency", "2026-11-09",
        "What does post-training dynamic int8 quantization of the Linear layers cost in quality and buy in speed?",
        "Dynamic int8 (torch.ao quantize_dynamic, qnnpack, nn.Linear only) changes validation bpb by less than "
        "0.01 and raises cached greedy decode tokens/s on CPU.",
        "|val bpb(int8) - val bpb(fp32)| < 0.01 AND decode tokens/s(int8) > tokens/s(fp32), per run.",
        (FLAGSHIP_RUN, BEST_BPE_RUN),
        {"device": "cpu", "engine": "qnnpack", "modules": "nn.Linear (qkv, proj, mlp, lm_head); embeddings fp32",
         "val_bpb": "full validation split, the project's own windowed evaluator",
         "decode": "cached greedy, 64-token prompt + 64 new tokens, 5 repeats, median", "torch_threads": 2},
        ("torch.ao.quantization is deprecated in torch 2.15 (torchao is the successor); it still runs here.",
         "At ~2M parameters the model is small enough that quantize/dequantize overhead can outweigh the savings."),
        "reports/serving/int8/<run_id>.json",
    ),
}
