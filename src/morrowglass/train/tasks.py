"""Training tasks: what a batch looks like and how a model is evaluated.

``LMTask`` is next-token language modelling on a text corpus. ``SumTask`` is the
synthetic digit-sum track, where only the answer token is scored.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch

from morrowglass.data import synthetic
from morrowglass.data.corpus import dataset_hash, load_split
from morrowglass.data.loader import TokenSplit, encode_split, sample_batch
from morrowglass.eval.metrics import evaluate_lm
from morrowglass.model.transformer import MorrowglassLM
from morrowglass.repro import sha256_bytes
from morrowglass.tokenizer.tokenizers import Tokenizer, get_tokenizer


class Task(Protocol):
    tokenizer: Tokenizer
    primary_metric: str  # key in eval dicts; lower is better

    def batch(self, batch_size: int, block_size: int, gen: torch.Generator) -> tuple[torch.Tensor, ...]: ...

    def evaluate(self, model: MorrowglassLM, device: str, full: bool) -> dict[str, Any]: ...

    def final_eval(self, model: MorrowglassLM, device: str) -> dict[str, Any]: ...

    def test_eval(self, model: MorrowglassLM, device: str) -> dict[str, Any]: ...

    def provenance(self) -> dict[str, Any]: ...


class LMTask:
    primary_metric = "val_bits_per_byte"

    def __init__(self, dataset: str, tokenizer_spec: str, eval_windows: int = 64) -> None:
        self.dataset = dataset
        self.tokenizer = get_tokenizer(tokenizer_spec, dataset)
        self.splits: dict[str, TokenSplit] = {
            s: encode_split(dataset, s, self.tokenizer) for s in ("train", "val", "test")
        }
        self.eval_windows = eval_windows

    def batch(self, batch_size: int, block_size: int, gen: torch.Generator) -> tuple[torch.Tensor, ...]:
        return sample_batch(self.splits["train"].ids, block_size, batch_size, gen)

    def evaluate(self, model: MorrowglassLM, device: str, full: bool = False) -> dict[str, Any]:
        mw = None if full else self.eval_windows
        val = evaluate_lm(model, self.splits["val"], device, max_windows=mw, ci=full)
        tr = evaluate_lm(model, self.splits["train"], device, max_windows=mw or self.eval_windows, ci=False)
        return {
            "val_loss": val["loss_nats_per_token"],
            "val_bits_per_byte": val["bits_per_byte"],
            "train_eval_loss": tr["loss_nats_per_token"],
            "train_eval_bits_per_byte": tr["bits_per_byte"],
        }

    def final_eval(self, model: MorrowglassLM, device: str) -> dict[str, Any]:
        """Validation only. The test split is locked; see ``morrowglass final-test``."""
        return {
            "val": evaluate_lm(model, self.splits["val"], device, ci=True, calibration=True),
            "train_subset": evaluate_lm(model, self.splits["train"], device, max_windows=256, ci=False),
        }

    def test_eval(self, model: MorrowglassLM, device: str) -> dict[str, Any]:
        return {"test": evaluate_lm(model, self.splits["test"], device, ci=True, calibration=True)}

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "dataset_hash": dataset_hash(self.dataset),
            "tokenizer": self.tokenizer.name,
            "tokenizer_sha256": self.tokenizer.fingerprint(),
            "vocab_size": self.tokenizer.vocab_size,
            "token_counts": {k: len(v) for k, v in self.splits.items()},
            "bytes": {k: v.n_bytes for k, v in self.splits.items()},
            "token_ids_sha256": {k: v.sha256 for k, v in self.splits.items()},
        }

    def train_text_sample(self) -> str:
        return load_split(self.dataset, "train")[:2000]


class SumTask:
    """Digit-sum mod 10. Primary metric: in-distribution validation error rate."""

    primary_metric = "val_iid_error"

    def __init__(self, block_size: int, pos_augment: bool = False, scratchpad: bool = False) -> None:
        from morrowglass.tokenizer.tokenizers import ByteTokenizer

        self.tokenizer: Tokenizer = ByteTokenizer()
        self.block_size = block_size
        self.pos_augment = pos_augment
        self.scratchpad = scratchpad
        self.examples = synthetic.generate()
        enc = synthetic.encode_scratch if scratchpad else synthetic.encode_examples
        text = synthetic.scratch_text if scratchpad else (lambda e: e.text)
        self.data = {k: enc(v, block_size) for k, v in self.examples.items()}
        self.lengths = {k: torch.tensor([len(text(e)) - 1 for e in v]) for k, v in self.examples.items()}

    def batch(self, batch_size: int, block_size: int, gen: torch.Generator) -> tuple[torch.Tensor, ...]:
        x, y = self.data["train"]
        idx = torch.randint(0, x.shape[0], (batch_size,), generator=gen)
        if not self.pos_augment:
            return x[idx], y[idx]
        # Position augmentation: shift each example right by a random offset (left-padding with
        # newlines) so every absolute position embedding receives gradient, not only the first ~14.
        xb, yb = x[idx].clone(), y[idx].clone()
        lens = self.lengths["train"][idx]
        max_off = (block_size - lens).clamp(min=0)
        off = (torch.rand(batch_size, generator=gen) * (max_off + 1).float()).long()
        for i in range(batch_size):
            k = int(off[i])
            if k:
                xb[i] = torch.roll(xb[i], k)
                yb[i] = torch.roll(yb[i], k)
        return xb, yb

    @torch.no_grad()
    def _scratch_accuracy(self, model: MorrowglassLM, split: str, device: str) -> dict[str, Any]:
        """Autoregressive: the model's own running sums are fed back; only the digits are given.

        Teacher-forcing the running sums would leak the previous answer into each step, so
        the final-answer accuracy here is the honest end-to-end measure.
        """
        was = model.training
        model.eval()
        exs = self.examples[split]
        by_n: dict[int, list[int]] = {}
        for i, e in enumerate(exs):
            by_n.setdefault(e.expr.count("+") + 1, []).append(i)
        final_ok = steps_ok = steps = 0
        for n_terms, idxs in sorted(by_n.items()):
            digits = torch.tensor([[int(c) for c in exs[i].expr.split("+")] for i in idxs])
            truth = digits.cumsum(1) % synthetic.MOD
            seq = torch.empty((len(idxs), 0), dtype=torch.long)
            preds = []
            for k in range(n_terms):
                step = torch.stack([digits[:, k] + ord("0"), torch.full((len(idxs),), ord(","))], 1)
                seq = torch.cat([seq, step], 1)
                logits = model(seq.to(device)).logits[:, -1].float().cpu()
                digit_logits = logits[:, ord("0") : ord("9") + 1]  # restrict the read-out to digits
                r = digit_logits.argmax(-1)
                preds.append(r)
                seq = torch.cat([seq, torch.stack([r + ord("0"), torch.full((len(idxs),), ord(";"))], 1)], 1)
            P = torch.stack(preds, 1)
            final_ok += int((P[:, -1] == truth[:, -1]).sum())
            steps_ok += int((truth == P).sum())
            steps += P.numel()
        if was:
            model.train()
        acc = final_ok / len(exs)
        return {"accuracy": acc, "error": 1 - acc, "step_accuracy": steps_ok / steps, "n": len(exs),
                "loss": None}  # no likelihood under free-running decoding

    @torch.no_grad()
    def _accuracy(self, model: MorrowglassLM, split: str, device: str) -> dict[str, Any]:
        if self.scratchpad:
            return self._scratch_accuracy(model, split, device)
        was = model.training
        model.eval()
        x, y = self.data[split]
        correct, total, nll = 0, 0, 0.0
        for i in range(0, x.shape[0], 512):
            xb, yb = x[i : i + 512].to(device), y[i : i + 512].to(device)
            logits = model(xb).logits.float()
            mask = yb != -100
            lp = torch.log_softmax(logits[mask], -1)
            tgt = yb[mask]
            nll += float(-lp.gather(-1, tgt[:, None]).sum())
            correct += int((lp.argmax(-1) == tgt).sum())
            total += int(mask.sum())
        if was:
            model.train()
        return {"accuracy": correct / total, "error": 1 - correct / total, "loss": nll / total, "n": total}

    def evaluate(self, model: MorrowglassLM, device: str, full: bool = False) -> dict[str, Any]:
        v = self._accuracy(model, "val_iid", device)
        out = {"val_iid_error": v["error"], "val_iid_accuracy": v["accuracy"], "val_loss": v["loss"]}
        if self.scratchpad:
            out["val_iid_step_accuracy"] = v["step_accuracy"]
        return out

    def final_eval(self, model: MorrowglassLM, device: str) -> dict[str, Any]:
        """Train + in-distribution validation only; the OOD splits are locked test sets."""
        return {k: self._accuracy(model, k, device) for k in ("train", "val_iid")}

    def test_eval(self, model: MorrowglassLM, device: str) -> dict[str, Any]:
        return {k: self._accuracy(model, k, device) for k in self.examples if k.startswith("test")}

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset": "synthetic-digitsum-mod10",
            "dataset_hash": sha256_bytes("|".join(synthetic.split_hash(v) for v in self.examples.values()).encode()),
            "tokenizer": "byte",
            "tokenizer_sha256": self.tokenizer.fingerprint(),
            "vocab_size": 256,
            "split_hashes": {k: synthetic.split_hash(v) for k, v in self.examples.items()},
            "split_sizes": {k: len(v) for k, v in self.examples.items()},
            "task": synthetic.describe(),
            "pos_augment": self.pos_augment,
            "format": "scratchpad (d,r;)*" if self.scratchpad else "plain (expr=answer)",
        }
