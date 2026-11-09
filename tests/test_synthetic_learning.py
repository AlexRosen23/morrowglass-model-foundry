"""Control for the digit-sum track: the SumTask pipeline itself can learn a simple rule.

The standard digit-sum runs stayed at chance (loss = ln 10). This test rules out a pipeline
bug (masking, target alignment, evaluation): with the answer defined as the *first* digit
(a copy rule), the same data encoding, trainer and evaluator must reach high accuracy.
"""

from __future__ import annotations

import torch

from morrowglass.data import synthetic
from morrowglass.model.transformer import ModelConfig, MorrowglassLM
from morrowglass.train.tasks import SumTask


def _copy_rule_task(block: int) -> SumTask:
    task = SumTask.__new__(SumTask)
    from morrowglass.tokenizer.tokenizers import ByteTokenizer

    task.tokenizer = ByteTokenizer()
    task.block_size = block
    task.pos_augment = False
    task.scratchpad = False
    gen = synthetic.generate(seed=5, splits={"train": (3, 6, 4000), "val_iid": (3, 6, 300),
                                             "test_ood_near": (8, 10, 50), "test_ood_far": (12, 16, 50)})
    task.examples = {k: [synthetic.Example(e.expr, e.expr[0]) for e in v] for k, v in gen.items()}
    task.data = {k: synthetic.encode_examples(v, block) for k, v in task.examples.items()}
    task.lengths = {k: torch.tensor([len(e.text) - 1 for e in v]) for k, v in task.examples.items()}
    return task


def test_sumtask_pipeline_learns_a_copy_rule() -> None:
    torch.manual_seed(0)
    task = _copy_rule_task(48)
    model = MorrowglassLM(ModelConfig(vocab_size=256, block_size=48, n_layer=2, n_head=2, d_model=48))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    for _ in range(250):
        x, y = task.batch(64, 48, gen)
        out = model(x, targets=y)
        assert out.loss is not None
        opt.zero_grad()
        out.loss.backward()
        opt.step()
    acc = task.evaluate(model, "cpu")["val_iid_accuracy"]
    assert acc > 0.9, f"pipeline failed to learn a copy rule: {acc:.3f}"


def test_scratchpad_encoding_and_free_running_eval_have_no_answer_leak() -> None:
    ex = synthetic.Example("3+9+4", "6")
    assert synthetic.scratch_text(ex) == "3,3;9,2;4,6;\n"
    x, y = synthetic.encode_scratch([ex], 16)
    targets = [(chr(int(x[0, i])), chr(int(y[0, i]))) for i in range(16) if y[0, i] != -100]
    assert targets == [(",", "3"), (",", "2"), (",", "6")]
    # A model that always predicts '0' must score exactly the share of answers equal to 0.
    task = SumTask(72, scratchpad=True)

    class Zero(torch.nn.Module):
        cfg = ModelConfig(vocab_size=256, block_size=72, n_layer=1, n_head=1, d_model=8)

        def forward(self, idx: torch.Tensor):  # type: ignore[no-untyped-def]
            logits = torch.zeros(idx.shape[0], idx.shape[1], 256)
            logits[..., ord("0")] = 1.0
            return type("O", (), {"logits": logits})()

    r = task._scratch_accuracy(Zero(), "val_iid", "cpu")  # type: ignore[arg-type]
    share_zero = sum(e.answer == "0" for e in task.examples["val_iid"]) / len(task.examples["val_iid"])
    assert abs(r["accuracy"] - share_zero) < 1e-9
