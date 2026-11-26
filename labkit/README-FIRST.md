# Read me first: Build a Transformer Language Model From Scratch on Your Laptop

A Localpane Lab Kit, version 1.0.0.

You build `scratchlm`, a small GPT-style (decoder-only) Transformer language model in plain PyTorch, in
8 steps: byte tokenizer and batches, causal self-attention, the Transformer block, the full model,
evaluation in bits per byte, training, sampling and a key/value cache. Each step comes with a starter
(the code so far, with the step's part stubbed out and marked `TODO`), tests that fail until the step
is done, hints, and one command to check your work. The finished project is in `solution/`.

**What you write and what PyTorch provides.** You write the tokenizer, the batching, attention (scores,
mask, softmax, heads), the blocks, the model's forward pass and loss, the parameter count, the
evaluation, the learning-rate schedule, the optimizer setup, the training step, sampling and the KV
cache. PyTorch provides tensors, autograd, `nn.Linear` / `nn.Embedding` / `nn.LayerNorm`, GELU,
cross-entropy and the AdamW optimizer. No `nn.Transformer`, no pretrained weights, no other ML
library. The training loop around your code, checkpoints and the command line are given.

The kit's code is adapted from Morrowglass Model Foundry, an open-source (MIT) miniature
language-model lab by the same author. What this kit adds is the step-by-step path: the workbook, the
starters, the graded tests and the checks.

## Requirements

- **Python 3.10 or later** with **PyTorch** and **pytest**. The kit was tested with Python 3.12.15,
  PyTorch 2.15.0 and pytest 9.2.0 on macOS (Apple Silicon), on the CPU. It uses only long-standing
  PyTorch features, but other versions, Linux and Windows have not been tested for 1.0.0.
- No GPU needed. Everything, including training, runs on the CPU. `eval` and `sample` accept
  `--device`, but only the CPU is tested.
- About 1 GB of disk for PyTorch, and internet access once to install it.
- Comfortable Python. Knowing what a tensor, a matrix product and a gradient are helps; you do not
  need prior Transformer or PyTorch experience beyond the basics.

## Time

About 8 to 10 hours in total, in sessions of an hour or two. Every step has its own estimate in
WORKBOOK.md.

## How to start

```sh
python3 -m venv .venv                     # in the kit folder, once
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r solution/requirements.txt  # torch and pytest, the tested versions
cd steps/step-01
grep -rn "TODO(step 1)" scratchlm
cd ../..
./check.sh 1                              # fails now; passes when step 1 is done
```

On Linux, `pip install torch` downloads a large CUDA build by default. For a CPU-only build, install
torch first with `pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`, then
`pip install pytest==9.2.0`.

Then open WORKBOOK.md at step 1. When `./check.sh 1` passes, continue with step 2, either in the same
folder (search for `TODO(step 2)`) or in `steps/step-02`, which has step 1 solved for you.

If you cannot run `check.sh` (Windows without WSL or Git Bash), run the tests directly inside the step
folder: `python -m pytest -k test_step01_`, then `python -m pytest -k "test_kit_"` for the provided
code.

## What is in the kit

| Path | Contents |
|---|---|
| `WORKBOOK.md` | the 8 steps: goal, concepts in plain words, where the starter code is, tests, hints, check command |
| `steps/step-01` … `steps/step-08` | one complete project per step: earlier steps solved, this step and later ones stubbed |
| `solution/` | the finished project |
| `data/` | practice text: Shakespeare plays and poems (public domain), a training file and a validation file |
| `check.sh` | `./check.sh N [dir]` runs step N's tests, then every earlier step's tests |
| `LICENSE-LABKIT.md` | your licence (single developer) |
| `THIRD-PARTY.md` | the software and text the kit uses, and their licences |
| `CHANGELOG.md` | versions |

Inside each project: `scratchlm/tokenizer.py`, `data.py`, `attention.py`, `model.py`, `evaluate.py`,
`train.py`, `sample.py` (your work, step by step), `checkpoint.py` and `cli.py` (given), and `tests/`
(43 graded `test_stepNN_*` tests and 6 `test_kit_*` tests for the given code).

## Try the finished model (optional, under a minute)

```sh
cd solution
python -m scratchlm train --data ../data/shakespeare-train.txt --val ../data/shakespeare-val.txt \
  --out runs/first --n-layer 2 --d-model 64 --block-size 64 --steps 300 --eval-interval 50
python -m scratchlm eval --ckpt runs/first/best.pt --data ../data/shakespeare-val.txt
python -m scratchlm sample --ckpt runs/first/best.pt --prompt "ROMEO:" --tokens 300
```

What we measured for exactly these commands (Apple Silicon Mac, 2 CPU threads, other work running
on the machine at the same time): a 120,576-parameter model; 300 steps of training in about 12
seconds; validation bits per byte falling from 8.0 (uniform guessing) to 4.54 after 50 steps and 3.66
after 300; 3.59 bits per byte on the full validation file. Its samples already have line breaks,
speaker-name lines and a few real short words, but most words are still misspelt. Your times will
differ with your machine and `--threads`.

## Limits, stated plainly

This is a learning project, not a production model or a training framework. The model is small,
uses a byte-level tokenizer and learned absolute positions, trains on the CPU only, without mixed
precision, dropout or distributed training, and it produces text that looks like Shakespeare only
from a distance. Model probabilities are scores, not calibrated confidence. No guarantees are made
about the quality of text a model trained with the kit produces.

## Help and updates

Updates within 1.x are free through your original download link. Questions or a bug in the kit: use
the contact option on the Localpane storefront page where you bought it.
