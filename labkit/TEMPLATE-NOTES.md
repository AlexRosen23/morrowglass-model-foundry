# Template notes: how this Localpane Lab Kit is built

This kit copies the layout, the two scripts and the generator of the QUORUM pilot kit (Go) and changes
only `kit.conf`, the marked source and the documents. This file is for the kit author; it is not
shipped.

## Layout

```
labkit/
  kit.conf            name, version, step count, comment syntax, test/compile/lint commands, shipped files, data
  src/                the finished project (Python package scratchlm + tests) with lab markers; author only
  tools/labgen.py     generator: src/ -> solution/ + steps/step-NN/ (Python 3 stdlib; same as the pilot,
                      plus runs/, .pytest_cache and .ruff_cache in SKIP_DIRS)
  verify.sh           proves the fail-on-starter / pass-on-solution matrix (author only)
  WORKBOOK.md README-FIRST.md LICENSE-LABKIT.md CHANGELOG.md THIRD-PARTY.md check.sh   shipped
  LISTING-DRAFT.md    storefront text for the owner to approve (author only)
  cover/              cover.svg source, cover.png 1280x720, render.html (author only)
scripts/package-labkit.sh   verify -> generate -> data -> licence -> hygiene + gitleaks -> deterministic zip + SHA256SUMS
```

The practice text is not duplicated in `labkit/`: `package-labkit.sh` copies the repository's committed
public-domain sample (`data/sample/shakespeare/{train,val}.txt`) into the zip's `data/` (kit.conf
`DATA_FILES`).

## Markers (Python)

```python
    def encode(self, text: str) -> list[int]:
        #lab:begin 1 encode text as UTF-8 byte values
        return list(text.encode("utf-8"))
        #lab:stub
        #> # TODO(step 1): what to do, in two to five lines.
        #> raise NotImplementedError("TODO(step 1): ByteTokenizer.encode")
        #lab:end
```

- Same rules as the pilot: a region belongs to one step; in `steps/step-NN` regions of steps `< NN` are
  solved and the rest stubbed; stub lines start with `#>` and keep their indentation.
- Stubs raise `NotImplementedError("TODO(step N) ...")`. pytest isolates each test, so a raising stub
  never breaks another step's tests. Code after a stub may be unreachable in starters (e.g. the
  `return ModelOutput(...)` in `LanguageModel.forward`); that is why lint runs on the solution only.
- Regions cover function bodies, never signatures or `__init__`, so every starter imports and every
  model can be constructed (the `test_kit_*` tests build models and save checkpoints on step-01).
  For the same reason the sinusoidal table is built lazily in `positions()`, not in `__init__`.
- Step 8 has a region inside the given `CausalSelfAttention.forward`: its stub raises only when the
  cache is requested, so the uncached path that steps 2 to 7 rely on keeps working.

## Test naming contract (pytest)

- `test_stepNN_*`: graded in step NN; must fail on `steps/step-NN` and pass on the solution. One file
  per step, named `tests/test_stepNN_*.py`, so pytest's `-k test_stepNN_` (substring match on test and
  module names) selects exactly that step.
- `test_kit_*` (in `tests/test_kit.py`): code the kit provides; must pass everywhere.
- A test for step NN may use only code from steps `<= NN`. Example: step 7's tests call
  `generate(..., use_cache=False)` explicitly, because the cache is step 8.
- Every training run in a test is a tiny CPU model (1-2 layers, d_model 32, block 16, <= 150 steps):
  about a second each. `tests/conftest.py` caps PyTorch at 2 threads.

## Commands (kit.conf)

pytest's `-k` takes an expression, not a regular expression, so `verify.sh` builds
`test_kit_ or test_step01_ or ...` for "earlier steps". All pytest calls pass `-o addopts=` (the
project's `-q` would otherwise double up and change the collect-only format that `COUNT_CMD` counts)
and `-p no:cacheprovider`, with `PYTHONDONTWRITEBYTECODE=1`, so no cache folders appear in generated
trees. `PY` selects the interpreter (needs torch, pytest and ruff for lint).

## Running

```sh
PY=/path/to/venv/bin/python labkit/verify.sh                    # about 3 minutes under nice on a busy Mac
SKIP_VERIFY=1 scripts/package-labkit.sh                        # after a green verify
```

## Known limits

- "Fails on the starter" is judged per step as a set; for 1.0.0 every graded test fails individually
  on its starter too (see the verify output).
- Only macOS on Apple Silicon with Python 3.12 / PyTorch 2.15.0 has been run. The README says so.
- The hygiene patterns in `package-labkit.sh` are tuned to this owner's machine and accounts.
