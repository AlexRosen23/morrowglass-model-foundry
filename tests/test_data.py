"""Split integrity and leakage guards for the corpus and the synthetic task."""

from __future__ import annotations

import json

import pytest

from morrowglass.data import corpus, synthetic
from morrowglass.repro import REPO_ROOT, sha256_file

SAMPLE = corpus.SAMPLE_DIR
MANIFEST = REPO_ROOT / "data" / "manifests" / "shakespeare.json"


def test_sample_hashes_match_manifest() -> None:
    m = json.loads((SAMPLE / "manifest.json").read_text())
    for split, meta in m["files"].items():
        assert sha256_file(SAMPLE / meta["path"]) == meta["sha256"], split


def test_full_manifest_splits_are_disjoint_by_work() -> None:
    m = json.loads(MANIFEST.read_text())
    works = {k: set(v["works"]) for k, v in m["files"].items()}
    assert not works["train"] & works["val"]
    assert not works["train"] & works["test"]
    assert not works["val"] & works["test"]
    assert sum(len(v) for v in works.values()) == m["n_works"]
    # Poems never become held-out data.
    assert not (works["val"] | works["test"]) & corpus.POEMS


def test_full_manifest_line_overlap_is_small_and_recorded() -> None:
    m = json.loads(MANIFEST.read_text())
    for k in ("val_vs_train", "test_vs_train"):
        assert m["overlap"][k]["fraction"] < 0.01


def test_split_assignment_is_deterministic_and_seed_sensitive() -> None:
    works = [corpus.Work(f"PLAY {i}", f"text {i}\n") for i in range(20)]
    a = corpus.assign_splits(works, seed=1)
    b = corpus.assign_splits(works, seed=1)
    c = corpus.assign_splits(works, seed=2)
    assert [w.title for w in a["val"]] == [w.title for w in b["val"]]
    assert [w.title for w in a["val"] + a["test"]] != [w.title for w in c["val"] + c["test"]]


def test_normalize_and_strip() -> None:
    raw = "x\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\nIt’s “fine” — ok   \r\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\r\n"
    body = corpus.strip_gutenberg(corpus.normalize(raw))
    assert "It's \"fine\" -- ok" in body and "\r" not in body and "GUTENBERG" not in body


def test_strip_requires_markers() -> None:
    with pytest.raises(ValueError):
        corpus.strip_gutenberg("no markers here")


def test_line_overlap_detects_copied_lines() -> None:
    train = "To be, or not to be, that is the question:\nshort\n"
    other = "To be, or not to be, that is the question:\nA different line that is long enough here.\n"
    r = corpus.line_overlap(train, other, min_len=20)
    assert r["lines_checked"] == 2 and r["lines_seen_in_train"] == 1


def test_synthetic_no_expression_leaks_across_splits() -> None:
    small = {"train": (3, 6, 3000), "val_iid": (3, 6, 300), "test_ood_near": (8, 10, 300), "test_ood_far": (12, 16, 300)}
    ex = synthetic.generate(seed=7, splits=small)
    sets = {k: {e.expr for e in v} for k, v in ex.items()}
    names = list(sets)
    for i, a in enumerate(names):
        assert len(sets[a]) == len(ex[a]), f"duplicates inside {a}"
        for b in names[i + 1 :]:
            assert not sets[a] & sets[b], f"{a} and {b} share expressions"


def test_synthetic_lengths_and_labels() -> None:
    ex = synthetic.generate(seed=3, splits={"train": (3, 6, 500), "val_iid": (3, 6, 50),
                                            "test_ood_near": (8, 10, 50), "test_ood_far": (12, 16, 50)})
    for name, (lo, hi, _) in {"train": (3, 6, 0), "test_ood_far": (12, 16, 0)}.items():
        for e in ex[name]:
            terms = e.expr.split("+")
            assert lo <= len(terms) <= hi
            assert int(e.answer) == sum(map(int, terms)) % 10


def test_synthetic_encoding_scores_only_the_answer() -> None:
    e = synthetic.Example("3+9+4", "6")
    x, y = synthetic.encode_examples([e], block_size=16)
    assert (y != -100).sum() == 1
    pos = int((y[0] != -100).nonzero())
    assert chr(int(x[0, pos])) == "=" and chr(int(y[0, pos])) == "6"


def test_synthetic_is_reproducible() -> None:
    a = synthetic.generate(seed=11, splits={"train": (3, 6, 100), "val_iid": (3, 6, 10),
                                            "test_ood_near": (8, 10, 10), "test_ood_far": (12, 16, 10)})
    b = synthetic.generate(seed=11, splits={"train": (3, 6, 100), "val_iid": (3, 6, 10),
                                            "test_ood_near": (8, 10, 10), "test_ood_far": (12, 16, 10)})
    assert synthetic.split_hash(a["train"]) == synthetic.split_hash(b["train"])
