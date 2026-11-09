"""Public-domain corpus pipeline: Project Gutenberg eBook #100 (Shakespeare, Complete Works).

Provenance and licence are recorded in ``docs/DATA_CARD.md``. The text of the works is in
the public domain in the United States. We strip the Project Gutenberg header, footer and
licence boilerplate (as the PG licence permits when the trademark is not used), so the
processed files contain only Shakespeare's public-domain text.

Splits are made **by work** (whole plays), never by random lines, so that validation and
test text comes from plays the model never saw during training. Poems always go to train.
"""

from __future__ import annotations

import random
import re
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from morrowglass.repro import REPO_ROOT, now_iso, sha256_bytes, sha256_file, write_json, write_verified

GUTENBERG_ID = 100
SOURCE_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
RAW_PATH = REPO_ROOT / "data" / "raw" / "pg100.txt"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "shakespeare"
SAMPLE_DIR = REPO_ROOT / "data" / "sample" / "shakespeare"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
SPLIT_SEED = 1729
N_VAL_WORKS = 3
N_TEST_WORKS = 3

POEMS = {
    "THE SONNETS",
    "A LOVER'S COMPLAINT",
    "THE PASSIONATE PILGRIM",
    "THE PHOENIX AND THE TURTLE",
    "THE RAPE OF LUCRECE",
    "VENUS AND ADONIS",
}

_PUNCT_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "—": "--",
        "–": "-",
        "…": "...",
        " ": " ",
    }
)


@dataclass(frozen=True)
class Work:
    title: str
    text: str

    @property
    def kind(self) -> str:
        return "poem" if self.title in POEMS else "play"


def fetch_raw(dest: Path = RAW_PATH, url: str = SOURCE_URL) -> Path:
    """Download the single source file (5.6 MB) if it is not already present."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "morrowglass-corpus-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https URL
        data = resp.read()
    write_verified(dest, data)
    return dest


def normalize(text: str) -> str:
    """CRLF->LF, NFC, typographic punctuation -> ASCII, trailing spaces trimmed."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text).translate(_PUNCT_MAP)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{4,}", "\n\n\n", text)


def strip_gutenberg(text: str) -> str:
    start = re.search(r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*\*\*\*$", text, re.M)
    end = re.search(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*\*\*\*$", text, re.M)
    if not start or not end:
        raise ValueError("Project Gutenberg START/END markers not found; unexpected source file")
    return text[start.end() : end.start()]


def split_works(body: str) -> list[Work]:
    """Split the body into works using the table of contents titles."""
    lines = body.split("\n")
    try:
        contents_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "Contents")
    except StopIteration as exc:
        raise ValueError("table of contents not found") from exc
    titles: list[str] = []
    i = contents_idx + 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    while i < len(lines) and lines[i].strip():
        titles.append(lines[i].strip())
        i += 1
    # Body titles appear un-indented on their own line, in order, after the contents block.
    starts: list[int] = []
    cursor = i
    for title in titles:
        j = cursor
        while j < len(lines) and lines[j] != title:
            j += 1
        if j >= len(lines):
            raise ValueError(f"title {title!r} not found in body")
        starts.append(j)
        cursor = j + 1
    works = []
    for k, title in enumerate(titles):
        end = starts[k + 1] if k + 1 < len(titles) else len(lines)
        text = "\n".join(lines[starts[k] : end]).strip() + "\n"
        works.append(Work(title=title, text=text))
    return works


def assign_splits(works: list[Work], seed: int = SPLIT_SEED) -> dict[str, list[Work]]:
    plays = sorted((w for w in works if w.kind == "play"), key=lambda w: w.title)
    rng = random.Random(seed)
    held = rng.sample(plays, N_VAL_WORKS + N_TEST_WORKS)
    val, test = held[:N_VAL_WORKS], held[N_VAL_WORKS:]
    held_titles = {w.title for w in held}
    train = [w for w in works if w.title not in held_titles]
    return {"train": train, "val": val, "test": test}


def line_overlap(train_text: str, other_text: str, min_len: int = 30) -> dict[str, Any]:
    """Fraction of substantive lines in ``other`` that also occur verbatim in train.

    Verse reuses stock lines and stage directions, so this is not zero; we record it so
    nobody mistakes memorised lines for generalisation.
    """
    train_lines = {ln.strip() for ln in train_text.split("\n") if len(ln.strip()) >= min_len}
    other_lines = [ln.strip() for ln in other_text.split("\n") if len(ln.strip()) >= min_len]
    hits = sum(1 for ln in other_lines if ln in train_lines)
    return {
        "min_line_chars": min_len,
        "lines_checked": len(other_lines),
        "lines_seen_in_train": hits,
        "fraction": round(hits / max(1, len(other_lines)), 5),
    }


def _split_text(ws: list[Work]) -> str:
    return "\n\n".join(w.text for w in ws)


def prepare(
    raw_path: Path = RAW_PATH, out_dir: Path = PROCESSED_DIR, seed: int = SPLIT_SEED
) -> dict[str, Any]:
    """Build train/val/test text files plus a manifest with hashes and overlap stats."""
    raw = raw_path.read_bytes()
    body = strip_gutenberg(normalize(raw.decode("utf-8")))
    works = split_works(body)
    splits = assign_splits(works, seed)
    texts = {name: _split_text(ws) for name, ws in splits.items()}
    titles = {name: [w.title for w in ws] for name, ws in splits.items()}
    overlap_titles = set(titles["train"]) & (set(titles["val"]) | set(titles["test"]))
    if overlap_titles or set(titles["val"]) & set(titles["test"]):
        raise AssertionError(f"a work appears in more than one split: {overlap_titles}")
    files: dict[str, Any] = {}
    for name, text in texts.items():
        data = text.encode("utf-8")
        sha = write_verified(out_dir / f"{name}.txt", data)
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        files[name] = {"path": f"{name}.txt", "bytes": len(data), "chars": len(text), "sha256": sha,
                       "non_ascii_chars": non_ascii, "works": titles[name]}
    manifest = {
        "dataset": "shakespeare",
        "source": {"name": "Project Gutenberg eBook #100: The Complete Works of William Shakespeare",
                   "url": SOURCE_URL, "raw_sha256": sha256_bytes(raw), "raw_bytes": len(raw),
                   "license": "Public domain in the USA; Project Gutenberg boilerplate removed"},
        "processing": ["CRLF->LF", "NFC", "typographic quotes/dashes->ASCII", "strip trailing spaces",
                       "strip PG header/footer", "split by work using table of contents"],
        "split_seed": seed,
        "split_policy": f"{N_VAL_WORKS} val + {N_TEST_WORKS} test plays sampled by seed; poems always train",
        "n_works": len(works),
        "files": files,
        "overlap": {"val_vs_train": line_overlap(texts["train"], texts["val"]),
                    "test_vs_train": line_overlap(texts["train"], texts["test"])},
        "created": now_iso(),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def make_sample(
    processed_dir: Path = PROCESSED_DIR, sample_dir: Path = SAMPLE_DIR,
    sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Cut a small offline sample (checked into git) from the processed splits."""
    sizes = sizes or {"train": 240_000, "val": 40_000, "test": 40_000}
    files = {}
    for name, n in sizes.items():
        text = (processed_dir / f"{name}.txt").read_text(encoding="utf-8")
        cut = text[:n]
        cut = cut[: cut.rfind("\n") + 1]
        sha = write_verified(sample_dir / f"{name}.txt", cut.encode())
        files[name] = {"path": f"{name}.txt", "bytes": len(cut.encode()), "sha256": sha}
    manifest = {"dataset": "shakespeare-sample", "derived_from": str(processed_dir.relative_to(REPO_ROOT)),
                "derived_from_sha256": {k: sha256_file(processed_dir / f"{k}.txt") for k in sizes},
                "policy": "first N characters of each processed split, cut at a line boundary",
                "files": files}
    write_json(sample_dir / "manifest.json", manifest)
    return manifest


def dataset_dir(name: str) -> Path:
    if name == "shakespeare":
        return PROCESSED_DIR
    if name == "shakespeare-sample":
        return SAMPLE_DIR
    raise ValueError(f"unknown dataset {name!r}")


def load_split(name: str, split: str) -> str:
    d = dataset_dir(name)
    path = d / f"{split}.txt"
    if not path.exists():
        hint = "run `make data`" if name == "shakespeare" else "the sample should be in git"
        raise FileNotFoundError(f"{path} missing ({hint})")
    return path.read_text(encoding="utf-8")


def dataset_hash(name: str) -> str:
    """Hash of the dataset manifest's split hashes (identifies the exact data used)."""
    import json

    m = json.loads((dataset_dir(name) / "manifest.json").read_text())
    return sha256_bytes("|".join(f"{k}:{v['sha256']}" for k, v in sorted(m["files"].items())).encode())
