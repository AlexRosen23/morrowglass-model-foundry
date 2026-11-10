"""Records are committed to a public repo: they must not carry the local username or folder layout."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from morrowglass.repro import REPO_ROOT, portable_command, portable_path, scrub_paths

HOME = str(Path.home())


def test_command_names_the_executable_and_keeps_paths_repo_relative() -> None:
    argv = [f"{REPO_ROOT}/.venv/bin/morrowglass", "run", "capacity", "--config", f"{REPO_ROOT}/configs/x.json"]
    assert portable_command(argv) == "morrowglass run capacity --config configs/x.json"
    assert portable_command([]) == ""


def test_paths_outside_the_repo_lose_the_home_dir_and_package_prefix() -> None:
    lm = f"{HOME}/Development/wt/.venv-lmeval/lib/python3.12/site-packages/lm_eval/tasks/wikitext/wikitext.yaml"
    assert portable_path(lm) == "lm_eval/tasks/wikitext/wikitext.yaml"
    assert portable_path(f"{HOME}/elsewhere/ckpt.pt") == "~/elsewhere/ckpt.pt"
    assert portable_path("eval/tasks/x.yaml") == "eval/tasks/x.yaml"
    nested = {"a": [f"{REPO_ROOT}/reports/runs/r", 3], "b": {"c": f"{HOME}/x"}}
    assert scrub_paths(nested) == {"a": ["reports/runs/r", 3], "b": {"c": "~/x"}}


def test_committed_records_name_no_local_paths() -> None:
    """Only tracked files: local, ignored outputs (reports/reproduce/, reports/smoke/) are not published."""
    tracked = subprocess.run(["git", "ls-files", "-z", "reports", "docs", "resume_evidence", "README.md"],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    leaks = []
    for rel in filter(None, tracked.stdout.split("\0")):
        p = REPO_ROOT / rel
        if p.suffix in {".json", ".jsonl", ".md", ".yaml", ".txt"} and p.is_file():
            text = p.read_text(errors="ignore")
            if "/Users/" in text or "/home/runner" in text or (HOME != "/" and HOME in text):
                leaks.append(rel)
    assert not leaks, f"local paths in committed records: {leaks[:5]}"
