#!/usr/bin/env python3
"""labgen: build a Localpane lab kit's step starters and clean solution from one marked source tree.

The source tree (labkit/src) is the finished, working project. Code a learner writes in step N sits
inside a marked region:

    <C>lab:begin N  short title
    ...finished code (kept in the solution and in every starter after step N)...
    <C>lab:stub
    <C>> stub code, written commented-out (it replaces the region in starters for steps <= N)
    <C>lab:end

<C> is the language's line-comment prefix ("//" for Go, "#" for Python or shell). A region with no
lab:stub section is simply removed from starters. Regions cannot nest.

Outputs, for a kit with S steps:
    OUT/solution/          every region solved, every marker and stub line removed
    OUT/steps/step-NN/     regions of steps < NN solved, regions of steps >= NN stubbed

Standard library only; Python 3.8+. Exit status 1 on any marker error (nothing half-written is kept).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

SKIP_DIRS = {".git", "bin", "var", "dist", "runs", "node_modules", "__pycache__", ".venv", ".pytest_cache", ".ruff_cache"}


class MarkerError(Exception):
    pass


def parse(path, text, comment, steps):
    """Return a list of chunks: ("text", lines) or ("region", step, solved_lines, stub_lines)."""
    c = re.escape(comment)
    re_begin = re.compile(r"^\s*" + c + r"lab:begin\s+(\d+)\b")
    re_stub = re.compile(r"^\s*" + c + r"lab:stub\s*$")
    re_end = re.compile(r"^\s*" + c + r"lab:end\s*$")
    re_stub_line = re.compile(r"^(\s*)" + c + r"> ?(.*)$")
    re_any = re.compile(c + r"lab:(begin|stub|end)")

    chunks, plain = [], []
    state, step, solved, stub, begin_line = "text", 0, [], [], 0
    for no, line in enumerate(text.splitlines(keepends=True), 1):
        where = f"{path}:{no}"
        if state == "text":
            m = re_begin.match(line)
            if m:
                step = int(m.group(1))
                if not 1 <= step <= steps:
                    raise MarkerError(f"{where}: step {step} is outside 1..{steps}")
                chunks.append(("text", plain))
                plain, solved, stub, state, begin_line = [], [], [], "solved", no
            elif re_any.search(line):
                raise MarkerError(f"{where}: lab:stub or lab:end outside a region")
            else:
                plain.append(line)
        elif state == "solved":
            if re_begin.match(line):
                raise MarkerError(f"{where}: regions cannot nest (open region began at line {begin_line})")
            if re_stub.match(line):
                state = "stub"
            elif re_end.match(line):
                chunks.append(("region", step, solved, stub))
                state = "text"
            else:
                solved.append(line)
        else:  # stub
            if re_end.match(line):
                chunks.append(("region", step, solved, stub))
                state = "text"
            elif re_any.search(line):
                raise MarkerError(f"{where}: unexpected marker inside a stub section")
            elif line.strip() == "":
                stub.append(line)
            else:
                m = re_stub_line.match(line)
                if not m:
                    raise MarkerError(f"{where}: stub lines must start with '{comment}>'")
                nl = "\n" if line.endswith("\n") else ""
                stub.append(m.group(1) + m.group(2).rstrip("\n") + nl)
    if state != "text":
        raise MarkerError(f"{path}:{begin_line}: region is never closed with lab:end")
    chunks.append(("text", plain))
    return chunks


def render(chunks, target):
    """target: step number for a starter, or None for the solution."""
    out = []
    for ch in chunks:
        if ch[0] == "text":
            out.extend(ch[1])
        else:
            _, step, solved, stub = ch
            out.extend(solved if target is None or step < target else stub)
    return "".join(out)


def walk(src):
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for f in sorted(files):
            if f == ".DS_Store":
                continue
            full = os.path.join(root, f)
            yield full, os.path.relpath(full, src)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="marked source tree (the finished project)")
    ap.add_argument("--out", required=True, help="output directory (replaced)")
    ap.add_argument("--comment", required=True, help='line-comment prefix, e.g. "//" or "#"')
    ap.add_argument("--steps", type=int, required=True, help="number of steps")
    ap.add_argument("--ext", default="", help="comma-separated extensions to scan for markers (default: all text files)")
    ap.add_argument("--format-cmd", default="", help="run in each generated tree afterwards, e.g. 'gofmt -w .'")
    a = ap.parse_args()

    exts = tuple(e if e.startswith(".") else "." + e for e in a.ext.split(",") if e)
    files, counts = [], {n: 0 for n in range(1, a.steps + 1)}
    try:
        for full, rel in walk(a.src):
            with open(full, "rb") as fh:
                raw = fh.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                files.append((rel, full, None, os.stat(full).st_mode))
                continue
            if exts and not rel.endswith(exts):
                chunks = [("text", text.splitlines(keepends=True))]
            else:
                chunks = parse(rel, text, a.comment, a.steps)
            for ch in chunks:
                if ch[0] == "region":
                    counts[ch[1]] += 1
            files.append((rel, full, chunks, os.stat(full).st_mode))
    except MarkerError as e:
        print(f"labgen: {e}", file=sys.stderr)
        return 1
    empty = [n for n, k in counts.items() if k == 0]
    if empty:
        print(f"labgen: steps with no marked region: {empty}", file=sys.stderr)
        return 1

    tmp = a.out.rstrip("/") + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    targets = [("solution", None)] + [(f"steps/step-{n:02d}", n) for n in range(1, a.steps + 1)]
    for name, target in targets:
        base = os.path.join(tmp, name)
        for rel, full, chunks, mode in files:
            dst = os.path.join(base, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if chunks is None:
                shutil.copyfile(full, dst)
            else:
                body = render(chunks, target)
                if re.search(re.escape(a.comment) + r"lab:(begin|stub|end)", body):
                    print(f"labgen: marker left in {name}/{rel}", file=sys.stderr)
                    return 1
                with open(dst, "w", encoding="utf-8") as fh:
                    fh.write(body)
            os.chmod(dst, mode & 0o777)
        if a.format_cmd:
            r = subprocess.run(a.format_cmd, shell=True, cwd=base)
            if r.returncode != 0:
                print(f"labgen: format command failed in {name}", file=sys.stderr)
                return 1
    shutil.rmtree(a.out, ignore_errors=True)
    os.rename(tmp, a.out)
    summary = ", ".join(f"step {n}: {k}" for n, k in counts.items())
    print(f"labgen: {len(files)} files, regions per step: {summary}; wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
