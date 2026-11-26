#!/usr/bin/env bash
# verify.sh: prove the lab kit is sound. It generates every starter and the solution from src/ with
# tools/labgen.py, then checks, for each step N:
#   solution     the step's tests exist (at least one) and PASS
#   step-N       the tree imports (pytest can collect every test), the step's own tests FAIL, and every
#                earlier step's tests plus the test_kit_* tests still PASS (so a stub never breaks work
#                the learner already did)
# "Fail on the starter" is judged for the step's tests as a set; the line also shows how many of them
# fail individually (a test may pass on a stub when the stub happens to be good enough for it).
# Exit status 0 only if the whole matrix is as expected. Logs: $OUT/logs/.
#
# Usage: PY=/path/to/python labkit/verify.sh     (that Python needs torch, pytest and ruff)
#        OUT=/some/dir labkit/verify.sh          (OUT defaults to a fresh temporary directory)
# Runs every test command under `nice -n 19`, one at a time. Every training run in the tests is a tiny
# CPU model that finishes in about a second.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=kit.conf
. "$here/kit.conf"
export PY
OUT=${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/labkit-verify.XXXXXX")}
build="$OUT/build"
logs="$OUT/logs"
mkdir -p "$logs"

python3 "$here/tools/labgen.py" --src "$here/src" --out "$build" --comment "$COMMENT" \
  --steps "$STEPS" --ext "$MARKER_EXT" --format-cmd "$FORMAT_CMD"

sel_step() { printf 'test_step%02d_' "$1"; }
sel_before() { # test_kit_ plus every step before $1, as a pytest -k expression
  local s="test_kit_" i
  for ((i = 1; i < $1; i++)); do s+=$(printf ' or test_step%02d_' "$i"); done
  printf '%s' "$s"
}
# run <dir> <log> <command with $SEL> <sel>: exit status of the command
run() {
  local dir=$1 log=$2 cmd=$3
  SEL=$4
  export SEL
  (cd "$dir" && nice -n 19 sh -c "$cmd") >"$log" 2>&1
}

fail=0
rows=()
note() { rows+=("$1"); printf '%s\n' "$1"; }

sol="$build/solution"
if run "$sol" "$logs/solution-lint.log" "$LINT_CMD" ""; then note "solution  lint (ruff)                 ok"; else note "solution  lint                       FAILED (see $logs/solution-lint.log)"; fail=1; fi
if run "$sol" "$logs/solution-kit.log" "$TEST_CMD" 'test_kit_'; then note "solution  test_kit_*                  PASS"; else note "solution  test_kit_*                  FAIL (unexpected)"; fail=1; fi
if run "$sol" "$logs/solution-all.log" "$TEST_CMD" 'test_'; then note "solution  all tests                   PASS ($(grep -Eo '[0-9]+ passed' "$logs/solution-all.log"))"; else note "solution  all tests                   FAIL (unexpected)"; fail=1; fi

for ((n = 1; n <= STEPS; n++)); do
  nn=$(printf '%02d' "$n")
  starter="$build/steps/step-$nn"
  SEL=$(sel_step "$n"); export SEL
  count=$(cd "$sol" && sh -c "$COUNT_CMD" 2>/dev/null || true)
  if [ "${count:-0}" -lt 1 ]; then note "step $nn   no test_step${nn}_* tests found"; fail=1; continue; fi

  t0=$(date +%s)
  if run "$sol" "$logs/step-$nn-solution.log" "$TEST_CMD" "$(sel_step "$n")"; then sol_res=PASS; else sol_res=FAIL; fail=1; fi
  if ! run "$starter" "$logs/step-$nn-compile.log" "$COMPILE_CMD" ""; then
    note "step $nn   $count tests | starter does NOT IMPORT (see $logs/step-$nn-compile.log)"; fail=1; continue
  fi
  if run "$starter" "$logs/step-$nn-starter.log" "$TEST_CMD" "$(sel_step "$n")"; then st_res=PASS; fail=1; else st_res=FAIL; fi
  failing=$(grep -c "^FAILED .*::test_step${nn}_" "$logs/step-$nn-starter.log" || true)
  st_res="$st_res ($failing/$count failing)"
  if run "$starter" "$logs/step-$nn-earlier.log" "$TEST_CMD" "$(sel_before "$n")"; then prev=PASS; else prev=FAIL; fail=1; fi
  t1=$(date +%s)
  mark=ok; { [ "$sol_res" = PASS ] && [ "${st_res%% *}" = FAIL ] && [ "$prev" = PASS ]; } || mark=UNEXPECTED
  note "step $nn   $(printf '%2d' "$count") tests | starter: $st_res (want FAIL) | solution: $sol_res (want PASS) | earlier steps + kit on starter: $prev (want PASS) | $((t1 - t0))s | $mark"
done

{
  echo "labkit verify: $KIT_TITLE $KIT_VERSION"
  echo "python: $("$PY" -c 'import platform, sys, torch; print(platform.python_version(), "| torch", torch.__version__, "|", sys.platform, platform.machine())' 2>/dev/null)"
  printf '%s\n' "${rows[@]}"
  [ $fail -eq 0 ] && echo "RESULT: OK" || echo "RESULT: FAILED"
} >"$OUT/verify-summary.txt"
echo "summary: $OUT/verify-summary.txt"
if [ $fail -eq 0 ]; then echo "RESULT: OK"; else echo "RESULT: FAILED"; exit 1; fi
