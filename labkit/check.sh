#!/bin/sh
# check.sh: "check your work" for one step of the lab kit.
#
#   ./check.sh 3                 runs step 3's tests in steps/step-03
#   ./check.sh 3 ~/my-model      runs them in your own copy instead
#   ./check.sh 8 solution        runs them on the finished project
#
# It runs test_step03_* first, then every earlier step's tests and the provided test_kit_* tests, so
# you also see if a change broke something you finished before. Uses $PYTHON if set, else python3;
# that Python needs torch and pytest (see README-FIRST.md).
set -e
if [ $# -lt 1 ]; then
  echo "usage: ./check.sh <step 1-8> [directory]" >&2
  exit 2
fi
case "$1" in
  [1-8]) step=$1 ;;
  *) echo "step must be 1 to 8" >&2; exit 2 ;;
esac
nn=$(printf '%02d' "$step")
here=$(cd "$(dirname "$0")" && pwd)
dir=${2:-"$here/steps/step-$nn"}
py=${PYTHON:-python3}
cd "$dir"
if ! "$py" -c "import torch, pytest" 2>/dev/null; then
  echo "$py cannot import torch and pytest. Activate your virtual environment first (README-FIRST.md)." >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1

echo "== step $step: test_step${nn}_* in $dir"
if "$py" -m pytest -q -k "test_step${nn}_"; then this=PASS; else this=FAIL; fi

earlier="test_kit_"
i=1
while [ "$i" -lt "$step" ]; do earlier="$earlier or test_step$(printf '%02d' "$i")_"; i=$((i + 1)); done
echo "== earlier steps and provided code: $earlier"
if "$py" -m pytest -q -k "$earlier" >/dev/null 2>&1; then before=PASS; else before=FAIL; fi

echo
echo "step $step tests:            $this"
echo "earlier steps + kit tests: $before"
if [ "$this" = PASS ] && [ "$before" = PASS ]; then
  if [ "$step" -lt 8 ]; then
    echo "Step $step is done. Next: WORKBOOK.md, step $((step + 1))."
  else
    echo "All 8 steps are done. See \"Done. Going further\" at the end of WORKBOOK.md."
  fi
else
  [ "$before" = FAIL ] && echo "Something you finished earlier broke: run $py -m pytest -q -k '$earlier' to see it."
  exit 1
fi
