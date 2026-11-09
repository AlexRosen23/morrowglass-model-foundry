#!/usr/bin/env bash
# Reproduce every standard-profile experiment in reports/ (about 45 min on an M5 Pro, MPS).
# Idempotent: arms whose summary.json already exists are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
MG="uv run --frozen morrowglass"
$MG sanity
$MG baselines --dataset shakespeare
for exp in scratch tokenization capacity optimization digitsum break-lr break-mask break-overfit; do
  echo "=== $exp $(date '+%H:%M:%S')"
  $MG run "$exp" --profile standard
done
echo "=== done $(date '+%H:%M:%S')"
