#!/usr/bin/env bash
# package-labkit.sh: build the buyer's zip for the lab kit in labkit/.
#
#   PY=/path/to/python scripts/package-labkit.sh   runs labkit/verify.sh first, then packages
#   SKIP_VERIFY=1 scripts/package-labkit.sh          packages only (use after a verify run you just did)
# PY must have torch, pytest and ruff (only verify.sh uses it).
#
# Output (dist/ is git-ignored):
#   dist/<slug>-<version>.zip      one top-level folder <slug>-<version>/ with the shipped docs,
#                                  check.sh, data/, steps/step-NN/ and solution/, plus SHA256SUMS of its files
#   dist/SHA256SUMS                checksum of the zip
#
# Before zipping it refuses to continue if the staged files contain personal paths, user names,
# e-mail addresses, build junk or anything gitleaks flags. It never uploads or publishes anything.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
kit="$root/labkit"
# shellcheck source=../labkit/kit.conf
. "$kit/kit.conf"
name="$KIT_SLUG-$KIT_VERSION"
dist="$root/dist"
stage="$dist/stage/$name"

if [ "${SKIP_VERIFY:-0}" != 1 ]; then
  "$kit/verify.sh"
fi

rm -rf "$dist/stage" "$dist/$name.zip" "$dist/SHA256SUMS"
mkdir -p "$dist/stage"
python3 "$kit/tools/labgen.py" --src "$kit/src" --out "$stage" --comment "$COMMENT" \
  --steps "$STEPS" --ext "$MARKER_EXT" --format-cmd "$FORMAT_CMD"
for f in $SHIP_FILES; do
  cp -p "$kit/$f" "$stage/$f"
done
# Practice data from the repository (kit.conf DATA_FILES: "source:target" pairs).
for pair in $DATA_FILES; do
  src=${pair%%:*} dst=${pair#*:}
  mkdir -p "$(dirname "$stage/$dst")"
  cp -p "$root/$src" "$stage/$dst"
done

# Fill the licence like Localpane's build-packages.mjs does: licensor and support address from the
# Localpane site config; the same "NOT SET" texts (and a warning) when a value is missing.
site_config=${LOCALPANE_SITE_CONFIG:-$LOCALPANE_SITE_CONFIG_DEFAULT}
python3 - "$site_config" "$stage/$LICENCE_TEMPLATE" "$KIT_SLUG" <<'PY'
import datetime, json, os, re, sys
cfg_path, lic, slug = sys.argv[1:4]
cfg = {}
if os.path.isfile(cfg_path):
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
else:
    print(f"package-labkit: WARNING: {cfg_path} not found", file=sys.stderr)
licensor = (cfg.get("licensor") or "").strip()
email = str(cfg.get("licenceSupportEmail") or "").strip()
if not licensor:
    licensor = 'LICENSOR NOT SET (the seller must set "licensor" in config/site.config.json before this package is sold)'
    print("package-labkit: WARNING: licensor not set; this zip is NOT ready to sell", file=sys.stderr)
if not email:
    email = "[SUPPORT EMAIL NOT SET]"
    print("package-labkit: WARNING: licenceSupportEmail not set; this zip is NOT ready to sell", file=sys.stderr)
vals = {"LICENSOR": licensor, "CONTACT_EMAIL": email, "YEAR": str(datetime.datetime.now(datetime.timezone.utc).year), "PACKAGE_NAME": slug}
with open(lic, encoding="utf-8") as fh:
    text = fh.read()
text = re.sub(r"\{\{(\w+)\}\}", lambda m: vals.get(m.group(1), m.group(0)), text)
left = re.findall(r"\{\{\w+\}\}", text)
if left:
    sys.exit(f"package-labkit: unfilled placeholders in the licence: {left}")
with open(lic, "w", encoding="utf-8") as fh:
    fh.write(text)
PY

# Build junk never ships (labgen skips these already; this is the second line of defence).
find "$stage" \( -name .DS_Store -o -name '*.pyc' -o -name '*.pt' -o -name '*.db' -o -name '*.log' \) -type f -delete
for d in runs dist .git node_modules __pycache__ .pytest_cache .ruff_cache .venv; do
  find "$stage" -name "$d" -type d -prune -exec rm -rf {} +
done

# Hygiene: no personal paths, account names, e-mail addresses or machine details. The filled licence
# is the one file allowed to carry the licensor's name and the support address (as in Localpane's
# paid packages), so it is checked separately: only those two values may appear there.
pattern='/Users/|/home/[a-z]|/private/tmp|scratchpad|alexrosen|AlexRosen23|atrosen|Claude Projects|Desktop/|\.local/|Constellation|OBELISK|Bedrock|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}'
if grep -rIEn --exclude="$LICENCE_TEMPLATE" "$pattern" "$stage"; then
  echo "package-labkit: personal or machine-specific text found above; fix labkit/ and rerun" >&2
  exit 1
fi
if grep -rIl --exclude="$LICENCE_TEMPLATE" 'Rosen' "$stage"; then
  echo "package-labkit: the licensor's name appears outside $LICENCE_TEMPLATE (files above)" >&2
  exit 1
fi
if grep -En '/Users/|/home/[a-z]|/private/tmp|Claude Projects|Desktop/|\.local/|Constellation|AlexRosen23' "$stage/$LICENCE_TEMPLATE"; then
  echo "package-labkit: machine-specific text in the licence" >&2
  exit 1
fi

if command -v gitleaks >/dev/null 2>&1; then
  gitleaks dir --no-banner --redact --exit-code 1 "$stage" >"$dist/gitleaks.log" 2>&1 || {
    cat "$dist/gitleaks.log" >&2
    echo "package-labkit: gitleaks reported findings" >&2
    exit 1
  }
  echo "gitleaks: no leaks found in the staged kit"
else
  echo "package-labkit: gitleaks not installed; secret scan SKIPPED" >&2
fi

sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

# Checksums of every shipped file, inside the zip.
(cd "$stage" && find . -type f ! -name SHA256SUMS | LC_ALL=C sort | sed 's#^\./##' | while read -r f; do sha256 "$f"; done >SHA256SUMS)

# Fixed timestamps and sorted entries, so the same source gives the same zip.
find "$stage" -exec touch -h -t 202611260000 {} +
(cd "$dist/stage" && find "$name" -print | LC_ALL=C sort | zip -X -q -D "$dist/$name.zip" -@)
(cd "$dist" && sha256 "$name.zip" >SHA256SUMS)
rm -rf "$dist/stage"

files=$(unzip -Z1 "$dist/$name.zip" | grep -vc '/$')
size=$(wc -c <"$dist/$name.zip" | tr -d ' ')
echo "built dist/$name.zip: $files files, $size bytes"
cat "$dist/SHA256SUMS"
