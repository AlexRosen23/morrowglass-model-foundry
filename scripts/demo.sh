#!/usr/bin/env bash
# Build the web UI, then serve the API (5100) and the built UI (5101). Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."
# Started from a launcher (e.g. the Constellation) the PATH can be minimal: add the user-local uv and node.
export PATH="$HOME/.local/bin:$HOME/.local/opt/node/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
API_PORT="${API_PORT:-5100}"; WEB_PORT="${WEB_PORT:-5101}"
for p in "$API_PORT" "$WEB_PORT"; do
  if lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then echo "port $p is busy; set API_PORT/WEB_PORT"; exit 1; fi
done
[ -d web/node_modules ] || (cd web && npm ci --no-audit --no-fund)
(cd web && npm run build >/dev/null)
uv run --frozen morrowglass ledger
uv run --frozen morrowglass serve --port "$API_PORT" &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM
(cd web && ./node_modules/.bin/vite preview --port "$WEB_PORT" --strictPort --host 127.0.0.1)
