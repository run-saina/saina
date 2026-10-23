#!/usr/bin/env bash
# Official-SDK acceptance against a local self-hosted server with MOCKED inference (wire compatibility only).
# Pinned clients: openai 3.37.0 (PyPI), openai 7.43.0 (npm), typesafe-sdk[zstd] 0.7.5 (PyPI), @typesafe-ai/sdk 0.7.0 (npm).
# Usage: scripts/acceptance/run.sh [WORKDIR]   (needs uv, Node.js 22.15+ for zstd, and network access to install)
# To check a hosted or other server instead, run the four client scripts directly with its URL and key.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../.." && pwd)
work=${1:-$(mktemp -d)}; port=${PORT:-8731}
uv venv -q -p 3.12 "$work/venv"
VIRTUAL_ENV="$work/venv" uv pip install -q -e "$repo[server,media]" "openai==3.37.0" "typesafe-sdk[zstd]==0.7.5"
mkdir -p "$work/js" && cp "$here"/*.mjs "$work/js/"
(cd "$work/js" && npm init -y >/dev/null && npm install --silent --save-exact openai@7.43.0 @typesafe-ai/sdk@0.7.0)
"$work/venv/bin/python" "$here/server.py" "$port" & server=$!
trap 'kill $server' EXIT
for _ in $(seq 50); do curl -sf "http://127.0.0.1:$port/livez" >/dev/null && break; sleep .2; done
status=0
"$work/venv/bin/python" "$here/openai_python.py" "http://127.0.0.1:$port/v1" k --omni saina-helm-2-omni-0.8b || status=1
(cd "$work/js" && node openai_js.mjs "http://127.0.0.1:$port/v1" k) || status=1
"$work/venv/bin/python" "$here/typesafe_python.py" "http://127.0.0.1:$port" k || status=1
(cd "$work/js" && node typesafe_js.mjs "http://127.0.0.1:$port" k) || status=1
exit $status
