#!/usr/bin/env bash
# Endpoint settings and dotenv files must invalidate a cached UI build.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/ui/src" "$work/bin"
echo '{}' >"$work/ui/package.json"
echo 'lock' >"$work/ui/pnpm-lock.yaml"
echo 'source' >"$work/ui/src/main.ts"
cat >"$work/bin/pnpm" <<'EOF'
#!/bin/sh
case "$1 $2" in
    'install --frozen-lockfile') mkdir -p node_modules ;;
    'run build') mkdir -p dist; echo page >dist/index.html; echo build >>"$UI_TEST_LOG" ;;
    *) exit 1 ;;
esac
EOF
chmod +x "$work/bin/pnpm"
export PATH="$work/bin:$PATH" UI_TEST_LOG="$work/builds"
VITE_API_URL=https://one.example "$root/mesh/scripts/build-ui.sh" "$work/ui" >/dev/null
VITE_API_URL=https://one.example "$root/mesh/scripts/build-ui.sh" "$work/ui" >/dev/null
[[ "$(wc -l <"$UI_TEST_LOG" | tr -d ' ')" == 1 ]]
VITE_API_URL=https://two.example "$root/mesh/scripts/build-ui.sh" "$work/ui" >/dev/null
[[ "$(wc -l <"$UI_TEST_LOG" | tr -d ' ')" == 2 ]]
echo 'VITE_MANAGEMENT_API_URL=https://management.example' >"$work/ui/.env.production"
touch -t 203001010000 "$work/ui/.env.production"
VITE_API_URL=https://two.example "$root/mesh/scripts/build-ui.sh" "$work/ui" >/dev/null
[[ "$(wc -l <"$UI_TEST_LOG" | tr -d ' ')" == 3 ]]
echo 'UI build staleness: ok'
