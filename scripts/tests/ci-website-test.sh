#!/usr/bin/env bash
# A successful npm build without the crate index must fail the website gate.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/scripts" "$work/mesh/website" "$work/bin"
cp "$root/scripts/ci-website.sh" "$work/scripts/"
cat >"$work/bin/npm" <<'EOF'
#!/bin/sh
if [ "$1 $2" = 'run build' ]; then
    mkdir -p ../../docs
    echo page >../../docs/index.html
fi
EOF
chmod +x "$work/bin/npm"
if PATH="$work/bin:$PATH" "$work/scripts/ci-website.sh" >"$work/output" 2>&1; then
    echo "accepted a website without its crate index" >&2
    exit 1
fi
grep -q 'did not produce the home page and crate index' "$work/output"
mkdir -p "$work/docs/crates"
echo roster >"$work/docs/crates/index.html"
PATH="$work/bin:$PATH" "$work/scripts/ci-website.sh"
echo 'website gate: ok'
