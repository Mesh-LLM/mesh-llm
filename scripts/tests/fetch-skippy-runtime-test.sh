#!/usr/bin/env bash
# A missing glibc probe must let the caller choose source fallback (status 3).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/mesh/scripts" "$work/bin"
cp "$root/scripts/fetch-skippy-runtime.sh" "$work/mesh/scripts/"
cat >"$work/mesh/scripts/skippy-pin.sh" <<'EOF'
#!/bin/sh
echo 'https://example.invalid/skippy.git deadbeef 0.80.3'
EOF
cat >"$work/bin/uname" <<'EOF'
#!/bin/sh
case "$1" in -s) echo Linux ;; -m) echo x86_64 ;; esac
EOF
cat >"$work/bin/git" <<'EOF'
#!/bin/sh
printf 'deadbeef\trefs/tags/v0.80.3\n'
EOF
cat >"$work/bin/getconf" <<'EOF'
#!/bin/sh
exit 1
EOF
cat >"$work/bin/curl" <<'EOF'
#!/bin/sh
while [ "$#" -gt 0 ]; do
    case "$1" in -o) output="$2"; shift 2 ;; *) url="$1"; shift ;; esac
done
case "$url" in
    */native-runtimes.json) cp "$TEST_CATALOG" "$output" ;;
    */native-runtimes.json.sha256) cp "$TEST_CATALOG_SHA" "$output" ;;
    *) exit 1 ;;
esac
EOF
chmod +x "$work/mesh/scripts/"* "$work/bin/"*
cat >"$work/catalog.json" <<'EOF'
{"artifacts":[{"platform":{"os":"linux","arch":"x86_64","min_glibc":"2.28"},"backend":{"kind":"cpu"},"url":"https://example.invalid/runtime.tar.gz","sha256":"unused"}]}
EOF
shasum -a 256 "$work/catalog.json" >"$work/catalog.sha256"
export TEST_CATALOG="$work/catalog.json" TEST_CATALOG_SHA="$work/catalog.sha256"
if PATH="$work/bin:$PATH" "$work/mesh/scripts/fetch-skippy-runtime.sh" "$work/out" cpu >"$work/output" 2>&1; then
    echo "accepted a runtime without a glibc probe" >&2
    exit 1
else
    status=$?
fi
[[ "$status" == 3 ]]
grep -q 'package needs glibc 2.28' "$work/output"
echo 'skippy glibc fallback: ok'
