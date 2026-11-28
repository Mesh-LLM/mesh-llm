#!/usr/bin/env bash
# Tests for scripts/ci-skippy-runtime.sh against a stand-in Skippy checkout
# whose packaging script writes one manifest and counts its runs.
set -euo pipefail

runtime="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ci-skippy-runtime.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
repo="$work/skippy"
mkdir -p "$repo/scripts"
cat >"$repo/scripts/package-native-runtime.sh" <<'EOF'
#!/usr/bin/env bash
out="${*: -1}"
mkdir -p "$out/fixture-runtime"
echo '{"runtime":{"id":"fixture-runtime"}}' >"$out/fixture-runtime/manifest.json"
echo built >>"$out/../builds"
EOF
chmod +x "$repo/scripts/package-native-runtime.sh"
git init -q "$repo"
git -C "$repo" add -A
git -C "$repo" -c user.name=test -c user.email=test@example.invalid commit -q -m fixture
sha="$(git -C "$repo" rev-parse HEAD)"
out="$work/out"
export CI_BUILD_IDENTITY=one

failures=0
check() { # check <ok|fail> <description> <command...>
    local want="$1" what="$2" got=ok
    shift 2
    "$@" >/dev/null 2>&1 || got=fail
    if [[ "$got" != "$want" ]]; then
        echo "FAIL: $what: got $got, want $want" >&2
        failures=$((failures + 1))
    fi
}
check ok "cold build" "$runtime" "$repo" "$sha" "$out"
check ok "warm reuse" "$runtime" "$repo" "$sha" "$out"
[[ "$(wc -l <"$work/builds")" -eq 1 ]] || { echo "FAIL: warm reuse rebuilt" >&2; failures=$((failures + 1)); }
receipt="$(cat "$out/receipt.txt")"
check fail "wrong head" "$runtime" "$repo" 0000000000000000000000000000000000000000 "$out"
check fail "wrong identity" env CI_BUILD_IDENTITY=two "$runtime" "$repo" "$sha" "$out"
mkdir "$out/extra" && cp "$out/fixture-runtime/manifest.json" "$out/extra/"
check fail "extra manifest" "$runtime" "$repo" "$sha" "$out"
rm -r "$out/extra"
echo edited >>"$out/fixture-runtime/manifest.json"
check fail "edited manifest" "$runtime" "$repo" "$sha" "$out"
[[ "$(cat "$out/receipt.txt")" == "$receipt" ]] || { echo "FAIL: a rejection rewrote the receipt" >&2; failures=$((failures + 1)); }
rm "$out/receipt.txt"
check fail "files but no receipt" "$runtime" "$repo" "$sha" "$out"
((failures == 0)) || exit 1
echo "ci-skippy-runtime: ok"
