#!/usr/bin/env bash
# Build Skippy's CPU runtime from a checkout, or accept a cached one.
#   scripts/ci-skippy-runtime.sh <skippy-checkout> <expected-sha> <out-dir>
# A fresh build writes <out-dir>/receipt.txt: the Skippy commit it was built
# from, the build identity (CI_BUILD_IDENTITY: runner image and recipe), and
# the SHA-256 of every runtime manifest. An existing <out-dir> is used only if
# its receipt matches the commit, the identity and the manifests on disk.
# The receipt catches a stale or mismatched cache entry. It is not proof
# against build code that lies.
set -euo pipefail

(($# == 3)) || { echo "usage: $0 <skippy-checkout> <expected-sha> <out-dir>" >&2; exit 2; }
checkout="$(cd "$1" && pwd -P)"
expected="$2"
mkdir -p "$3"
out="$(cd "$3" && pwd -P)"
identity="${CI_BUILD_IDENTITY:-local}"
receipt="$out/receipt.txt"
sha256() { { sha256sum "$@" 2>/dev/null || shasum -a 256 "$@"; }; }

head="$(git -C "$checkout" rev-parse HEAD)"
[[ "$head" == "$expected" ]] || { echo "Skippy checkout is at $head, expected $expected" >&2; exit 1; }

if [[ -f "$receipt" ]]; then
    [[ "$(sed -n 's/^skippy=//p' "$receipt")" == "$expected" ]] ||
        { echo "cached runtime was built from another Skippy commit" >&2; exit 1; }
    [[ "$(sed -n 's/^identity=//p' "$receipt")" == "$identity" ]] ||
        { echo "cached runtime was built with another build identity" >&2; exit 1; }
    recorded="$(grep -E '^[0-9a-f]{64}  ' "$receipt" || true)"
    [[ -n "$recorded" ]] || { echo "cached runtime receipt lists no manifest" >&2; exit 1; }
    [[ "$recorded" == "$(cd "$out" && sha256 ./*/manifest.json)" ]] ||
        { echo "cached runtime manifests do not match the receipt" >&2; exit 1; }
    echo "using cached runtime built from $expected"
    exit 0
fi
[[ -z "$(ls -A "$out")" ]] || { echo "$out has files but no receipt" >&2; exit 1; }

(cd "$checkout" && scripts/package-native-runtime.sh --build --backend cpu --no-model-package-tool --out "$out")
{
    echo "skippy=$head"
    echo "identity=$identity"
    (cd "$out" && sha256 ./*/manifest.json)
} >"$receipt"
cat "$receipt"
