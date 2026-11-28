#!/usr/bin/env bash
# Keep the source-build commit and all registry crate versions on one release.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/scripts"
cp "$root/scripts/skippy-pin.sh" "$work/scripts/"
cp "$root/Cargo.toml" "$work/Cargo.toml"
if grep -q '^skippy-ffi = .*git =' "$work/Cargo.toml"; then
    # CI overrides change the working manifest, not the committed release pin.
    git -C "$root" show HEAD:Cargo.toml >"$work/Cargo.toml"
fi
read -r url rev version <<<"$("$work/scripts/skippy-pin.sh")"
[[ "$url" == "https://github.com/Mesh-LLM/skippy" ]]
[[ "$rev" =~ ^[0-9a-f]{40}$ ]]
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
count="$(awk -v version="$version" '
    /^(hf_hub|skippy-[a-z-]+) = \{/ {
        if (index($0, "version = \"=" version "\"") == 0 || index($0, "git =") != 0) exit 1
        count++
    }
    END { print count }
' "$work/Cargo.toml")"
[[ "$count" == 25 ]]
echo "skippy pin: ok ($count crates at $version)"
