#!/usr/bin/env bash
# Keep the source-build commit and all registry crate versions on one release.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
read -r url rev version <<<"$("$root/scripts/skippy-pin.sh")"
[[ "$url" == "https://github.com/Mesh-LLM/skippy" ]]
[[ "$rev" =~ ^[0-9a-f]{40}$ ]]
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
count="$(awk -v version="$version" '
    /^(hf_hub|skippy-[a-z-]+) = \{/ {
        if (index($0, "version = \"=" version "\"") == 0 || index($0, "git =") != 0) exit 1
        count++
    }
    END { print count }
' "$root/Cargo.toml")"
[[ "$count" == 25 ]]
echo "skippy pin: ok ($count crates at $version)"
