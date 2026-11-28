#!/usr/bin/env bash
# CI-only: replace the Skippy registry pin in a disposable checkout. Resolve and
# share the resulting Cargo.lock before running checks; never commit either file.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sha="${1:-}"
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected a full Skippy commit SHA' >&2; exit 2; }
[[ ! -f "$root/.cargo/skippy-local.toml" ]] || { echo 'Remove the local Skippy override first' >&2; exit 1; }
url="$(sed -n 's/^skippy-ffi = .*git = "\([^"]*\)".*/\1/p' "$root/Cargo.toml")"
if [[ -z "$url" ]]; then
    pin="$("$root/scripts/skippy-pin.sh")"
    read -r url _ <<<"$pin"
fi
tmp="$(mktemp "$root/Cargo.toml.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
awk -v url="$url" -v sha="$sha" '
    /^(hf_hub|skippy-[a-z-]+) = \{/ {
        # The exact Git SHA replaces the release constraint, including when
        # an unreleased branch declares an older package version.
        if (index($0, "git =")) {
            if (!sub(/rev = "[^"]*"/, "rev = \"" sha "\"")) exit 1
            gsub(/, version = "[^"]*"/, "")
        } else if (!sub(/version = "[^"]*"/, "git = \"" url "\", rev = \"" sha "\"")) exit 1
        count++
    }
    { print }
    END { if (!count) exit 1 }
' "$root/Cargo.toml" >"$tmp"
mv "$tmp" "$root/Cargo.toml"
echo "Testing Skippy $url@$sha (temporary CI pin)"
