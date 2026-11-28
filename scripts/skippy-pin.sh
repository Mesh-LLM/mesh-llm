#!/usr/bin/env bash
# Print the Skippy repository, release commit and exact crates.io version.
# The commit names the source used for native-runtime fallback builds; Cargo
# gets the Rust crates from the matching release recorded in Cargo.lock.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pin_url="https://github.com/Mesh-LLM/skippy"
pin_rev="29c1109bf62e992f480889948da219c8adaae864"
pin_version="0.80.4"
if [[ ! -f "$REPO_ROOT/.cargo/skippy-local.toml" ]]; then
    manifest_version="$(sed -n 's/^skippy-ffi = .*version = "=\([0-9][0-9.]*\)".*/\1/p' "$REPO_ROOT/Cargo.toml")"
    [[ "$manifest_version" == "$pin_version" ]] || {
        echo "skippy-ffi in Cargo.toml must pin Skippy $pin_version" >&2
        exit 1
    }
fi
printf '%s %s %s\n' "$pin_url" "$pin_rev" "$pin_version"
