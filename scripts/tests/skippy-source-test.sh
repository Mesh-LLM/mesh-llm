#!/usr/bin/env bash
# A local checkout at another version must override and restore registry pins.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
work="$(cd "$work" && pwd -P)"
fixture="$work/mesh"
checkout="$work/skippy"
mkdir -p "$fixture/scripts" "$fixture/.cargo" "$fixture/consumer/src" "$checkout/crates/skippy-ffi/src"
cp "$root/scripts/skippy-pin.sh" "$root/scripts/skippy-source.sh" "$fixture/scripts/"
cat >"$fixture/Cargo.toml" <<'EOF'
[workspace]
members = ["consumer"]
resolver = "2"

[workspace.dependencies]
skippy-ffi = { version = "=0.80.3", features = ["dynamic-runtime"] }
skippy-runtime = { version = "=0.80.3" }
EOF
cat >"$fixture/.cargo/config.toml" <<'EOF'
include = [{ path = "skippy-local.toml", optional = true }]
EOF
cat >"$fixture/consumer/Cargo.toml" <<'EOF'
[package]
name = "consumer"
version = "0.1.0"
edition = "2024"

[dependencies]
skippy-ffi.workspace = true
EOF
echo 'pub fn consumer() {}' >"$fixture/consumer/src/lib.rs"
echo 'version = 4' >"$fixture/Cargo.lock"
cat >"$checkout/Cargo.toml" <<'EOF'
[workspace.package]
version = "0.80.2"
EOF
cat >"$checkout/crates/skippy-ffi/Cargo.toml" <<'EOF'
[package]
name = "skippy-ffi"
version = "0.80.2"

[features]
dynamic-runtime = []
EOF
echo 'pub fn ffi() {}' >"$checkout/crates/skippy-ffi/src/lib.rs"
git init -q "$checkout"
git -C "$checkout" -c user.name=test -c user.email=test@example.invalid add .
git -C "$checkout" -c user.name=test -c user.email=test@example.invalid commit -qm initial

"$fixture/scripts/skippy-source.sh" "$checkout" >/dev/null
grep -q 'skippy-ffi = { version = "=0.80.2"' "$fixture/Cargo.toml"
grep -q 'skippy-runtime = { version = "=0.80.2"' "$fixture/Cargo.toml"
grep -q '^\[patch.crates-io\]$' "$fixture/.cargo/skippy-local.toml"
grep -Fq "skippy-ffi = { path = \"$checkout/crates/skippy-ffi\" }" "$fixture/.cargo/skippy-local.toml"
(cd "$fixture" && cargo metadata --offline --format-version 1) |
    jq -e 'any(.packages[]; .name == "skippy-ffi" and .version == "0.80.2" and .source == null)' >/dev/null
"$fixture/scripts/skippy-source.sh" --pinned >/dev/null
grep -q 'skippy-ffi = { version = "=0.80.3"' "$fixture/Cargo.toml"
grep -q 'skippy-runtime = { version = "=0.80.3"' "$fixture/Cargo.toml"
[[ ! -e "$fixture/.cargo/skippy-local.toml" ]]
[[ ! -e "$fixture/.cargo/skippy-pinned.lock" ]]
[[ "$(cat "$fixture/Cargo.lock")" == 'version = 4' ]]
"$fixture/scripts/skippy-source.sh" --pinned >/dev/null
[[ "$(cat "$fixture/Cargo.lock")" == 'version = 4' ]]
echo 'skippy source: ok'
