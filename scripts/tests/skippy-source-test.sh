#!/usr/bin/env bash
# A local checkout at another version must override and restore registry pins.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
read -r _ _ pinned_version <<<"$("$root/scripts/skippy-pin.sh")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
work="$(cd "$work" && pwd -P)"
fixture="$work/mesh"
checkout="$fixture/.deps/skippy"
mkdir -p "$fixture/scripts" "$fixture/.cargo" "$fixture/consumer/src" "$checkout/crates/skippy-ffi/src" "$checkout/crates/skippy-runtime/src"
cp "$root/scripts/skippy-pin.sh" "$root/scripts/skippy-source.sh" "$root/scripts/verify-skippy-source.sh" "$fixture/scripts/"
cat >"$fixture/Cargo.toml" <<EOF
[workspace]
members = ["consumer"]
resolver = "2"
exclude = [".deps"]

[workspace.package]
version = "0.1.0"

[workspace.dependencies]
skippy-ffi = { version = "=$pinned_version", features = ["dynamic-runtime"] }
skippy-runtime = { version = "=$pinned_version" }
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
skippy-runtime.workspace = true
EOF
echo 'pub fn consumer() {}' >"$fixture/consumer/src/lib.rs"
echo 'version = 4' >"$fixture/Cargo.lock"
cat >"$checkout/Cargo.toml" <<'EOF'
[workspace]
members = ["crates/*"]

[workspace.package]
version = "0.80.2"
EOF
cat >"$checkout/crates/skippy-ffi/Cargo.toml" <<'EOF'
[package]
name = "skippy-ffi"
version.workspace = true

[features]
dynamic-runtime = []
EOF
echo 'pub fn ffi() {}' >"$checkout/crates/skippy-ffi/src/lib.rs"
cat >"$checkout/crates/skippy-runtime/Cargo.toml" <<'EOF'
[package]
name = "skippy-runtime"
version.workspace = true
EOF
echo 'pub fn runtime() {}' >"$checkout/crates/skippy-runtime/src/lib.rs"
git init -q "$checkout"
git -C "$checkout" -c user.name=test -c user.email=test@example.invalid add .
git -C "$checkout" -c user.name=test -c user.email=test@example.invalid commit -qm initial

CARGO_NET_OFFLINE=true "$fixture/scripts/skippy-source.sh" "$checkout" >/dev/null
grep -q 'skippy-ffi = { version = "=0.80.2"' "$fixture/Cargo.toml"
grep -q 'skippy-runtime = { version = "=0.80.2"' "$fixture/Cargo.toml"
grep -q '^\[patch.crates-io\]$' "$fixture/.cargo/skippy-local.toml"
grep -Fq "skippy-ffi = { path = \"$checkout/crates/skippy-ffi\" }" "$fixture/.cargo/skippy-local.toml"
(cd "$fixture" && cargo metadata --offline --format-version 1) |
    jq -e 'all(.packages[] | select(.name | startswith("skippy-")); .version == "0.80.2" and .source == null)' >/dev/null
rogue="$work/rogue/crates/skippy-runtime"
mkdir -p "$rogue/src"
cp "$checkout/crates/skippy-runtime/Cargo.toml" "$rogue/Cargo.toml"
sed 's/version.workspace = true/version = "0.80.2"/' "$rogue/Cargo.toml" >"$work/rogue-manifest"
cp "$work/rogue-manifest" "$rogue/Cargo.toml"
cp "$checkout/crates/skippy-runtime/src/lib.rs" "$rogue/src/lib.rs"
sed "s|$checkout/crates/skippy-runtime|$rogue|" "$fixture/.cargo/skippy-local.toml" >"$work/rogue-override"
cp "$fixture/.cargo/skippy-local.toml" "$work/good-override"
cp "$work/rogue-override" "$fixture/.cargo/skippy-local.toml"
if (cd "$fixture" && CARGO_NET_OFFLINE=true scripts/verify-skippy-source.sh "$checkout" >"$work/rogue-output" 2>&1); then
    echo "accepted a Skippy crate outside the selected checkout" >&2
    exit 1
fi
grep -q 'Cargo resolved a Skippy crate outside' "$work/rogue-output"
cp "$work/good-override" "$fixture/.cargo/skippy-local.toml"
"$fixture/scripts/skippy-source.sh" --pinned >/dev/null
grep -Fq "skippy-ffi = { version = \"=$pinned_version\"" "$fixture/Cargo.toml"
grep -Fq "skippy-runtime = { version = \"=$pinned_version\"" "$fixture/Cargo.toml"
[[ ! -e "$fixture/.cargo/skippy-local.toml" ]]
[[ ! -e "$fixture/.cargo/skippy-pinned.lock" ]]
[[ "$(cat "$fixture/Cargo.lock")" == 'version = 4' ]]
"$fixture/scripts/skippy-source.sh" --pinned >/dev/null
[[ "$(cat "$fixture/Cargo.lock")" == 'version = 4' ]]
mkdir -p "$work/windows-stubs"
cat >"$work/windows-stubs/uname" <<'EOF'
#!/bin/sh
echo MINGW64_NT-10.0
EOF
cat >"$work/windows-stubs/cygpath" <<'EOF'
#!/bin/sh
echo C:/mesh/.deps/skippy
EOF
cat >"$work/windows-stubs/cargo" <<'EOF'
#!/bin/sh
cat <<'JSON'
{"packages":[
  {"name":"skippy-ffi","source":null,"manifest_path":"C:\\mesh\\.deps\\skippy\\crates\\skippy-ffi\\Cargo.toml"},
  {"name":"skippy-runtime","source":null,"manifest_path":"C:\\mesh\\.deps\\skippy\\crates\\skippy-runtime\\Cargo.toml"}
]}
JSON
EOF
chmod +x "$work/windows-stubs/"*
PATH="$work/windows-stubs:$PATH" "$fixture/scripts/skippy-source.sh" "$checkout" >/dev/null
grep -Fq 'skippy-ffi = { path = "C:/mesh/.deps/skippy/crates/skippy-ffi" }' "$fixture/.cargo/skippy-local.toml"
"$fixture/scripts/skippy-source.sh" --pinned >/dev/null
echo 'skippy source: ok'
