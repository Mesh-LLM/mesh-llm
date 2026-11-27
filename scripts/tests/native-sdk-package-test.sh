#!/usr/bin/env bash
# The Kotlin UniFFI alias must carry the same digest as the primary library.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/artifact/lib"
printf 'native library\n' >"$work/artifact/lib/libmesh.so"
cp "$work/artifact/lib/libmesh.so" "$work/artifact/lib/libuniffi.so"
sha="$(shasum -a 256 "$work/artifact/lib/libmesh.so" | cut -d ' ' -f 1)"
jq -n --arg sha "$sha" '{schema_version:1,artifact_id:"test",sdk_version:"0.80.0",target_triple:"x86_64-unknown-linux-gnu",library:"lib/libmesh.so",uniffi_library:"lib/libuniffi.so",library_sha256:$sha,features:["chat"]}' >"$work/artifact/manifest.json"
"$root/mesh/scripts/verify-native-sdk-package.sh" "$work/artifact" >/dev/null
printf 'stale alias\n' >"$work/artifact/lib/libuniffi.so"
if "$root/mesh/scripts/verify-native-sdk-package.sh" "$work/artifact" >"$work/output" 2>&1; then
    echo "accepted a stale UniFFI alias" >&2
    exit 1
fi
grep -q 'library_sha256 mismatch for lib/libuniffi.so' "$work/output"
mkdir -p "$work/producer/mesh/scripts" "$work/producer/scripts" "$work/producer/target/x86_64-unknown-linux-gnu/debug"
cp "$root/mesh/scripts/package-native-sdk.sh" "$work/producer/mesh/scripts/"
cat >"$work/producer/Cargo.toml" <<'EOF'
[workspace.package]
version = "0.80.0"
EOF
printf 'packaged library\n' >"$work/producer/target/x86_64-unknown-linux-gnu/debug/libmeshllm_ffi.so"
"$work/producer/mesh/scripts/package-native-sdk.sh" --backend cpu --target x86_64-unknown-linux-gnu --profile debug --out "$work/produced" >/dev/null
jq -e '.features == ["mesh-inference", "model-management", "local-serving", "chat", "responses"]' \
    "$work/produced/meshllm-native-linux-x86_64-cpu/manifest.json" >/dev/null
"$root/mesh/scripts/verify-native-sdk-package.sh" "$work/produced/meshllm-native-linux-x86_64-cpu" >/dev/null
echo 'native SDK alias: ok'
