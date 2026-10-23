#!/usr/bin/env bash
# Sign a native-runtime catalog with the stock minisign CLI and verify the
# result against the committed trusted public key. Used by release.yml and
# verify-runtime-signing.yml so both exercise the same signing step.
#
# usage: sign-native-runtime-catalog.sh <native-runtimes.json> <out.minisig>
# env:   MESH_NATIVE_RUNTIME_CATALOG_MINISIGN_SECRET_KEY (unencrypted minisign key)
set -euo pipefail

catalog=${1:?catalog path required}
signature=${2:?signature output path required}
public_key="$(dirname "$0")/../skippy/crates/skippy-runtime-install/trust/native-runtime-catalog.pub"

if [[ -z "${MESH_NATIVE_RUNTIME_CATALOG_MINISIGN_SECRET_KEY:-}" ]]; then
  echo "MESH_NATIVE_RUNTIME_CATALOG_MINISIGN_SECRET_KEY is not configured; unsigned publication is forbidden" >&2
  exit 1
fi
command -v minisign >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq minisign; }

key_file=$(mktemp)
trap 'rm -f "$key_file"' EXIT
printf '%s\n' "$MESH_NATIVE_RUNTIME_CATALOG_MINISIGN_SECRET_KEY" > "$key_file"

# Unencrypted key: minisign must not prompt, so stdin is closed.
minisign -S -s "$key_file" -m "$catalog" -x "$signature" \
  -t "MeshLLM native runtime release catalog" < /dev/null
# Fails if the secret does not match the committed public key.
minisign -V -p "$public_key" -m "$catalog" -x "$signature"
