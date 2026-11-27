#!/usr/bin/env bash
# Unsupported Vulkan legacy assets must not replace GPU installs with CPU.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
for archive in mesh-llm-x86_64-unknown-linux-gnu.tar.gz \
    mesh-llm-aarch64-unknown-linux-gnu.tar.gz \
    mesh-llm-x86_64-pc-windows-msvc.zip; do
    printf 'host archive %s\n' "$archive" >"$work/$archive"
    (cd "$work" && { sha256sum "$archive" 2>/dev/null || shasum -a 256 "$archive"; }) >"$work/$archive.sha256"
done
"$root/scripts/create-compatibility-aliases.sh" "$work"
[[ -s "$work/mesh-llm-x86_64-unknown-linux-gnu-rocm.tar.gz" ]]
[[ -s "$work/mesh-llm-x86_64-pc-windows-msvc-cuda-13.zip" ]]
[[ ! -e "$work/mesh-llm-x86_64-unknown-linux-gnu-vulkan.tar.gz" ]]
[[ ! -e "$work/mesh-llm-x86_64-pc-windows-msvc-vulkan.zip" ]]
(cd "$work" && { shasum -a 256 -c mesh-llm-x86_64-unknown-linux-gnu-rocm.tar.gz.sha256 2>/dev/null || sha256sum -c mesh-llm-x86_64-unknown-linux-gnu-rocm.tar.gz.sha256; }) >/dev/null
echo 'compatibility aliases: ok'
