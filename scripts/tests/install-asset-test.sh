#!/usr/bin/env bash
# install.sh asks for the plain host archive on every supported platform,
# including a machine that reports an NVIDIA GPU. Releases no longer carry
# per-GPU archives, so asking for one would be a 404.
set -euo pipefail

install="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/install.sh"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# A machine that looks like it has an NVIDIA GPU and CUDA 12.
printf '#!/bin/sh\necho "NVIDIA-SMI 550.54 CUDA Version: 12.4"\n' > "$work/nvidia-smi"
chmod +x "$work/nvidia-smi"

failures=0
expect() { # expect <uname -s> <uname -m> <want-asset-or-FAIL>
    local got
    # `$0` must not be install.sh, or sourcing it would run the installer.
    # The install directory and release URL are pointed nowhere in case it does.
    got="$(PATH="$work:$PATH" MESH_LLM_TEST_UNAME_S="$1" MESH_LLM_TEST_UNAME_M="$2" \
        MESH_LLM_INSTALL_DIR="$work/bin" MESH_LLM_INSTALL_URL_BASE="file://$work/none" \
        bash -c 'source "$1" && asset_name' asset-test "$install" 2>/dev/null)" || got=FAIL
    if [[ "$got" != "$3" ]]; then
        echo "FAIL: $1/$2 asked for '$got', want '$3'" >&2
        failures=$((failures + 1))
    fi
}

expect Darwin arm64 mesh-llm-aarch64-apple-darwin.tar.gz
expect Linux x86_64 mesh-llm-x86_64-unknown-linux-gnu.tar.gz
expect Linux amd64 mesh-llm-x86_64-unknown-linux-gnu.tar.gz
expect Linux aarch64 mesh-llm-aarch64-unknown-linux-gnu.tar.gz
expect Linux armv7l FAIL
expect Darwin x86_64 FAIL

if grep -Eq 'gnu-(cuda|rocm|vulkan)' "$install"; then
    echo "FAIL: install.sh still names a GPU archive" >&2
    failures=$((failures + 1))
fi

((failures == 0)) || exit 1
echo "install-asset-test: ok"
