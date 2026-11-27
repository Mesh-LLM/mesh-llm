#!/usr/bin/env bash
# Product bundles must match the runtime OS, architecture and CUDA toolkit.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/bundle/native-runtimes/runtime"
cat >"$work/bundle/mesh-llm" <<'EOF'
#!/bin/sh
echo '{"product_version":"0.80.0","skippy_abi":"0.1.87"}'
EOF
chmod +x "$work/bundle/mesh-llm"
manifest="$work/bundle/native-runtimes/runtime/manifest.json"
cat >"$manifest" <<'EOF'
{"schema_version":2,"runtime":{"id":"test","release_version":"0.80.3","skippy_abi":"0.1.87","backend":{"kind":"cpu"},"platform":{"os":"linux","arch":"x86_64","target":"x86_64-unknown-linux-gnu"}}}
EOF
export MESH_RELEASE_OS=Linux MESH_RELEASE_ARCH=x86_64
# shellcheck source=mesh/scripts/package-release.sh
source "$root/mesh/scripts/package-release.sh"
HOST_TARGET_TRIPLE=x86_64-unknown-linux-gnu
write_product_manifest "$work/bundle" "$work/bundle/mesh-llm" "$work/bundle/native-runtimes/runtime" 0.80.0 cpu
[[ -s "$work/bundle/product-manifest.json" ]]
jq '.runtime.platform.arch = "aarch64"' "$manifest" >"$work/wrong-arch"
cp "$work/wrong-arch" "$manifest"
if (write_product_manifest "$work/bundle" "$work/bundle/mesh-llm" "$work/bundle/native-runtimes/runtime" 0.80.0 cpu) >"$work/output" 2>&1; then
    echo "accepted a runtime for another architecture" >&2
    exit 1
fi
grep -q 'native runtime platform does not match' "$work/output"
jq '.runtime.platform.arch = "x86_64" | .runtime.backend = {"kind":"cuda","cuda":{"toolkit_major":12}}' "$manifest" >"$work/cuda"
cp "$work/cuda" "$manifest"
MESH_CUDA_VERSION=12.8 write_product_manifest "$work/bundle" "$work/bundle/mesh-llm" "$work/bundle/native-runtimes/runtime" 0.80.0 cuda
if (MESH_CUDA_VERSION=13.0 write_product_manifest "$work/bundle" "$work/bundle/mesh-llm" "$work/bundle/native-runtimes/runtime" 0.80.0 cuda) >"$work/output" 2>&1; then
    echo "accepted the wrong CUDA toolkit" >&2
    exit 1
fi
grep -q 'CUDA toolkit does not match requested major' "$work/output"
echo 'product runtime platform: ok'
