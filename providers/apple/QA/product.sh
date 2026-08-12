#!/usr/bin/env bash
set -euo pipefail

APPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$APPLE_ROOT/../.." && pwd)"
PRODUCT_VERSION="${MESH_APPLE_RUNTIME_PRODUCT_VERSION:?set the composed MeshLLM product version}"
PRODUCT_OUTPUT="${MESH_APPLE_RUNTIME_PRODUCT_OUTPUT:-$REPO_ROOT/target/apple-runtime/product}"
ARCHIVE="$PRODUCT_OUTPUT/mesh-llm-${PRODUCT_VERSION#v}-aarch64-apple-darwin.tar.gz"
QA_OUTPUT="$REPO_ROOT/target/apple-runtime/product-qa"
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mesh-apple-product.XXXXXX")"

cleanup() {
    rm -rf "$TEMP_ROOT"
}
trap cleanup EXIT

[[ -f "$ARCHIVE" ]] || {
    echo "missing composed Apple product: $ARCHIVE" >&2
    echo "run just apple::product $PRODUCT_VERSION" >&2
    exit 2
}

tar -xzf "$ARCHIVE" -C "$TEMP_ROOT"
BUNDLE="$TEMP_ROOT/mesh-bundle"
HOST_BINARY="$BUNDLE/mesh-llm"
PROVIDER_BUNDLE="$BUNDLE/provider-runtimes/apple/meshllm-apple-runtime-darwin-arm64"

[[ -x "$HOST_BINARY" ]] || {
    echo "composed product has no executable MeshLLM host" >&2
    exit 2
}
[[ -f "$PROVIDER_BUNDLE/provider-runtime.json" ]] || {
    echo "composed product has no adjacent Apple provider runtime" >&2
    exit 2
}

python3 - "$BUNDLE/product-manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
providers = manifest.get("provider_runtimes", [])
assert len(providers) == 1, manifest
provider = providers[0]
assert provider["id"] == "meshllm-apple-runtime-darwin-arm64", provider
assert provider["provider_kind"] == "apple", provider
assert provider["path"] == (
    "provider-runtimes/apple/meshllm-apple-runtime-darwin-arm64"
), provider
PY

MESH_APPLE_RUNTIME_MESH_HOST_BINARY="$HOST_BINARY" \
MESH_APPLE_RUNTIME_MESH_OUTPUT_DIR="$QA_OUTPUT" \
MESH_APPLE_RUNTIME_MESH_AUTO_DISCOVERY=1 \
    "$APPLE_ROOT/QA/mesh.sh"

python3 - "$QA_OUTPUT/summary.json" "$ARCHIVE" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["product_archive"] = sys.argv[2]
summary["provider_discovery"] = "adjacent_product_bundle"
summary["provider_bundle_override_used"] = False
summary["provider_index_override_used"] = False
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "status": summary["status"],
    "model": summary["model"],
    "versioned_model": summary["versioned_model"],
    "provider_discovery": summary["provider_discovery"],
    "provider_bundle_override_used": summary["provider_bundle_override_used"],
    "completion_content": summary["completion"]["choices"][0]["message"]["content"],
    "tool_executions": summary["tool"]["mesh_tool_executions"],
}, indent=2, sort_keys=True))
PY
