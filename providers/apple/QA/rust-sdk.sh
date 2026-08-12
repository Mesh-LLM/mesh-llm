#!/usr/bin/env bash
set -euo pipefail

APPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$APPLE_ROOT/../.." && pwd)"
PACKAGE_ROOT="$REPO_ROOT/target/apple-runtime/package/meshllm-apple-runtime-darwin-arm64"
OUTPUT_DIR="$REPO_ROOT/target/apple-runtime/rust-sdk"
SUMMARY="$OUTPUT_DIR/summary.json"

[[ -f "$PACKAGE_ROOT/provider-runtime.json" ]] || {
    echo "missing packaged Apple provider; run just apple::package" >&2
    exit 2
}

mkdir -p "$OUTPUT_DIR"
MESH_LLM_PROVIDER_RUNTIME_BUNDLE_DIR=/invalid/embedded-sdk-must-ignore-environment \
MESH_LLM_PROVIDER_RUNTIME_INDEX=/invalid/embedded-sdk-must-ignore-environment.json \
MESH_LLM_PROVIDER_RUNTIME_CACHE_DIR="$OUTPUT_DIR/provider-cache" \
MESH_LLM_APPLE_PROVIDER_ALLOW_AD_HOC=1 \
    just --justfile "$REPO_ROOT/Justfile" with-lld \
        cargo run --quiet -p mesh-llm-sdk --features serving \
        --example apple_system -- "$PACKAGE_ROOT" \
        >"$SUMMARY"

python3 - "$SUMMARY" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["status"] == "pass", summary
assert summary["provider_discovery"] == "rust_sdk_typed_config", summary
assert summary["versioned_model"] == "apple/system@27.0", summary
assert summary["completion_content"] == "rust sdk apple ready", summary
executions = summary["tool_executions"]
assert executions == [{
    "name": "mesh_fixture_lookup",
    "arguments": {"key": "rust-sdk"},
    "result": "mesh-fixture-value-for-rust-sdk",
}], summary
print(json.dumps(summary, indent=2, sort_keys=True))
PY
