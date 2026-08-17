#!/usr/bin/env bash
set -euo pipefail

APPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$APPLE_ROOT/../.." && pwd)"
PACKAGE_ROOT="$REPO_ROOT/target/apple-runtime/package/meshllm-apple-runtime-darwin-arm64"
OUTPUT_DIR="$REPO_ROOT/target/apple-runtime/carriers"
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/mesh-apple-runtime-carriers.XXXXXX")"

cleanup() {
    rm -rf "$TEMP_ROOT"
}
trap cleanup EXIT

[[ -f "$PACKAGE_ROOT/provider-runtime.json" ]] || {
    echo "missing Apple runtime package; run just apple::package" >&2
    exit 2
}

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

layouts=(
    "cli/runtimes/apple"
    "swift/MeshLLMAppleRuntime.bundle/Contents/Resources/runtime"
    "node/node_modules/@mesh-llm/apple-runtime-darwin-arm64/runtime"
    "jvm/ai/meshllm/apple-runtime/macos-arm64/runtime"
)

for layout in "${layouts[@]}"; do
    destination="$TEMP_ROOT/$layout"
    mkdir -p "$(dirname "$destination")"
    ditto "$PACKAGE_ROOT" "$destination"
    binary="$destination/bin/mesh-apple-runtime"
    codesign --verify --strict --verbose=2 "$binary"
    name="$(echo "$layout" | tr '/@' '__')"
    shasum -a 256 "$binary" | awk '{print $1}' > "$OUTPUT_DIR/$name.sha256"
    "$binary" status > "$OUTPUT_DIR/$name.json"
done

python3 - "$PACKAGE_ROOT/provider-runtime.json" "$OUTPUT_DIR" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
output_dir = pathlib.Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
runtime = manifest["runtime"]
relative = runtime["entrypoint"]
expected = runtime["files"][relative].removeprefix("sha256:")
actual = hashlib.sha256((manifest_path.parent / relative).read_bytes()).hexdigest()
assert actual == expected, (actual, expected)

results = []
for path in sorted(output_dir.glob("*.json")):
    status = json.loads(path.read_text())
    assert status["runtimeID"] == "apple/runtime", status
    system_model = next(model for model in status["models"] if model["modelID"] == "apple/system")
    assert system_model["availability"] == "available", system_model
    results.append(path.stem)

for digest_path in sorted(output_dir.glob("*.sha256")):
    digest = digest_path.read_text().strip()
    assert digest == expected, (digest_path.name, digest, expected)

summary = {"status": "pass", "carriers": results, "binary_sha256": actual}
(output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
