#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/package-sdk-provider-runtime.sh [options]

Copy one already-built Apple provider runtime into host-capable macOS SDK
resource layouts. The exact signed executable is reused by every carrier.

Options:
  --sdk node|swift|kotlin|all  SDK package to update. Defaults to all.
  --runtime-dir DIR            Bundle root containing provider-runtime.json.
                               Defaults to the Golden Gate QA package output.
  -h, --help                   Show this help.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SDK="all"
RUNTIME_DIR="$REPO_ROOT/target/apple-runtime/package/meshllm-apple-runtime-darwin-arm64"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --sdk)
            SDK="${2:?missing SDK name}"
            shift 2
            ;;
        --runtime-dir)
            RUNTIME_DIR="${2:?missing runtime directory}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

case "$SDK" in
    node|swift|kotlin|all) ;;
    *)
        echo "unsupported SDK: $SDK" >&2
        usage
        exit 1
        ;;
esac

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || {
    echo "Apple provider SDK packaging requires Apple silicon macOS" >&2
    exit 2
}
[[ -f "$RUNTIME_DIR/provider-runtime.json" ]] || {
    echo "provider runtime is missing provider-runtime.json: $RUNTIME_DIR" >&2
    exit 2
}
[[ -x "$RUNTIME_DIR/bin/mesh-apple-runtime" ]] || {
    echo "provider runtime entrypoint is missing or not executable: $RUNTIME_DIR/bin/mesh-apple-runtime" >&2
    exit 2
}

codesign --verify --strict --verbose=2 "$RUNTIME_DIR/bin/mesh-apple-runtime"

write_resource_manifest() {
    local destination="$1"
    python3 - "$destination" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
entries = [
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path.name not in {".gitkeep", "manifest.txt"}
]
(root / "manifest.txt").write_text("\n".join(sorted(entries)) + "\n", encoding="utf-8")
PY
}

copy_runtime() {
    local destination="$1"
    local include_resource_manifest="$2"
    rm -rf "$destination"
    mkdir -p "$destination"
    ditto "$RUNTIME_DIR" "$destination"
    : > "$destination/.gitkeep"
    if [[ "$include_resource_manifest" == "1" ]]; then
        write_resource_manifest "$destination"
    fi
    cmp "$RUNTIME_DIR/bin/mesh-apple-runtime" "$destination/bin/mesh-apple-runtime"
    codesign --verify --strict --verbose=2 "$destination/bin/mesh-apple-runtime"
}

package_node() {
    copy_runtime "$REPO_ROOT/sdk/node/apple-runtime-darwin-arm64/runtime" 0
}

package_swift() {
    copy_runtime "$REPO_ROOT/sdk/swift/Sources/MeshLLMAppleProviderResources/Resources/apple" 0
}

package_kotlin() {
    copy_runtime "$REPO_ROOT/sdk/kotlin/apple-runtime-macos-arm64/src/main/resources/mesh-llm/provider-runtimes/apple" 1
}

case "$SDK" in
    node) package_node ;;
    swift) package_swift ;;
    kotlin) package_kotlin ;;
    all)
        package_node
        package_swift
        package_kotlin
        ;;
esac

echo "Packaged the same signed Apple provider runtime for SDK carrier: $SDK"
