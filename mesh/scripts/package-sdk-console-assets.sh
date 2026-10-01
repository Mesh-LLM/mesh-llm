#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/package-sdk-console-assets.sh [options]

Build and copy the MeshLLM web console distribution into SDK package resource
locations.

Options:
  --sdk node|swift|kotlin|all  SDK package to update. Defaults to all.
  --dist DIR                   Existing console dist directory. Defaults to mesh/crates/mesh-llm-ui/dist.
  --skip-build                 Do not run scripts/build-ui.sh before copying.
  -h, --help                   Show this help.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/automation.sh"
SDK="all"
DIST_DIR="$REPO_ROOT/mesh/crates/mesh-llm-ui/dist"
SKIP_BUILD=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --sdk)
            SDK="${2:?missing SDK name}"
            shift 2
            ;;
        --dist)
            DIST_DIR="${2:?missing dist directory}"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
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

if [[ "$SKIP_BUILD" == "0" ]]; then
    MESH_LLM_BUILD_PROFILE="${MESH_LLM_BUILD_PROFILE:-release}" \
        "$SCRIPT_DIR/build-ui.sh" "$REPO_ROOT/mesh/crates/mesh-llm-ui"
fi

if [[ ! -f "$DIST_DIR/index.html" ]]; then
    echo "console dist is missing index.html: $DIST_DIR" >&2
    exit 1
fi

write_manifest() {
    local dest="$1"
    mesh_automation prepared-input sdk-console-manifest "$dest"
}

copy_console_assets() {
    local dest="$1"
    rm -rf "$dest"
    mkdir -p "$dest"
    cp -R "$DIST_DIR"/. "$dest"/
    rm -f "$dest/.mesh-llm-ui-build-env"
    : > "$dest/.gitkeep"
    write_manifest "$dest"
    "$SCRIPT_DIR/verify-sdk-console-assets.sh" "$dest"
}

package_node() {
    copy_console_assets "$REPO_ROOT/mesh/sdk/node/console"
}

package_swift() {
    copy_console_assets "$REPO_ROOT/mesh/sdk/swift/Sources/MeshLLM/Resources/Console"
}

package_kotlin() {
    copy_console_assets "$REPO_ROOT/mesh/sdk/kotlin/src/main/resources/mesh-llm/console"
}

case "$SDK" in
    node)
        package_node
        ;;
    swift)
        package_swift
        ;;
    kotlin)
        package_kotlin
        ;;
    all)
        package_node
        package_swift
        package_kotlin
        ;;
esac
