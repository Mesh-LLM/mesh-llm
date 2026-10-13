#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/automation.sh"
SDK=""
ASSET_DIR=""

usage() {
    echo "Usage: $0 [--sdk node|swift|kotlin|all] [asset-dir]" >&2
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --sdk) SDK="${2:?missing SDK name}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            if [[ -n "$ASSET_DIR" ]]; then
                usage
                exit 1
            fi
            ASSET_DIR="$1"
            shift
            ;;
    esac
done

case "$SDK" in
    ""|node|swift|kotlin|all) ;;
    *) echo "unsupported SDK: $SDK" >&2; usage; exit 1 ;;
esac

verify_sdk() {
    case "$1" in
        node)
            mesh_automation prepared-input sdk-console-verify "$REPO_ROOT/mesh/sdk/node/console"
            grep -Fq '"console/"' "$REPO_ROOT/mesh/sdk/node/package.json" \
                || { echo "mesh/sdk/node/package.json must include console/ in files" >&2; exit 1; }
            ;;
        swift)
            mesh_automation prepared-input sdk-console-verify "$REPO_ROOT/mesh/sdk/swift/Sources/MeshLLM/Resources/Console"
            grep -Fq '.copy("Resources/Console")' "$REPO_ROOT/Package.swift" \
                || { echo "Package.swift must copy Resources/Console" >&2; exit 1; }
            ;;
        kotlin)
            mesh_automation prepared-input sdk-console-verify "$REPO_ROOT/mesh/sdk/kotlin/src/main/resources/mesh-llm/console"
            grep -Fq 'src/main/resources/mesh-llm/console' "$REPO_ROOT/mesh/sdk/kotlin/build.gradle.kts" \
                || { echo "mesh/sdk/kotlin/build.gradle.kts must package console resources" >&2; exit 1; }
            ;;
    esac
}

if [[ -n "$ASSET_DIR" ]]; then
    mesh_automation prepared-input sdk-console-verify "$ASSET_DIR"
elif [[ "$SDK" == "all" || -z "$SDK" ]]; then
    verify_sdk node
    verify_sdk swift
    verify_sdk kotlin
else
    verify_sdk "$SDK"
fi
