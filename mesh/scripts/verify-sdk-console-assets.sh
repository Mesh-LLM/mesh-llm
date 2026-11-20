#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

verify_console_dir() {
    local root="${1%/}" name entry ref
    for name in index.html manifest.txt; do
        [[ -f "$root/$name" ]] || { echo "missing console $name: $root/$name" >&2; exit 1; }
    done
    grep -qx 'index.html' "$root/manifest.txt" || { echo "console manifest.txt must include index.html" >&2; exit 1; }
    while IFS= read -r entry; do
        [[ -n "$entry" ]] || continue
        if [[ "$entry" == /* || "$entry" == *\\* || "/$entry/" == *"/../"* || "/$entry/" == *"//"* ]]; then
            echo "unsafe console manifest path: $entry" >&2; exit 1
        fi
        [[ -f "$root/$entry" ]] || { echo "console manifest references missing asset: $entry" >&2; exit 1; }
    done <"$root/manifest.txt"
    while IFS= read -r ref; do
        ref="${ref#./}"; ref="${ref#/}"
        [[ -f "$root/$ref" ]] || { echo "console index references missing asset: $ref" >&2; exit 1; }
    done < <(grep -oE '(src|href)="(\./|/)?assets/[^"?#]+' "$root/index.html" | sed -E 's/^(src|href)="//')
    compgen -G "$root/assets/*.js" >/dev/null || { echo "console assets must include at least one JavaScript asset under assets/" >&2; exit 1; }
    echo "verified console assets: $root"
}
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
            verify_console_dir "$REPO_ROOT/mesh/sdk/node/console"
            grep -Fq '"console/"' "$REPO_ROOT/mesh/sdk/node/package.json" \
                || { echo "mesh/sdk/node/package.json must include console/ in files" >&2; exit 1; }
            ;;
        swift)
            verify_console_dir "$REPO_ROOT/mesh/sdk/swift/Sources/MeshLLM/Resources/Console"
            grep -Fq '.copy("Resources/Console")' "$REPO_ROOT/Package.swift" \
                || { echo "Package.swift must copy Resources/Console" >&2; exit 1; }
            ;;
        kotlin)
            verify_console_dir "$REPO_ROOT/mesh/sdk/kotlin/src/main/resources/mesh-llm/console"
            grep -Fq 'src/main/resources/mesh-llm/console' "$REPO_ROOT/mesh/sdk/kotlin/build.gradle.kts" \
                || { echo "mesh/sdk/kotlin/build.gradle.kts must package console resources" >&2; exit 1; }
            ;;
    esac
}

if [[ -n "$ASSET_DIR" ]]; then
    verify_console_dir "$ASSET_DIR"
elif [[ "$SDK" == "all" || -z "$SDK" ]]; then
    verify_sdk node
    verify_sdk swift
    verify_sdk kotlin
else
    verify_sdk "$SDK"
fi
