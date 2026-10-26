#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/automation.sh"
OUT=""
REPO="${GITHUB_REPOSITORY:-Mesh-LLM/mesh-llm}"
TAG="${RELEASE_TAG:-}"
RUNTIME_VERSION=""
TMP_ROOT=""
trap 'rm -rf "$TMP_ROOT"' EXIT

usage() {
    cat >&2 <<'EOF'
Usage: scripts/generate-native-runtime-release-manifest.sh --tag TAG --out FILE [--repo OWNER/REPO] [--runtime-version VERSION] <native-runtime.tar.gz> [...]

Generates native-runtimes.json for a GitHub release from packaged native
runtime artifacts. Each artifact archive must have its canonical .sha256
sidecar and contain a manifest.json with the native runtime resolver fields
emitted by package-native-runtime.sh. The tag locates the published archives;
--runtime-version selects their required runtime release (defaults to Skippy
RUNTIME_VERSION), independently of the product publication tag.
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --out)
            OUT="${2:?missing output file}"
            shift 2
            ;;
        --repo)
            REPO="${2:?missing repo}"
            shift 2
            ;;
        --runtime-version)
            RUNTIME_VERSION="${2:?missing runtime version}"
            shift 2
            ;;
        --tag)
            TAG="${2:?missing release tag}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "unknown argument: $1" >&2
            usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [[ -z "$OUT" || -z "$TAG" || "$#" -lt 1 ]]; then
    usage
    exit 1
fi

if [[ -z "$RUNTIME_VERSION" ]]; then
    RUNTIME_VERSION="$(cat "$SCRIPT_DIR/../skippy/crates/skippy-native-runtime/RUNTIME_VERSION")"
fi
if [[ ! "$RUNTIME_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
    echo "invalid Skippy runtime release version: $RUNTIME_VERSION" >&2
    exit 1
fi

if [[ -z "$TMP_ROOT" ]]; then
    TMP_ROOT="$(mktemp -d)"
fi

for archive in "$@"; do
    mesh_automation native verify-runtime-package --portable "$archive"
done

mesh_automation product runtime-release-manifest "$OUT" "$REPO" "$TAG" "$RUNTIME_VERSION" "$TMP_ROOT" "$@"
echo "generated native runtime release manifest: $OUT"
