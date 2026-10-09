#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    printf 'Usage: %s <mesh-llm-binary> <native-runtime-root>\n' "$0" >&2
    exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/automation.sh
source "$REPO_ROOT/scripts/lib/automation.sh"
binary="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
runtime_root="$(cd "$2" && pwd)"
mesh_automation automation client-readiness --binary "$binary" --native-runtime-root "$runtime_root"
