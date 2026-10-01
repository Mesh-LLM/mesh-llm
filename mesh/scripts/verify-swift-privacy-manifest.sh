#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/lib/automation.sh"
MANIFEST="${1:-$REPO_ROOT/mesh/sdk/swift/PrivacyInfo.xcprivacy}"
XCFRAMEWORK="${2:-}"
privacy_args=(--template "$MANIFEST" --plutil "$(command -v plutil)")
if [[ -n "$XCFRAMEWORK" ]]; then
    privacy_args+=(--xcframework "$XCFRAMEWORK")
fi
mesh_automation release swift-privacy "${privacy_args[@]}"
