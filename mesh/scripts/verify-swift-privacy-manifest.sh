#!/usr/bin/env bash
# Lint the Swift privacy manifest; with an xcframework, require each slice to embed it unchanged.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${1:-$REPO_ROOT/mesh/sdk/swift/PrivacyInfo.xcprivacy}"
XCFRAMEWORK="${2:-}"
plutil -lint "$MANIFEST"
[[ -n "$XCFRAMEWORK" ]] || exit 0
found=0
while IFS= read -r embedded; do
    found=1
    cmp -s <(plutil -convert xml1 -o - "$MANIFEST") <(plutil -convert xml1 -o - "$embedded") ||
        { echo "privacy manifest differs: $embedded" >&2; exit 1; }
done < <(find "$XCFRAMEWORK" -name PrivacyInfo.xcprivacy -not -path "$MANIFEST")
[[ "$found" -eq 1 ]] || { echo "no PrivacyInfo.xcprivacy embedded in $XCFRAMEWORK" >&2; exit 1; }
echo "privacy manifest embedded in $XCFRAMEWORK"
