#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:-$REPO_ROOT/sdk/swift/PrivacyInfo.xcprivacy}"
XCFRAMEWORK="${2:-}"

python3 - "$MANIFEST" <<'PY'
import plistlib
import sys

manifest_path = sys.argv[1]
with open(manifest_path, "rb") as fh:
    manifest = plistlib.load(fh)

expected = {
    "NSPrivacyAccessedAPICategoryFileTimestamp": {"C617.1", "3B52.1"},
    "NSPrivacyAccessedAPICategoryDiskSpace": {"E174.1"},
    "NSPrivacyAccessedAPICategorySystemBootTime": {"35F9.1"},
}

if manifest.get("NSPrivacyTracking") is not False:
    raise SystemExit("NSPrivacyTracking must be false")
if manifest.get("NSPrivacyCollectedDataTypes") != []:
    raise SystemExit("NSPrivacyCollectedDataTypes must stay empty unless SDK data collection changes")
if manifest.get("NSPrivacyTrackingDomains") != []:
    raise SystemExit("NSPrivacyTrackingDomains must stay empty when tracking is false")

actual = {}
for entry in manifest.get("NSPrivacyAccessedAPITypes", []):
    category = entry.get("NSPrivacyAccessedAPIType")
    reasons = set(entry.get("NSPrivacyAccessedAPITypeReasons", []))
    if not category or not reasons:
        raise SystemExit(f"invalid required-reason API entry: {entry!r}")
    if category in actual:
        raise SystemExit(f"duplicate required-reason API entry: {category}")
    actual[category] = reasons

for category, reasons in expected.items():
    if actual.get(category) != reasons:
        raise SystemExit(
            f"{category} reasons mismatch: {sorted(actual.get(category, []))} != {sorted(reasons)}"
        )

unexpected = sorted(set(actual) - set(expected))
if unexpected:
    raise SystemExit(f"unexpected required-reason API categories: {', '.join(unexpected)}")
PY

plutil -lint "$MANIFEST" >/dev/null
echo "verified Swift privacy manifest: $MANIFEST"

if [[ -n "$XCFRAMEWORK" ]]; then
  if [[ ! -d "$XCFRAMEWORK" ]]; then
    echo "XCFramework does not exist: $XCFRAMEWORK" >&2
    exit 1
  fi

  privacy_count="$(find "$XCFRAMEWORK" -name PrivacyInfo.xcprivacy | wc -l | tr -d ' ')"
  if [[ "$privacy_count" -lt 1 ]]; then
    echo "PrivacyInfo.xcprivacy not embedded in $XCFRAMEWORK" >&2
    exit 1
  fi

  while IFS= read -r embedded_manifest; do
    cmp -s "$MANIFEST" "$embedded_manifest" || {
      echo "embedded privacy manifest differs from template: $embedded_manifest" >&2
      exit 1
    }
    plutil -lint "$embedded_manifest" >/dev/null
  done < <(find "$XCFRAMEWORK" -name PrivacyInfo.xcprivacy -print)

  echo "verified $privacy_count embedded privacy manifest file(s) in $XCFRAMEWORK"
fi
