#!/usr/bin/env bash
# Preserve both the executable entrypoint and the sourced release-name helpers.
set -euo pipefail
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    # shellcheck disable=SC1090
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/mesh/scripts/package-release.sh"
else
    exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/mesh/scripts/package-release.sh" "$@"
fi
