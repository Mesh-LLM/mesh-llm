#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/lib/automation.sh
source "$REPO_ROOT/scripts/lib/automation.sh"

if [[ "$#" == 1 && "$1" == --help ]]; then
    mesh_automation automation ui-build --help
    exit "$?"
fi
UI_DIR="${1:?usage: build-ui.sh /path/to/ui}"
if [[ "$#" != 1 ]]; then
    echo 'usage: build-ui.sh /path/to/ui' >&2
    exit 1
fi
mesh_automation automation ui-build --ui-dir "$UI_DIR"
