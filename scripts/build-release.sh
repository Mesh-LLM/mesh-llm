#!/usr/bin/env bash
# Compatibility entry point. All Unix host profiles share build-host.sh so
# development, PR, main, and release builds cannot drift.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/build-host.sh" --profile release
