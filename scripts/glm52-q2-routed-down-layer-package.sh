#!/usr/bin/env bash
# Stable workspace entrypoint; implementation lives with its owning product.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/skippy/scripts/glm52-q2-routed-down-layer-package.sh" "$@"
