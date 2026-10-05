#!/usr/bin/env bash
# Stable workspace entrypoint; implementation lives with its owning product.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/skippy/scripts/skippy-ci-smoke.sh" "$@"
