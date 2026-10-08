#!/usr/bin/env bash
# Native competitive acquisition accepts an explicit source-pinned request.
# Prepare the executor with `just automation-bootstrap`; see
# skippy/docs/COMPETITIVE_BENCHMARK.md for request and executable identities.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
executor="${MESH_LLM_AUTOMATION_BIN:-$repo_root/target/debug/xtask}"
if [[ ! -x "$executor" ]]; then
  echo "Prepare native automation with just automation-bootstrap, or set MESH_LLM_AUTOMATION_BIN." >&2
  exit 1
fi
exec "$executor" automation replay-matrix competitive-inputs-prefetch "$@"
