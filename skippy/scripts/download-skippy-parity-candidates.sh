#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/automation.sh
source "$REPO_ROOT/scripts/lib/automation.sh"
usage() {
  cat >&2 <<'EOF'
usage: scripts/download-skippy-parity-candidates.sh [--dry-run] [--status CSV] [--priority CSV]

Downloads Hugging Face GGUF/package artifacts for Skippy parity rows that still
need certification evidence. By default this includes:

  needs_candidate,candidate_multimodal,package_or_remote_only

and only P0/P1 popularity-priority rows.

Candidate rows may carry an immutable revision and per-file integrity records.
The downloader pins those revisions and verifies every downloaded model and
projector file before returning success.

Environment:
  SKIPPY_PARITY_MANIFEST=/path/to/llama-parity-candidates.json
  SKIPPY_PARITY_DOWNLOAD_STATUSES=needs_candidate,candidate_multimodal
  SKIPPY_PARITY_DOWNLOAD_PRIORITIES=p0,p1

Examples:
  scripts/download-skippy-parity-candidates.sh --dry-run
  scripts/download-skippy-parity-candidates.sh
  scripts/download-skippy-parity-candidates.sh --status needs_candidate
  scripts/download-skippy-parity-candidates.sh --priority p0
EOF
}

for argument in "$@"; do
  case "$argument" in -h|--help) usage; exit 0 ;; esac
done
hf_command="$(command -v hf)" || { echo 'hf CLI is required' >&2; exit 1; }
if [[ "$hf_command" != /* ]]; then hf_command="$PWD/$hf_command"; fi
mesh_automation models parity-download --cadence manual \
  --manifest "${SKIPPY_PARITY_MANIFEST:-$REPO_ROOT/skippy/docs/llama-parity-candidates.json}" \
  --model-manifest "${SKIPPY_PARITY_MODEL_MANIFEST:-$REPO_ROOT/ci/model-artifacts/manifests/skippy-parity.json}" \
  --status "${SKIPPY_PARITY_DOWNLOAD_STATUSES:-needs_candidate,candidate_multimodal,package_or_remote_only}" \
  --priority "${SKIPPY_PARITY_DOWNLOAD_PRIORITIES:-p0,p1}" \
  --hf-command "$hf_command" "$@"
