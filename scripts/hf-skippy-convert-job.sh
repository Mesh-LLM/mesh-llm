#!/usr/bin/env bash
set -euo pipefail
# The prepared job image owns the pinned native automation executable and tool manifest.
automation="${MESH_LLM_AUTOMATION_BIN:?set the prepared pinned native xtask executable}"
case "$automation" in /*) ;; *) echo 'native conversion requires an absolute automation executable' >&2; exit 64;; esac
if [[ ! -f "$automation" || -L "$automation" || ! -x "$automation" ]]; then
  echo 'native conversion automation executable refused' >&2
  exit 64
fi
exec "$automation" automation hf-certify generic-job "$@"
