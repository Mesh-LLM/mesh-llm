#!/usr/bin/env bash
# Shared default for direct shell entry points; just reads the same data file.
# Explicit caller/SDK deployment targets remain authoritative.
if [[ "$(uname -s)" == Darwin ]]; then
  if [[ -z "${MACOSX_DEPLOYMENT_TARGET:-}" ]]; then
    MACOSX_DEPLOYMENT_TARGET="$(cat "$(dirname "${BASH_SOURCE[0]}")/macos-deployment-target.txt")"
  fi
  export MACOSX_DEPLOYMENT_TARGET
fi
