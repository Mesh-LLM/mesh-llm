#!/usr/bin/env bash

# Toolchain discovery belongs to the caller's HOME before any isolated client HOME.
MESH_AUTOMATION_DISCOVERY_HOME="${HOME:-}"

mesh_automation() {
    if [[ "${MESH_LLM_AUTOMATION_BIN+set}" == set ]]; then
        if [[ "$MESH_LLM_AUTOMATION_BIN" != /* || ! -f "$MESH_LLM_AUTOMATION_BIN" || ! -x "$MESH_LLM_AUTOMATION_BIN" ]]; then
            echo 'MESH_LLM_AUTOMATION_BIN must be an absolute executable' >&2
            return 1
        fi
        "$MESH_LLM_AUTOMATION_BIN" "$@"
    else
        env HOME="$MESH_AUTOMATION_DISCOVERY_HOME" just --justfile "$REPO_ROOT/Justfile" automation-run "$@"
    fi
}
