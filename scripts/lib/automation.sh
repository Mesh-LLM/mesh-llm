#!/usr/bin/env bash

# Toolchain discovery belongs to the caller's HOME before any isolated client HOME.
MESH_AUTOMATION_DISCOVERY_HOME="${HOME:-}"

mesh_automation() {
    if [[ "${MESH_LLM_AUTOMATION_BIN+set}" == set ]]; then
        local automation_binary="$MESH_LLM_AUTOMATION_BIN"
        if [[ "$automation_binary" =~ ^[A-Za-z]:[\\/] || "$automation_binary" == \\\\* ]]; then
            if [[ "$OSTYPE" != msys* && "$OSTYPE" != cygwin* ]]; then
                echo 'MESH_LLM_AUTOMATION_BIN must be an absolute executable' >&2
                return 1
            fi
            automation_binary="$(cygpath -u -- "$automation_binary")" || {
                echo 'MESH_LLM_AUTOMATION_BIN must be an absolute executable' >&2
                return 1
            }
        fi
        if [[ "$automation_binary" != /* || ! -f "$automation_binary" || ! -x "$automation_binary" ]]; then
            echo 'MESH_LLM_AUTOMATION_BIN must be an absolute executable' >&2
            return 1
        fi
        "$automation_binary" "$@"
    else
        env HOME="$MESH_AUTOMATION_DISCOVERY_HOME" just --justfile "$REPO_ROOT/Justfile" automation-run "$@"
    fi
}
