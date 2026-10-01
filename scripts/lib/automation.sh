#!/usr/bin/env bash

mesh_automation() {
    if [[ -n "${MESH_LLM_AUTOMATION_BIN:-}" ]]; then
        "$MESH_LLM_AUTOMATION_BIN" "$@"
    else
        (cd "$REPO_ROOT" && cargo xtool "$@")
    fi
}
