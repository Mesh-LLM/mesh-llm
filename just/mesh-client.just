# Start a lite client — no GPU, no model, just a local HTTP proxy to the mesh host.

# Only needs the mesh-llm binary (no llama.cpp binaries or model).
mesh-client join="" port="9337" console="3131" config="":
    #!/usr/bin/env bash
    set -euo pipefail
    args=(client --port "{{ port }}" --console "{{ console }}")
    if [[ -n "{{ join }}" ]]; then
        args+=(--join "{{ join }}")
    fi
    if [[ -n "{{ config }}" ]]; then
        args+=(--config "{{ config }}")
    fi
    "{{ mesh_bin }}" "${args[@]}"

# Build and auto-join a mesh (discover via Nostr)
auto: build
    "{{ mesh_bin }}" --auto
