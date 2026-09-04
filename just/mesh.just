# Download the default model (GLM-4.7-Flash Q4_K_M, 17GB)
download-model:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{ models_dir }}"
    if [ -f "{{ model }}" ]; then
        echo "Model already exists: {{ model }}"
    else
        echo "Downloading GLM-4.7-Flash Q4_K_M (~17GB)..."
        curl -L -o "{{ model }}" \
            "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf"
    fi

# ── QUIC Mesh ──────────────────────────────────────────────────

mesh_bin := env("MESH_LLM_BIN", "target/release/mesh-llm")

# Prints an invite token for other nodes to join.
mesh-worker gguf=model:
    "{{ mesh_bin }}" --model {{ gguf }}

# Join an existing mesh and serve through the embedded runtime.
mesh-join join="" port="9337" gguf=model split="":
    #!/usr/bin/env bash
    set -euo pipefail
    ARGS="--model {{ gguf }} --port {{ port }}"
    if [ -n "{{ join }}" ]; then
        ARGS="$ARGS --join {{ join }}"
    fi
    if [ -n "{{ split }}" ]; then
        ARGS="$ARGS --tensor-split {{ split }}"
    fi
    exec "{{ mesh_bin }}" $ARGS

# Create a portable product tarball containing the neutral host and default runtime.
bundle output="/tmp/mesh-llm-bundle.tar.gz": release-build
    #!/usr/bin/env bash
    set -euo pipefail
    staging_dir="$(mktemp -d)"
    trap 'rm -rf "$staging_dir"' EXIT
    version="v$("{{ mesh_bin }}" --version | awk '{print $NF}')"
    scripts/package-release.sh "$version" "$staging_dir"
    stable_archive=""
    for candidate in "$staging_dir"/mesh-llm-*.tar.gz; do
        [[ -f "$candidate" ]] || continue
        case "$(basename "$candidate")" in
            mesh-llm-"$version"-*) continue ;;
        esac
        if [[ -n "$stable_archive" ]]; then
            echo "multiple stable product archives were produced" >&2
            exit 1
        fi
        stable_archive="$candidate"
    done
    if [[ -z "$stable_archive" ]]; then
        echo "product packaging did not produce a stable archive" >&2
        exit 1
    fi
    mkdir -p "$(dirname "{{ output }}")"
    cp "$stable_archive" "{{ output }}"
    cp "$stable_archive.sha256" "{{ output }}.sha256"
    echo "Bundle: {{ output }} ($(du -sh "{{ output }}" | cut -f1))"
