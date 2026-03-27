#!/usr/bin/env bash

set -euo pipefail

REPO="${MESH_LLM_INSTALL_REPO:-michaelneale/mesh-llm}"
INSTALL_DIR="${MESH_LLM_INSTALL_DIR:-$HOME/.local/bin}"

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required command not found: $1" >&2
        exit 1
    fi
}

path_contains_install_dir() {
    case ":$PATH:" in
        *":$INSTALL_DIR:"*) return 0 ;;
        *) return 1 ;;
    esac
}

linux_flavor_suffix() {
    if command -v nvidia-smi >/dev/null 2>&1 || command -v nvcc >/dev/null 2>&1 || [[ -e /dev/nvidiactl ]] || [[ -d /proc/driver/nvidia/gpus ]]; then
        echo "-cuda"
        return
    fi
    echo ""
}

asset_name() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"
    case "$os/$arch" in
        Darwin/arm64)
            echo "mesh-llm-aarch64-apple-darwin.tar.gz"
            ;;
        Linux/x86_64)
            echo "mesh-llm-x86_64-unknown-linux-gnu$(linux_flavor_suffix).tar.gz"
            ;;
        *)
            echo "error: unsupported platform: $os/$arch" >&2
            exit 1
            ;;
    esac
}

need_cmd curl
need_cmd tar
need_cmd mktemp

ASSET="$(asset_name)"
URL="https://github.com/${REPO}/releases/latest/download/${ASSET}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

ARCHIVE="$TMP_DIR/$ASSET"
echo "Downloading $URL"
curl -fsSL "$URL" -o "$ARCHIVE"

tar -xzf "$ARCHIVE" -C "$TMP_DIR"

if [[ ! -d "$TMP_DIR/mesh-bundle" ]]; then
    echo "error: release archive did not contain mesh-bundle/" >&2
    exit 1
fi

mkdir -p "$INSTALL_DIR"
for file in "$TMP_DIR"/mesh-bundle/*; do
    mv -f "$file" "$INSTALL_DIR/"
done

echo "Installed $ASSET to $INSTALL_DIR"

if ! path_contains_install_dir; then
    echo
    echo "$INSTALL_DIR is not on your PATH."
    echo "Add it with one of these commands:"
    echo
    echo "bash:"
    echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc"
    echo "  source ~/.bashrc"
    echo
    echo "zsh:"
    echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.zshrc"
    echo "  source ~/.zshrc"
fi
