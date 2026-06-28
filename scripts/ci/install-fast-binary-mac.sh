#!/usr/bin/env bash
set -euo pipefail

repo="${1:-}"
install_dir="${2:-$HOME/.local/bin}"

command -v gh >/dev/null || { echo "GitHub CLI (gh) is required" >&2; exit 1; }
if [[ -n "$repo" ]]; then
    run_id="$(gh run list --repo "$repo" --workflow fast-reusable-binaries.yml --status success --limit 1 --json databaseId --jq '.[0].databaseId')"
else
    run_id="$(gh run list --workflow fast-reusable-binaries.yml --status success --limit 1 --json databaseId --jq '.[0].databaseId')"
fi
[[ -n "$run_id" ]] || { echo "No successful fast binary workflow run found" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
if [[ -n "$repo" ]]; then
    gh run download "$run_id" --repo "$repo" --name mesh-llm-mac-arm64-metal --dir "$tmp"
else
    gh run download "$run_id" --name mesh-llm-mac-arm64-metal --dir "$tmp"
fi
(cd "$tmp" && shasum -a 256 -c mesh-llm-mac-arm64-metal.tar.gz.sha256)
tar -xzf "$tmp/mesh-llm-mac-arm64-metal.tar.gz" -C "$tmp"
mkdir -p "$install_dir"
install -m 0755 "$tmp/mesh-llm-mac-arm64-metal/mesh-llm" "$install_dir/mesh-llm"
"$install_dir/mesh-llm" --version
