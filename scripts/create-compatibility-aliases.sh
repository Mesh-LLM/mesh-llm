#!/usr/bin/env bash
# Offer legacy 0.80.0 GPU archive names only where the pinned Skippy release
# publishes a replacement runtime. The Vulkan aliases are withheld.
set -euo pipefail

(($# == 1)) || { echo "usage: $0 <release-artifact-dir>" >&2; exit 2; }
artifacts="$(cd "$1" && pwd -P)"
cd "$artifacts"
sha256() { { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1"; } | cut -d ' ' -f 1; }
alias_of() {
    local host="$1" name
    shift
    [[ -s "$host" && -s "$host.sha256" ]] || { echo "missing checked host archive: $host" >&2; exit 1; }
    for name in "$@"; do
        [[ ! -e "$name" ]] || { echo "compatibility alias already exists: $name" >&2; exit 1; }
        cp "$host" "$name"
        printf '%s  %s\n' "$(sha256 "$name")" "$name" >"$name.sha256"
        cmp "$host" "$name"
        [[ "$(cut -d ' ' -f 1 "$name.sha256")" == "$(cut -d ' ' -f 1 "$host.sha256")" ]] ||
            { echo "$name is not the archive checked as $host" >&2; exit 1; }
    done
}
linux=mesh-llm-x86_64-unknown-linux-gnu
arm=mesh-llm-aarch64-unknown-linux-gnu
windows=mesh-llm-x86_64-pc-windows-msvc
alias_of "$linux.tar.gz" "$linux"-{cuda-12,cuda-13,rocm}.tar.gz
alias_of "$arm.tar.gz" "$arm"-{cuda-12,cuda-13}.tar.gz
alias_of "$windows.zip" "$windows"-{cuda-12,cuda-13,rocm}.zip
