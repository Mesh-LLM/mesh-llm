#!/usr/bin/env bash
# Verify that Cargo resolved every Skippy dependency from the selected checkout.
#   scripts/verify-skippy-source.sh <skippy-checkout>
set -euo pipefail

(($# == 1)) || { echo "usage: $0 <skippy-checkout>" >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkout="$(cd "$1" && pwd -P)"
windows=false
case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) checkout="$(cygpath -m "$checkout")"; windows=true ;;
esac
command -v jq >/dev/null 2>&1 || { echo "jq is required to verify Skippy sources" >&2; exit 1; }

required="$(awk '
    /^(hf_hub|skippy-[a-z-]+) = \{/ {
        name = $1
        if (name == "hf_hub") name = "skippy-hf-hub"
        print name
    }
' "$root/Cargo.toml" | jq -R -s 'split("\n") | map(select(length > 0))')"
metadata="$(cd "$root" && cargo metadata --format-version 1)"
jq -e --arg checkout "$checkout" --argjson windows "$windows" --argjson required "$required" '
    def normalized_path:
        if $windows then gsub("\\\\"; "/") | ascii_downcase else . end;
    def from_checkout:
        .name as $name |
        (.manifest_path | normalized_path) as $manifest |
        .source == null and
        (($manifest == (($checkout + "/tools/process-supervision/Cargo.toml") | normalized_path) and $name == "skippy-subprocess") or
         (($manifest | startswith(($checkout + "/crates/") | normalized_path)) and
          ($manifest | endswith(("/" + $name + "/Cargo.toml") | normalized_path))));
    .packages as $packages |
      ($required | length > 0) and
      all($required[]; . as $name |
          [$packages[] | select(.name == $name)] | (length == 1) and (.[0] | from_checkout)) and
      all($packages[] | select(.name | startswith("skippy-")); from_checkout)
' <<<"$metadata" >/dev/null || {
    echo "Cargo resolved a Skippy crate outside $checkout, or a pinned Skippy crate is missing" >&2
    exit 1
}
echo "Cargo resolved all Skippy crates from $checkout"
