#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mapfile -t crates < <(
    sed -n '/^publish_crates=(/,/^)/p' scripts/publish-crates.sh \
        | sed -n 's/^    \([[:alnum:]_-]*\)$/\1/p'
)

if [[ "${#crates[@]}" -eq 0 ]]; then
    echo "No crates found in scripts/publish-crates.sh" >&2
    exit 1
fi

echo "Building Rustdoc for ${#crates[@]} published crates"

cargo clean --doc

cargo_args=(doc --locked --no-deps)
for crate in "${crates[@]}"; do
    cargo_args+=(--package "$crate")
done
cargo "${cargo_args[@]}"

landing_page=""
if [[ -f docs/crates/index.html ]]; then
    landing_page="$(mktemp)"
    cp docs/crates/index.html "$landing_page"
fi

rm -rf docs/crates
mkdir -p docs/crates
cp -R target/doc/. docs/crates/

if [[ -n "$landing_page" ]]; then
    cp "$landing_page" docs/crates/index.html
    rm -f "$landing_page"
fi

echo "Published crate docs to docs/crates/"
