#!/usr/bin/env bash
# Build and validate the public website before a pull request can merge.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root/mesh/website"
npm ci
npm run test:website-data
npm run test:catalog-filters
npm run build
[[ -s "$root/docs/index.html" && -s "$root/docs/crates/index.html" ]] || {
    echo "website build did not produce the home page and crate index" >&2
    exit 1
}
