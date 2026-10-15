#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Build one platform's plugin package, following mesh-llm's plugin package
# contract (docs/plugins/README.md in Mesh-LLM/mesh-llm):
#
#   capsules-<version>-<target>.tar.gz
#     capsules/
#       capsules                   the executable
#       plugin.toml
#       plugin-manifest.json                printed by the executable itself
#       bundle/register-mesh-plugin-ui.js   the Evidence page
#       README.md                           (INSTALL.md)
#       LICENSE
#
# Usage: scripts/package.sh VERSION TARGET BINARY OUT_DIR [MANIFEST_JSON]
#   MANIFEST_JSON  the output of `BINARY --print-package-manifest`, when the
#                  binary cannot run here (packaging another platform's
#                  build); by default the binary is run to print it.
#
# Needs GNU tar (`tar` on Linux, `gtar` elsewhere), gzip, jq and a sha256 tool.
# The archive is byte-for-byte deterministic for the same inputs: sorted
# entries, owner 0, fixed modes, mtime SOURCE_DATE_EPOCH (or 0), gzip -n.
# Only .tar.gz: mesh-llm's .zip extractor does not keep the executable bit.
set -eu

PLUGIN=capsules
[ $# -ge 4 ] || { echo "usage: $0 VERSION TARGET BINARY OUT_DIR [MANIFEST_JSON]" >&2; exit 2; }
version=$1 target=$2 binary=$3 out=$4 manifest_in=${5:-}
root=$(cd "$(dirname "$0")/.." && pwd)

fail() { echo "package: $*" >&2; exit 1; }

case "$target" in
  aarch64-apple-darwin|x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu) ;;
  *) fail "unsupported target $target" ;;
esac
core=$(printf '%s' "$version" | sed -n 's/^\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)\(-[0-9A-Za-z.-][0-9A-Za-z.-]*\)\{0,1\}$/\1/p')
[ -n "$core" ] || fail "version $version is not MAJOR.MINOR.PATCH[-PRERELEASE]"
declared=$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$root/plugin.toml")
[ "$declared" = "$core" ] || fail "plugin.toml declares $declared, the release is $version; bump plugin.toml first"
[ -x "$binary" ] || fail "no executable at $binary"
[ "$(ls "$root/bundle" 2>/dev/null)" = "register-mesh-plugin-ui.js" ] ||
  fail "bundle/ must hold exactly register-mesh-plugin-ui.js (run pnpm build in web-ui/)"

if command -v gtar >/dev/null 2>&1; then tar=gtar; else tar=tar; fi
"$tar" --version 2>/dev/null | grep -q 'GNU tar' || fail "GNU tar is required"
if command -v sha256sum >/dev/null 2>&1; then sha() { sha256sum "$1"; }; else sha() { shasum -a 256 "$1"; }; fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
stage=$work/$PLUGIN
mkdir -p "$stage/bundle"

if [ -n "$manifest_in" ]; then
  cp "$manifest_in" "$stage/plugin-manifest.json"
else
  "$binary" --print-package-manifest > "$stage/plugin-manifest.json" || fail "the binary could not print its manifest"
fi
# The printed manifest must equal the reviewed one: a drift means the reviewed
# file is not what ships.
[ "$(jq -S . "$stage/plugin-manifest.json")" = "$(jq -S . "$root/plugin.package.json")" ] ||
  fail "the binary's manifest differs from plugin.package.json"

cp "$binary" "$stage/$PLUGIN"
cp "$root/plugin.toml" "$root/LICENSE" "$stage/"
cp "$root/INSTALL.md" "$stage/README.md"
cp "$root/bundle/register-mesh-plugin-ui.js" "$stage/bundle/"

find "$stage" -type d -exec chmod 755 {} +
find "$stage" -type f -exec chmod 644 {} +
chmod 755 "$stage/$PLUGIN"

mkdir -p "$out"
archive=$out/$PLUGIN-$version-$target.tar.gz
( cd "$work" && "$tar" --sort=name --format=ustar --owner=0 --group=0 --numeric-owner \
    --mtime="@${SOURCE_DATE_EPOCH:-0}" -cf - "$PLUGIN" ) | gzip -n -9 > "$archive"
sha "$archive"
