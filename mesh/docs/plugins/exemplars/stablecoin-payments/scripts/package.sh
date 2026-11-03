#!/usr/bin/env bash
# Package only the production binaries. No wallet state, credentials or fixtures.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'Usage: package.sh BINARY_DIRECTORY TARGET_TRIPLE VERSION' >&2
  exit 2
fi
binary_directory=$(cd -- "$1" && pwd)
target_triple=$2
package_version=$3
[[ "$target_triple" =~ ^[a-zA-Z0-9_-]+$ && "$package_version" =~ ^[a-zA-Z0-9._+-]+$ ]] || exit 2
workspace_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p -- "$workspace_root/dist"
stage_root=$(mktemp -d)
trap 'rm -rf -- "$stage_root"' EXIT
for plugin_name in pay-stablecoin wallet-ousd-tempo; do
  plugin_stage="$stage_root/$plugin_name"
  mkdir -- "$plugin_stage"
  cp -- "$binary_directory/$plugin_name" "$plugin_stage/$plugin_name"
  chmod 755 "$plugin_stage/$plugin_name"
  printf 'name = "%s"\nversion = "%s"\n' "$plugin_name" "$package_version" > "$plugin_stage/plugin.toml"
  "$plugin_stage/$plugin_name" --manifest > "$plugin_stage/plugin-manifest.json"
  archive="$workspace_root/dist/$plugin_name-v$package_version-$target_triple.tar.gz"
  tar -C "$stage_root" -czf "$archive" "$plugin_name"
  if command -v sha256sum >/dev/null; then
    (cd -- "$workspace_root/dist" && sha256sum "$(basename -- "$archive")") > "$archive.sha256"
  else
    (cd -- "$workspace_root/dist" && shasum -a 256 "$(basename -- "$archive")") > "$archive.sha256"
  fi
  printf '%s\n' "$archive"
done
