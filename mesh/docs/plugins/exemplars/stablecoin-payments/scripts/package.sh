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
for plugin_name in pay-stablecoin wallet-ousd-tempo wallet-usdt-ethereum; do
  plugin_stage="$stage_root/$plugin_name"
  mkdir -- "$plugin_stage"
  cp -- "$binary_directory/$plugin_name" "$plugin_stage/$plugin_name"
  chmod 755 "$plugin_stage/$plugin_name"
  printf 'name = "%s"\nversion = "%s"\n' "$plugin_name" "$package_version" > "$plugin_stage/plugin.toml"
  "$plugin_stage/$plugin_name" --manifest > "$plugin_stage/plugin-manifest.json"
  if [[ "$plugin_name" == wallet-usdt-ethereum ]]; then
    wdk_source="$workspace_root/wallets/usdt-ethereum-wdk"
    if [[ ! -d "$wdk_source/node_modules" ]]; then
      echo 'Run just wdk-install before packaging the Ethereum wallet.' >&2
      exit 1
    fi
    mkdir -- "$plugin_stage/wdk"
    cp -- "$wdk_source/"*.mjs "$wdk_source/package.json" "$wdk_source/package-lock.json" "$plugin_stage/wdk/"
    cp -R -- "$wdk_source/node_modules" "$plugin_stage/wdk/node_modules"
  fi
  archive="$workspace_root/dist/$plugin_name-v$package_version-$target_triple.tar.gz"
  COPYFILE_DISABLE=1 tar -C "$stage_root" -czf "$archive" "$plugin_name"
  if command -v sha256sum >/dev/null; then
    (cd -- "$workspace_root/dist" && sha256sum "$(basename -- "$archive")") > "$archive.sha256"
  else
    (cd -- "$workspace_root/dist" && shasum -a 256 "$(basename -- "$archive")") > "$archive.sha256"
  fi
  printf '%s\n' "$archive"
done
