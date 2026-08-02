#!/usr/bin/env bash
set -euo pipefail

: "${TARGETARCH:?TARGETARCH is required}"
: "${SCCACHE_VERSION:?SCCACHE_VERSION is required}"

download_cache="${DOWNLOAD_CACHE_DIR:-/var/cache/mesh-downloads}"
install_dir="${SCCACHE_INSTALL_DIR:-/usr/local/bin}"
mkdir -p "$download_cache"
mkdir -p "$install_dir"

case "$TARGETARCH" in
  amd64) rust_arch=x86_64 ;;
  arm64) rust_arch=aarch64 ;;
  *) echo "unsupported architecture: $TARGETARCH" >&2; exit 1 ;;
esac

archive="sccache-v${SCCACHE_VERSION}-${rust_arch}-unknown-linux-musl.tar.gz"
base="https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}"
archive_path="${download_cache}/${archive}"
checksum_path="${archive_path}.sha256"

expected_sha=""
if [[ -s "$archive_path" && -s "$checksum_path" ]]; then
  expected_sha="$(awk 'NR == 1 { print $1 }' "$checksum_path")"
fi
if [[ ! "$expected_sha" =~ ^[[:xdigit:]]{64}$ ]] \
    || ! printf '%s  %s\n' "$expected_sha" "$archive_path" | sha256sum -c - >/dev/null 2>&1; then
  rm -f "$archive_path" "$checksum_path"
  archive_tmp="$(mktemp "${archive_path}.tmp.XXXXXX")"
  checksum_tmp="$(mktemp "${checksum_path}.tmp.XXXXXX")"
  if ! curl -fsSL --retry 3 "${base}/${archive}" -o "$archive_tmp" \
      || ! curl -fsSL --retry 3 "${base}/${archive}.sha256" -o "$checksum_tmp"; then
    rm -f "$archive_tmp" "$checksum_tmp"
    exit 1
  fi
  expected_sha="$(awk 'NR == 1 { print $1 }' "$checksum_tmp")"
  if [[ ! "$expected_sha" =~ ^[[:xdigit:]]{64}$ ]] \
      || ! printf '%s  %s\n' "$expected_sha" "$archive_tmp" | sha256sum -c -; then
    rm -f "$archive_tmp" "$checksum_tmp"
    exit 1
  fi
  mv -f "$archive_tmp" "$archive_path"
  mv -f "$checksum_tmp" "$checksum_path"
fi

temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT
tar -xzf "$archive_path" -C "$temporary"
install -m 0755 \
  "$temporary/sccache-v${SCCACHE_VERSION}-${rust_arch}-unknown-linux-musl/sccache" \
  "$install_dir/sccache"
"$install_dir/sccache" --version
