#!/bin/bash
set -euo pipefail
url=''
destination=''
while (( $# )); do
    case "$1" in
        -o) destination="$2"; shift 2 ;;
        --retry|--connect-timeout|--max-time) shift 2 ;;
        -fsSL) shift ;;
        https://github.com/mozilla/sccache/releases/download/v0.16.0/*) url="$1"; shift ;;
        *) echo 'unowned curl argument' >&2; exit 91 ;;
    esac
done
file="${url##*/}"
case "$file" in
    sccache-v0.16.0-x86_64-unknown-linux-musl.tar.gz|sccache-v0.16.0-x86_64-unknown-linux-musl.tar.gz.sha256) ;;
    *) echo 'unowned local artifact' >&2; exit 92 ;;
esac
case "$destination" in "$FIXTURE_ROOT/cache/$file.tmp."*) ;; *) exit 93 ;; esac
printf '%s\n' "$file" >> "$FIXTURE_ROOT/curl.calls"
if [[ "$FAIL_AT" == offline || "$FAIL_AT" == archive ]]; then
    printf 'partial failed download' > "$destination"
    exit 22
fi
if [[ "$file" == *.sha256 && "$FAIL_AT" == checksum ]]; then
    printf 'partial failed checksum' > "$destination"
    exit 22
fi
if [[ "$file" == *.sha256 && "$FAIL_AT" == digest ]]; then
    printf '%064d\n' 0 > "$destination"
else
    cp "$FIXTURE_ROOT/source/$file" "$destination"
fi
