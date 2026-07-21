#!/usr/bin/env bash
set -euo pipefail

skippy_quantize_bin="${SKIPPY_QUANTIZE_BIN:-target/release/skippy-quantize}"
if [[ ! -x "$skippy_quantize_bin" ]]; then
  echo "missing executable: $skippy_quantize_bin" >&2
  echo "build it with: just skippy-quantize-standalone-release-build" >&2
  exit 1
fi

exec "$skippy_quantize_bin" quantize-layer-package "$@"
