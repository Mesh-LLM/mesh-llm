#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
GLM-5.2 routed-down quantization must run locally for this experiment.

Use the local BF16 GGUF shards on micstudio:

  SOURCE_ROOT=/Users/lab/glm52-work/bf16-gguf \
  SOURCE_PREFIX=BF16 \
  scripts/glm52-q2-routed-down-layer-package.sh

Hugging Face can be used later for publishing artifacts, but not for
quantization execution.
EOF

exit 2
