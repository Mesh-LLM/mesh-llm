#!/usr/bin/env zsh
# Compatibility entry point for the unified macOS development product.
# The host is dynamic; native ABI compilation happens only in runtime packaging.

setopt errexit nounset pipefail

SCRIPT_DIR="${0:A:h}"
exec "$SCRIPT_DIR/build-development-product.sh" --profile "${MESH_LLM_BUILD_PROFILE:-debug}" "$@"
