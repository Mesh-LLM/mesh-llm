#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

./sdk/swift/scripts/build-xcframework.sh

scripts/ci-sdk-fixture.sh "$1" "$2" "$3" -- \
    bash -lc '
        set -euo pipefail
        cd '"$REPO_ROOT"'
        swift run \
            --package-path sdk/swift/example/MeshExampleApp \
            MeshExampleApp \
            "$MESH_SDK_INVITE_TOKEN"
    '
