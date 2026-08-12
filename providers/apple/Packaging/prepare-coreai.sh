#!/usr/bin/env bash
set -euo pipefail

APPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COREAI_ROOT="$APPLE_ROOT/.build/checkouts/coreai-models/swift"

[[ -d "$COREAI_ROOT" ]] || {
    echo "Core AI dependency checkout is missing; run swift package resolve first" >&2
    exit 2
}

chmod -R u+w "$COREAI_ROOT"

# The pinned Core AI package currently targets an earlier Foundation Models
# beta. Keep the compatibility delta explicit and local until Apple publishes
# a revision that builds against the Golden Gate SDK shipped with this repo.
python3 - "$COREAI_ROOT" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
replacements = {
    root / "Sources/CoreAIShared/Runtime/NDArray+Helpers.swift": (
        "    let view = array.mutableView(as: type)\n",
        "    var view = array.mutableView(as: type)\n",
    ),
    root / "Sources/CoreAILanguageModels/LanguageModel/CoreAILanguageModel.swift": (
        "LanguageModelCapabilities(caps)",
        "LanguageModelCapabilities(capabilities: caps)",
    ),
    root / "Sources/CoreAILanguageModels/VLM/CoreAIVisionLanguageModel.swift": (
        "LanguageModelCapabilities([.vision])",
        "LanguageModelCapabilities(capabilities: [.vision])",
    ),
}

for path, (before, after) in replacements.items():
    text = path.read_text()
    if before not in text:
        if after in text:
            continue
        raise SystemExit(f"Core AI compatibility patch no longer applies: {path}")
    path.write_text(text.replace(before, after))

# Swift 6.4 makes NDArray mutable views and their contiguous-element accessors
# explicitly mutating. The upstream package still has a few `let` bindings in
# those paths; normalize only bindings whose initializer is mutableView.
for path in (root / "Sources").rglob("*.swift"):
    text = path.read_text()
    patched = re.sub(r"^(\s*)let (\w*[Vv]iew) = (.+\.mutableView\(.+)$", r"\1var \2 = \3", text, flags=re.MULTILINE)
    if patched != text:
        path.write_text(patched)
PY
