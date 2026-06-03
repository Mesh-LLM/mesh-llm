#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/verify-sdk-console-assets.sh [--sdk node|swift|kotlin|all] [asset-dir]

Verifies packaged SDK console assets. When --sdk is provided, the canonical
SDK resource directory and package metadata are checked. When asset-dir is
provided, only that directory is checked.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SDK=""
ASSET_DIR=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --sdk)
            SDK="${2:?missing SDK name}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -n "$ASSET_DIR" ]]; then
                usage
                exit 1
            fi
            ASSET_DIR="$1"
            shift
            ;;
    esac
done

case "$SDK" in
    ""|node|swift|kotlin|all) ;;
    *)
        echo "unsupported SDK: $SDK" >&2
        usage
        exit 1
        ;;
esac

verify_dir() {
    local dir="$1"
    python3 - "$dir" <<'PY'
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
import sys

root = Path(sys.argv[1])
index = root / "index.html"
manifest = root / "manifest.txt"

if not index.is_file():
    raise SystemExit(f"missing console index.html: {index}")
if not manifest.is_file():
    raise SystemExit(f"missing console manifest.txt: {manifest}")

class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for name in ("src", "href"):
            value = attrs.get(name)
            if value:
                self.refs.append(value)

parser = AssetParser()
parser.feed(index.read_text(encoding="utf-8"))

def local_path(ref):
    parsed = urlparse(ref)
    if parsed.scheme or parsed.netloc or ref.startswith(("data:", "mailto:", "#")):
        return None
    path = parsed.path
    if not path or path == "/":
        return None
    return path[1:] if path.startswith("/") else path

missing = []
for ref in parser.refs:
    rel = local_path(ref)
    if rel and not (root / rel).is_file():
        missing.append(rel)
if missing:
    raise SystemExit("console index references missing assets: " + ", ".join(sorted(missing)))

manifest_entries = [
    line.strip()
    for line in manifest.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if "index.html" not in manifest_entries:
    raise SystemExit("console manifest.txt must include index.html")
for rel in manifest_entries:
    if rel.startswith("/") or "\\" in rel or any(part in ("", "..") for part in rel.split("/")):
        raise SystemExit(f"unsafe console manifest path: {rel}")
    if not (root / rel).is_file():
        raise SystemExit(f"console manifest references missing asset: {rel}")

if not any(path.suffix == ".js" for path in root.glob("assets/*")):
    raise SystemExit("console assets must include at least one JavaScript asset under assets/")

referenced_css = [local_path(ref) for ref in parser.refs if local_path(ref) and local_path(ref).endswith(".css")]
if referenced_css and not any(path.suffix == ".css" for path in root.glob("assets/*")):
    raise SystemExit("console index references CSS, but no CSS asset exists under assets/")

print(f"verified console assets: {root}")
PY
}

verify_node_metadata() {
    grep -Fq '"console/"' "$REPO_ROOT/sdk/node/package.json" \
        || { echo "sdk/node/package.json must include console/ in files" >&2; exit 1; }
}

verify_swift_metadata() {
    grep -Fq '.copy("Resources/Console")' "$REPO_ROOT/Package.swift" \
        || { echo "Package.swift must copy Resources/Console" >&2; exit 1; }
}

verify_kotlin_metadata() {
    grep -Fq 'src/main/resources/mesh-llm/console' "$REPO_ROOT/sdk/kotlin/build.gradle.kts" \
        || { echo "sdk/kotlin/build.gradle.kts must package console resources" >&2; exit 1; }
}

verify_sdk() {
    case "$1" in
        node)
            verify_dir "$REPO_ROOT/sdk/node/console"
            verify_node_metadata
            ;;
        swift)
            verify_dir "$REPO_ROOT/sdk/swift/Sources/MeshLLM/Resources/Console"
            verify_swift_metadata
            ;;
        kotlin)
            verify_dir "$REPO_ROOT/sdk/kotlin/src/main/resources/mesh-llm/console"
            verify_kotlin_metadata
            ;;
    esac
}

if [[ -n "$ASSET_DIR" ]]; then
    verify_dir "$ASSET_DIR"
elif [[ "$SDK" == "all" || -z "$SDK" ]]; then
    verify_sdk node
    verify_sdk swift
    verify_sdk kotlin
else
    verify_sdk "$SDK"
fi
