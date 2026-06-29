#!/usr/bin/env bash
set -euo pipefail

TMP_ROOT=""
trap 'rm -rf "$TMP_ROOT"' EXIT

usage() {
    cat >&2 <<'EOF'
Usage: scripts/verify-native-runtime-package.sh <artifact-dir-or-tar.gz> [...]

Verifies MeshLLM native runtime artifacts:
  - manifest schema and resolver fields
  - artifact directory name matches runtime.id
  - all runtime.libraries exist
  - library_sha256 matches the primary library
  - Linux shared-library RUNPATH/RPATH is relocatable and resolves packaged deps
  - archive checksum sidecar when present
EOF
}

sha256_file() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        echo "shasum or sha256sum is required" >&2
        exit 1
    fi
}

python_bin() {
    local candidate
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    echo "Python 3.9 or newer is required to verify native runtimes" >&2
    exit 1
}

verify_sidecar_checksum() {
    local archive="$1"
    local sidecar="$archive.sha256"
    if [[ ! -f "$sidecar" ]]; then
        return 0
    fi
    local expected actual
    expected="$(awk '{print $1}' "$sidecar")"
    actual="$(sha256_file "$archive")"
    if [[ "$expected" != "$actual" ]]; then
        echo "archive checksum mismatch: $archive" >&2
        echo "  expected: $expected" >&2
        echo "  actual:   $actual" >&2
        exit 1
    fi
}

artifact_dir_for_input() {
    local input="$1"
    if [[ -d "$input" ]]; then
        printf '%s\n' "$input"
        return 0
    fi
    case "$input" in
        *.tar.gz|*.tgz) ;;
        *)
            echo "unsupported native runtime artifact input: $input" >&2
            exit 1
            ;;
    esac
    verify_sidecar_checksum "$input"
    if [[ -z "$TMP_ROOT" ]]; then
        TMP_ROOT="$(mktemp -d)"
    fi
    local extract_dir
    extract_dir="$TMP_ROOT/$(basename "$input" | tr -cd 'A-Za-z0-9_.-')"
    mkdir -p "$extract_dir"
    tar -C "$extract_dir" -xzf "$input"
    local count
    count="$(find "$extract_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
    if [[ "$count" != "1" ]]; then
        echo "expected archive to contain one top-level artifact directory: $input" >&2
        exit 1
    fi
    find "$extract_dir" -mindepth 1 -maxdepth 1 -type d -print -quit
}

verify_artifact_dir() {
    local artifact_dir="$1"
    local manifest="$artifact_dir/manifest.json"
    if [[ ! -f "$manifest" ]]; then
        echo "missing manifest: $manifest" >&2
        exit 1
    fi
    "$(python_bin)" - "$artifact_dir" "$manifest" <<'PY'
import hashlib
import json
import os
import sys

artifact_dir, manifest_path = sys.argv[1:3]
with open(manifest_path, encoding="utf-8") as fh:
    manifest = json.load(fh)

if "runtime" not in manifest:
    raise SystemExit("missing manifest field: runtime")
runtime = manifest["runtime"]
required = {
    "id",
    "mesh_version",
    "skippy_abi",
    "platform",
    "backend",
    "libraries",
}
missing = sorted(required - runtime.keys())
if missing:
    raise SystemExit(f"missing runtime manifest field(s): {', '.join(missing)}")
if os.path.basename(os.path.normpath(artifact_dir)) != runtime["id"]:
    raise SystemExit("artifact directory name must match runtime id")
if not isinstance(runtime["libraries"], list) or not runtime["libraries"]:
    raise SystemExit("runtime libraries must be a non-empty list")
platform = runtime["platform"]
if not isinstance(platform, dict) or not platform.get("os") or not platform.get("arch"):
    raise SystemExit("runtime platform must declare os and arch")
backend = runtime["backend"]
if not isinstance(backend, dict) or not backend.get("kind"):
    raise SystemExit("runtime backend must declare kind")

for rel_path in runtime["libraries"]:
    if os.path.isabs(rel_path) or ".." in rel_path.split(os.sep):
        raise SystemExit(f"library path must be relative inside the artifact: {rel_path}")
    path = os.path.join(artifact_dir, rel_path)
    if not os.path.isfile(path):
        raise SystemExit(f"missing library: {path}")

build = manifest.get("build") or {}
library_sha256 = build.get("library_sha256")
primary_library = build.get("primary_library") or runtime["libraries"][0]
if library_sha256:
    primary = os.path.join(artifact_dir, primary_library)
    with open(primary, "rb") as fh:
        actual = hashlib.sha256(fh.read()).hexdigest()
    if actual != library_sha256:
        raise SystemExit(
            f"library_sha256 mismatch for {primary_library}: {actual} != {library_sha256}"
        )
PY
    verify_macos_runtime_paths "$artifact_dir" "$manifest"
    verify_linux_runtime_paths "$artifact_dir" "$manifest"
    echo "verified native runtime artifact: $artifact_dir"
}

verify_macos_runtime_paths() {
    local artifact_dir="$1"
    local manifest="$2"
    if ! find -L "$artifact_dir" -type f -name '*.dylib' -print -quit | grep -q .; then
        return 0
    fi
    if ! command -v otool >/dev/null 2>&1; then
        echo "otool is required to verify macOS native runtime dylibs" >&2
        exit 1
    fi
    "$(python_bin)" - "$artifact_dir" "$manifest" <<'PY'
import json
import os
import subprocess
import sys

artifact_dir, manifest_path = sys.argv[1:3]
with open(manifest_path, encoding="utf-8") as fh:
    manifest = json.load(fh)

libraries = manifest["runtime"]["libraries"]
library_names = {os.path.basename(path) for path in libraries}
for rel_path in libraries:
    if not rel_path.endswith(".dylib"):
        continue
    path = os.path.join(artifact_dir, rel_path)
    load_output = subprocess.check_output(["otool", "-L", path], text=True)
    deps = [line.split()[0] for line in load_output.splitlines()[1:] if line.strip()]
    for dep in deps:
        if os.path.basename(dep) in library_names and dep.startswith("/"):
            raise SystemExit(f"{rel_path} depends on absolute packaged dylib path: {dep}")

    link_output = subprocess.check_output(["otool", "-l", path], text=True)
    has_loader_path_rpath = False
    in_rpath = False
    for line in link_output.splitlines():
        fields = line.split()
        if fields[:2] == ["cmd", "LC_RPATH"]:
            in_rpath = True
            continue
        if in_rpath and fields[:1] == ["path"]:
            if len(fields) > 1 and fields[1] == "@loader_path":
                has_loader_path_rpath = True
            in_rpath = False
    if not has_loader_path_rpath:
        raise SystemExit(f"{rel_path} is missing @loader_path LC_RPATH")
PY
}

verify_linux_runtime_paths() {
    local artifact_dir="$1"
    local manifest="$2"
    if ! "$(python_bin)" - "$manifest" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    manifest = json.load(fh)
raise SystemExit(0 if manifest["runtime"]["platform"].get("os") == "linux" else 1)
PY
    then
        return 0
    fi
    if ! command -v readelf >/dev/null 2>&1; then
        echo "readelf is required to verify Linux native runtime shared libraries" >&2
        exit 1
    fi
    "$(python_bin)" - "$artifact_dir" "$manifest" <<'PY'
import json
import os
import platform
import re
import shutil
import subprocess
import sys

artifact_dir, manifest_path = sys.argv[1:3]
with open(manifest_path, encoding="utf-8") as fh:
    manifest = json.load(fh)

libraries = manifest["runtime"]["libraries"]
library_names = {os.path.basename(path) for path in libraries}
artifact_root = os.path.realpath(artifact_dir)
dynamic_re = re.compile(r"\((NEEDED|RPATH|RUNPATH)\).*\[(.*)\]")
suspicious_tokens = (
    "/home/runner/work",
    ".deps/llama-build",
    "build-stage-abi",
)


def dynamic_entries(path: str) -> tuple[list[str], list[str]]:
    output = subprocess.check_output(["readelf", "-d", path], text=True)
    needed: list[str] = []
    search_paths: list[str] = []
    for line in output.splitlines():
        match = dynamic_re.search(line)
        if not match:
            continue
        tag, value = match.groups()
        if tag == "NEEDED":
            needed.append(value)
        else:
            search_paths.extend(entry for entry in value.split(":") if entry)
    return needed, search_paths


def verify_ldd_resolution(rel_path: str, needed: list[str]) -> None:
    packaged_needed = [dep for dep in needed if os.path.basename(dep) in library_names]
    if not packaged_needed or platform.system() != "Linux":
        return
    if shutil.which("ldd") is None:
        raise SystemExit("ldd is required to verify Linux packaged dependency resolution")
    path = os.path.join(artifact_dir, rel_path)
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    output = subprocess.check_output(["ldd", path], env=env, text=True, stderr=subprocess.STDOUT)
    for dep in packaged_needed:
        dep_name = os.path.basename(dep)
        match = re.search(rf"^\s*{re.escape(dep_name)}\s+=>\s+(\S+)", output, re.MULTILINE)
        if match is None:
            raise SystemExit(f"{rel_path} ldd output is missing packaged dependency {dep_name}")
        resolved = match.group(1)
        if resolved == "not":
            raise SystemExit(f"{rel_path} does not resolve packaged dependency {dep_name} without LD_LIBRARY_PATH")
        if not os.path.realpath(resolved).startswith(artifact_root + os.sep):
            raise SystemExit(
                f"{rel_path} resolves packaged dependency {dep_name} outside artifact: {resolved}"
            )


for rel_path in libraries:
    name = os.path.basename(rel_path)
    if ".so" not in name:
        continue
    needed, search_paths = dynamic_entries(os.path.join(artifact_dir, rel_path))
    packaged_needed = [dep for dep in needed if os.path.basename(dep) in library_names]
    for entry in search_paths:
        if any(token in entry for token in suspicious_tokens):
            raise SystemExit(f"{rel_path} contains build-directory runtime search path: {entry}")
        if entry.startswith("/"):
            raise SystemExit(f"{rel_path} contains absolute runtime search path: {entry}")
    if packaged_needed and not any("$ORIGIN" in entry for entry in search_paths):
        joined = ", ".join(packaged_needed)
        raise SystemExit(f"{rel_path} needs packaged libraries ({joined}) but is missing $ORIGIN RPATH/RUNPATH")
    verify_ldd_resolution(rel_path, needed)
PY
}

if [[ "$#" -lt 1 ]]; then
    usage
    exit 1
fi

for input in "$@"; do
    artifact_dir="$(artifact_dir_for_input "$input")"
    verify_artifact_dir "$artifact_dir"
done
