#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_ROOT=""
ARTIFACT_DIR_RESULT=""
ARTIFACT_INDEX=0
PORTABLE=0
trap 'rm -rf "$TMP_ROOT"' EXIT

usage() {
    cat >&2 <<'EOF'
Usage: scripts/verify-native-runtime-package.sh [--portable] <artifact-dir-or-tar.gz> [...]

Verifies MeshLLM native runtime artifacts:
  - manifest schema and resolver fields
  - artifact directory name matches runtime.id
  - all runtime.libraries exist
  - library_sha256 matches the primary library
  - Linux shared-library RUNPATH/RPATH is relocatable and resolves packaged deps
  - Windows non-system DLL imports are present in the artifact
  - required archive checksum sidecar
  - archive paths and links cannot escape the extraction directory

--portable validates integrity, archive shape, manifest schema, paths, and
checksums without running host-specific binary dependency probes.
EOF
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
    "$(python_bin)" "$SCRIPT_DIR/verify-checksum-sidecar.py" "$archive"
}

artifact_dir_for_input() {
    local input="$1"
    if [[ -d "$input" ]]; then
        ARTIFACT_DIR_RESULT="$input"
        return 0
    fi
    case "$input" in
        *.tar.gz|*.tgz) ;;
        *)
            echo "unsupported native runtime artifact input: $input" >&2
            exit 1
            ;;
    esac
    verify_sidecar_checksum "$input" || return 1
    if [[ -z "$TMP_ROOT" ]]; then
        TMP_ROOT="$(mktemp -d)"
    fi
    local extract_dir
    extract_dir="$TMP_ROOT/artifact-$ARTIFACT_INDEX"
    ARTIFACT_INDEX=$((ARTIFACT_INDEX + 1))
    mkdir -p "$extract_dir"
    "$(python_bin)" "$SCRIPT_DIR/safe-extract-tar.py" "$input" "$extract_dir" ||
        return 1
    local count entry
    count="$(
        find "$extract_dir" -mindepth 1 -maxdepth 1 -print |
            wc -l |
            tr -d ' '
    )"
    entry="$(find "$extract_dir" -mindepth 1 -maxdepth 1 -print -quit)"
    if [[ "$count" != "1" || ! -d "$entry" || -L "$entry" ]]; then
        echo "expected archive to contain one top-level artifact directory: $input" >&2
        return 1
    fi
    ARTIFACT_DIR_RESULT="$entry"
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
from pathlib import Path, PurePosixPath
import re
import sys

artifact_dir, manifest_path = sys.argv[1:3]
artifact_root = Path(artifact_dir).resolve(strict=True)
windows_drive = re.compile(r"^[A-Za-z]:")


def artifact_file(label, raw_path):
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise SystemExit(f"{label} path must be a non-empty string")
    if "\\" in raw_path:
        raise SystemExit(
            f"{label} path must use forward slashes inside the artifact: "
            f"{raw_path}"
        )
    rel_path = PurePosixPath(raw_path)
    if (
        rel_path.is_absolute()
        or windows_drive.match(raw_path)
        or ".." in rel_path.parts
    ):
        raise SystemExit(
            f"{label} path must be relative inside the artifact: {raw_path}"
        )
    candidate = artifact_root.joinpath(*rel_path.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise SystemExit(f"missing {label}: {candidate}") from None
    try:
        resolved.relative_to(artifact_root)
    except ValueError:
        raise SystemExit(
            f"{label} path resolves outside the artifact: {raw_path}"
        ) from None
    if not resolved.is_file():
        raise SystemExit(f"missing {label}: {candidate}")
    return resolved


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
    "files",
}
missing = sorted(required - runtime.keys())
if missing:
    raise SystemExit(f"missing runtime manifest field(s): {', '.join(missing)}")
for field in ("id", "mesh_version", "skippy_abi"):
    if not isinstance(runtime[field], str) or not runtime[field]:
        raise SystemExit(f"runtime {field} must be a non-empty string")
if os.path.basename(os.path.normpath(artifact_dir)) != runtime["id"]:
    raise SystemExit("artifact directory name must match runtime id")
if not isinstance(runtime["libraries"], list) or not runtime["libraries"]:
    raise SystemExit("runtime libraries must be a non-empty list")
platform = runtime["platform"]
if not isinstance(platform, dict):
    raise SystemExit("runtime platform must be an object")
runtime_os = platform.get("os")
runtime_arch = platform.get("arch")
if not isinstance(runtime_os, str) or not runtime_os:
    raise SystemExit("runtime platform must declare os and arch")
if runtime_os not in {"linux", "macos", "windows"}:
    raise SystemExit(f"unsupported runtime platform os: {runtime_os!r}")
if not isinstance(runtime_arch, str) or not runtime_arch:
    raise SystemExit("runtime platform must declare os and arch")
if runtime_arch not in {"x86_64", "aarch64", "arm"}:
    raise SystemExit(f"unsupported runtime platform arch: {runtime_arch!r}")
target = platform.get("target")
if not isinstance(target, str) or not target:
    raise SystemExit("runtime platform target must be a non-empty string")
target_contracts = {
    "aarch64-apple-darwin": ("macos", "aarch64"),
    "x86_64-apple-darwin": ("macos", "x86_64"),
    "x86_64-unknown-linux-gnu": ("linux", "x86_64"),
    "aarch64-unknown-linux-gnu": ("linux", "aarch64"),
    "armv7-unknown-linux-gnueabihf": ("linux", "arm"),
    "aarch64-linux-android": ("linux", "aarch64"),
    "armv7-linux-androideabi": ("linux", "arm"),
    "x86_64-linux-android": ("linux", "x86_64"),
    "x86_64-pc-windows-msvc": ("windows", "x86_64"),
}
target_contract = target_contracts.get(target)
if target_contract is None:
    raise SystemExit(f"unsupported runtime platform target: {target}")
if (runtime_os, runtime_arch) != target_contract:
    raise SystemExit(
        "runtime os/arch do not match target: "
        f"{runtime_os}/{runtime_arch} != "
        f"{target_contract[0]}/{target_contract[1]}"
    )
backend = runtime["backend"]
if not isinstance(backend, dict):
    raise SystemExit("runtime backend must be an object")
backend_kind = backend.get("kind")
if not isinstance(backend_kind, str) or not backend_kind:
    raise SystemExit("runtime backend must declare kind")
if backend_kind not in {"cpu", "metal", "cuda", "rocm", "vulkan"}:
    raise SystemExit(f"unsupported runtime backend kind: {backend_kind!r}")
backend_operating_systems = {
    "cpu": {"linux", "macos", "windows"},
    "metal": {"macos"},
    "cuda": {"linux", "windows"},
    "rocm": {"linux", "windows"},
    "vulkan": {"linux", "windows"},
}
if runtime_os not in backend_operating_systems[backend_kind]:
    raise SystemExit(
        f"runtime backend {backend_kind} is unsupported on {runtime_os}"
    )

for rel_path in runtime["libraries"]:
    artifact_file("library", rel_path)

files = runtime["files"]
tools = runtime["tools"] if "tools" in runtime else {}
if not isinstance(files, dict) or not isinstance(tools, dict):
    raise SystemExit("runtime files and tools must be checksum maps")
if not files:
    raise SystemExit("runtime files must be a non-empty checksum map")
missing_library_checksums = [
    rel_path
    for rel_path in runtime["libraries"]
    if rel_path not in files
]
if missing_library_checksums:
    raise SystemExit(
        "runtime libraries are missing file checksums: "
        + ", ".join(missing_library_checksums)
    )
sha256_pattern = re.compile(r"^[0-9a-f]{64}$")
for kind, checksums in (("file", files), ("tool", tools)):
    for rel_path, expected in checksums.items():
        path = artifact_file(kind, rel_path)
        if not isinstance(expected, str) or not sha256_pattern.fullmatch(
            expected
        ):
            raise SystemExit(
                f"{kind} checksum must be a canonical SHA-256 for {rel_path}"
            )
        with open(path, "rb") as fh:
            actual = hashlib.sha256(fh.read()).hexdigest()
        if actual != expected:
            raise SystemExit(f"{kind} checksum mismatch for {rel_path}")
        if (
            kind == "tool"
            and runtime_os != "windows"
            and os.name != "nt"
            and not os.access(path, os.X_OK)
        ):
            raise SystemExit(f"runtime tool is not executable: {rel_path}")

build = manifest.get("build")
if not isinstance(build, dict):
    raise SystemExit("runtime build metadata must be an object")
library_sha256 = build.get("library_sha256")
primary_library = build.get("primary_library")
if not isinstance(primary_library, str) or not primary_library:
    raise SystemExit("build.primary_library must be a non-empty string")
if primary_library not in runtime["libraries"]:
    raise SystemExit(
        "build.primary_library must be declared in runtime.libraries"
    )
if not isinstance(library_sha256, str) or not sha256_pattern.fullmatch(
    library_sha256
):
    raise SystemExit(
        "build.library_sha256 must be a canonical SHA-256"
    )
if files[primary_library] != library_sha256:
    raise SystemExit(
        "build.library_sha256 must match runtime.files for "
        f"{primary_library}"
    )
primary = artifact_file("primary library", primary_library)
with open(primary, "rb") as fh:
    actual = hashlib.sha256(fh.read()).hexdigest()
if actual != library_sha256:
    raise SystemExit(
        f"library_sha256 mismatch for {primary_library}: "
        f"{actual} != {library_sha256}"
    )
PY
    if [[ "$PORTABLE" == "1" ]]; then
        echo "verified portable native runtime artifact: $artifact_dir"
        return 0
    fi
    verify_macos_runtime_paths "$artifact_dir" "$manifest"
    verify_linux_runtime_paths "$artifact_dir" "$manifest"
    verify_windows_runtime_dependencies "$artifact_dir" "$manifest"
    echo "verified native runtime artifact: $artifact_dir"
}

verify_windows_runtime_dependencies() {
    local artifact_dir="$1"
    local manifest="$2"
    if ! "$(python_bin)" - "$manifest" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    runtime = json.load(fh)["runtime"]
raise SystemExit(0 if runtime["platform"].get("os") == "windows" else 1)
PY
    then
        return 0
    fi
    "$(python_bin)" "$SCRIPT_DIR/windows-native-runtime-deps.py" verify \
        --lib-dir "$artifact_dir/lib" \
        --scan-dir "$artifact_dir/tools"
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
tools = list((manifest["runtime"].get("tools") or {}).keys())
library_names = {os.path.basename(path) for path in libraries}
for rel_path in [*libraries, *tools]:
    if not (rel_path.endswith(".dylib") or rel_path in tools):
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
    expected_rpath = "@loader_path/../lib" if rel_path in tools else "@loader_path"
    if expected_rpath not in {
        fields[1]
        for line in link_output.splitlines()
        if (fields := line.split()) and fields[:1] == ["path"]
    }:
        raise SystemExit(f"{rel_path} is missing {expected_rpath} LC_RPATH")
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
tools = list((manifest["runtime"].get("tools") or {}).keys())
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


for rel_path in [*libraries, *tools]:
    name = os.path.basename(rel_path)
    if ".so" not in name and rel_path not in tools:
        continue
    needed, search_paths = dynamic_entries(os.path.join(artifact_dir, rel_path))
    packaged_needed = [dep for dep in needed if os.path.basename(dep) in library_names]
    for entry in search_paths:
        if any(token in entry for token in suspicious_tokens):
            raise SystemExit(f"{rel_path} contains build-directory runtime search path: {entry}")
        if entry.startswith("/"):
            raise SystemExit(f"{rel_path} contains absolute runtime search path: {entry}")
    expected_origin = "$ORIGIN/../lib" if rel_path in tools else "$ORIGIN"
    if packaged_needed and expected_origin not in search_paths:
        joined = ", ".join(packaged_needed)
        raise SystemExit(f"{rel_path} needs packaged libraries ({joined}) but is missing {expected_origin} RPATH/RUNPATH")
    verify_ldd_resolution(rel_path, needed)
PY
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --portable)
            PORTABLE=1
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "unknown argument: $1" >&2
            usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [[ "$#" -lt 1 ]]; then
    usage
    exit 1
fi

for input in "$@"; do
    artifact_dir_for_input "$input"
    verify_artifact_dir "$ARTIFACT_DIR_RESULT"
done
