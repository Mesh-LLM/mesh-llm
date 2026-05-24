#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/dist/native-sdk-crates"
TMP_ROOT=""
trap 'rm -rf "$TMP_ROOT"' EXIT

usage() {
    cat >&2 <<'EOF'
Usage: scripts/package-native-sdk-crate.sh [options] <native-sdk-artifact-dir-or-tar.gz>

Generate a crates.io-ready native runtime crate from a verified native SDK
runtime artifact. The generated crate contains the native library files and
exports their paths through Cargo build metadata.

Options:
  --out DIR       Output directory. Defaults to dist/native-sdk-crates.
  -h, --help      Show this help.

Generated crates use:
  links = "meshllm_native_runtime"

Cargo exposes build metadata to dependents as:
  DEP_MESHLLM_NATIVE_RUNTIME_ARTIFACT_ID
  DEP_MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR
  DEP_MESHLLM_NATIVE_RUNTIME_MANIFEST
  DEP_MESHLLM_NATIVE_RUNTIME_LIB_DIR
  DEP_MESHLLM_NATIVE_RUNTIME_LIBRARY
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --out)
            OUT_DIR="${2:?missing output directory}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
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

if [[ "$#" -ne 1 ]]; then
    usage
    exit 1
fi

INPUT="$1"

"$SCRIPT_DIR/verify-native-sdk-package.sh" "$INPUT"

artifact_dir_for_input() {
    local input="$1"

    if [[ -d "$input" ]]; then
        printf '%s\n' "$input"
        return 0
    fi

    TMP_ROOT="$(mktemp -d)"
    tar -C "$TMP_ROOT" -xzf "$input"
    find "$TMP_ROOT" -mindepth 1 -maxdepth 1 -type d -print -quit
}

artifact_dir="$(artifact_dir_for_input "$INPUT")"
manifest="$artifact_dir/manifest.json"

read_manifest_field() {
    python3 - "$manifest" "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    manifest = json.load(fh)
value = manifest[sys.argv[2]]
if value is None:
    value = ""
print(value)
PY
}

artifact_id="$(read_manifest_field artifact_id)"
version="$(read_manifest_field sdk_version)"
platform="$(read_manifest_field platform)"
flavor="$(read_manifest_field flavor)"
target_triple="$(read_manifest_field target_triple)"
backend="$(read_manifest_field backend)"
crate_name="${artifact_id//_/-}"
lib_name="${crate_name//-/_}"
crate_dir="$OUT_DIR/$crate_name"

rm -rf "$crate_dir"
mkdir -p "$crate_dir/native" "$crate_dir/src"

cp "$manifest" "$crate_dir/native/manifest.json"
cp -R "$artifact_dir/lib" "$crate_dir/native/lib"

cat > "$crate_dir/Cargo.toml" <<EOF
[workspace]

[package]
name = "$crate_name"
version = "$version"
edition = "2021"
license = "MIT OR Apache-2.0"
description = "MeshLLM native runtime artifact for $platform $flavor"
repository = "https://github.com/Mesh-LLM/mesh-llm"
homepage = "https://github.com/Mesh-LLM/mesh-llm"
readme = "README.md"
links = "meshllm_native_runtime"
include = [
    "Cargo.toml",
    "README.md",
    "build.rs",
    "src/lib.rs",
    "native/manifest.json",
    "native/lib/*",
]

[lib]
name = "$lib_name"
path = "src/lib.rs"

[package.metadata.meshllm.native]
artifact_id = "$artifact_id"
platform = "$platform"
flavor = "$flavor"
target_triple = "$target_triple"
backend = "$backend"
EOF

cat > "$crate_dir/build.rs" <<'EOF'
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let source_artifact_dir = manifest_dir.join("native");
    let source_manifest_path = source_artifact_dir.join("manifest.json");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let artifact_dir = out_dir.join("native");
    if artifact_dir.exists() {
        fs::remove_dir_all(&artifact_dir).expect("remove stale native runtime from OUT_DIR");
    }
    copy_dir_all(&source_artifact_dir, &artifact_dir)
        .expect("copy native runtime artifact into OUT_DIR");

    let manifest_path = artifact_dir.join("manifest.json");
    let lib_dir = artifact_dir.join("lib");

    let manifest = fs::read_to_string(&manifest_path).expect("read native runtime manifest");
    let artifact_id = json_string_field(&manifest, "artifact_id").expect("manifest artifact_id");
    let library = json_string_field(&manifest, "library").expect("manifest library");

    let library_path = artifact_dir.join(&library);

    println!("cargo:rerun-if-changed={}", source_manifest_path.display());
    println!("cargo:rerun-if-changed={}", source_artifact_dir.join(&library).display());

    println!("cargo:artifact_id={artifact_id}");
    println!("cargo:artifact_dir={}", artifact_dir.display());
    println!("cargo:manifest={}", manifest_path.display());
    println!("cargo:lib_dir={}", lib_dir.display());
    println!("cargo:library={}", library_path.display());

    println!("cargo:rustc-env=MESHLLM_NATIVE_RUNTIME_ARTIFACT_ID={artifact_id}");
    println!(
        "cargo:rustc-env=MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR={}",
        artifact_dir.display()
    );
    println!(
        "cargo:rustc-env=MESHLLM_NATIVE_RUNTIME_MANIFEST={}",
        manifest_path.display()
    );
    println!(
        "cargo:rustc-env=MESHLLM_NATIVE_RUNTIME_LIB_DIR={}",
        lib_dir.display()
    );
    println!(
        "cargo:rustc-env=MESHLLM_NATIVE_RUNTIME_LIBRARY={}",
        library_path.display()
    );
}

fn copy_dir_all(source: &Path, destination: &Path) -> std::io::Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let destination_path = destination.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&entry.path(), &destination_path)?;
        } else {
            fs::copy(entry.path(), destination_path)?;
        }
    }
    Ok(())
}

fn json_string_field(source: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let key_index = source.find(&needle)?;
    let after_key = &source[key_index + needle.len()..];
    let colon_index = after_key.find(':')?;
    let mut rest = after_key[colon_index + 1..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    rest = &rest[1..];

    let mut value = String::new();
    let mut escaped = false;
    for ch in rest.chars() {
        if escaped {
            value.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' => return Some(value),
            _ => value.push(ch),
        }
    }
    None
}
EOF

cat > "$crate_dir/src/lib.rs" <<'EOF'
use std::path::PathBuf;

pub const MANIFEST_JSON: &str = include_str!("../native/manifest.json");
pub const ARTIFACT_ID: &str = env!("MESHLLM_NATIVE_RUNTIME_ARTIFACT_ID");

pub fn artifact_dir() -> PathBuf {
    PathBuf::from(env!("MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR"))
}

pub fn manifest_path() -> PathBuf {
    PathBuf::from(env!("MESHLLM_NATIVE_RUNTIME_MANIFEST"))
}

pub fn lib_dir() -> PathBuf {
    PathBuf::from(env!("MESHLLM_NATIVE_RUNTIME_LIB_DIR"))
}

pub fn library_path() -> PathBuf {
    PathBuf::from(env!("MESHLLM_NATIVE_RUNTIME_LIBRARY"))
}

EOF

cat > "$crate_dir/README.md" <<EOF
# $crate_name

This crate packages the MeshLLM native SDK runtime artifact:

- artifact: \`$artifact_id\`
- target: \`$target_triple\`
- backend: \`$backend\`
- flavor: \`$flavor\`

It is intended as a target/backend-specific dependency for Rust crates and SDK
packaging tools. The crate build script copies the native runtime into Cargo's
\`OUT_DIR/native\` and exports those build-output paths to dependent build
scripts through the \`meshllm_native_runtime\` link target:

\`\`\`text
DEP_MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR
DEP_MESHLLM_NATIVE_RUNTIME_MANIFEST
DEP_MESHLLM_NATIVE_RUNTIME_LIB_DIR
DEP_MESHLLM_NATIVE_RUNTIME_LIBRARY
\`\`\`

Application build scripts should copy \`DEP_MESHLLM_NATIVE_RUNTIME_ARTIFACT_DIR\`
into their final app bundle, installer, container image, or package resource
directory, then load \`DEP_MESHLLM_NATIVE_RUNTIME_LIBRARY\` from that packaged
location at runtime.

Rust code that depends on this crate directly can also call:

\`\`\`rust
$lib_name::artifact_dir();
$lib_name::library_path();
\`\`\`
EOF

(
    cd "$crate_dir"
    cargo package --allow-dirty
)

echo "generated native SDK runtime crate:"
echo "  crate:   $crate_name"
echo "  dir:     $crate_dir"
echo "  package: $crate_dir/target/package/$crate_name-$version.crate"
