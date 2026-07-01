#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: scripts/release-version.sh <version|vversion>" >&2
    exit 1
fi

raw_version="$1"
version="${raw_version#v}"

if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
    echo "invalid version: $raw_version" >&2
    echo "expected semantic version like 0.49.0, 0.49.0-rc.1, v0.49.0, or v0.49.0-rc.1" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

require_file() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo "missing required file: $file" >&2
        exit 1
    fi
}

update_manifest_version() {
    local file="$1"
    local next="$2"
    local before
    local after
    if perl -0777 -ne 'exit((/\[package\][^[]*?\nversion\.workspace\s*=\s*true/s) ? 0 : 1)' "$file"; then
        return
    fi
    before="$(cat "$file")"
    after="$(perl -0777 -pe 's/(\[package\][^[]*?\nversion\s*=\s*")[^"]+(")/${1}'"$next"'$2/s' "$file")"
    if [[ "$before" == "$after" ]]; then
        if perl -0777 -ne 'exit((/\[package\][^[]*?\nversion\s*=\s*"'"$next"'"/s) ? 0 : 1)' "$file"; then
            return
        fi
        echo "failed to update [package].version in $file" >&2
        exit 1
    fi
    printf '%s\n' "$after" >"$file"
}

update_workspace_version() {
    local file="$1"
    local next="$2"
    local before
    local after
    before="$(cat "$file")"
    after="$(perl -0777 -pe 's/(\[workspace\.package\][^[]*?\nversion\s*=\s*")[^"]+(")/${1}'"$next"'$2/s' "$file")"
    if [[ "$before" == "$after" ]]; then
        if perl -0777 -ne 'exit((/\[workspace\.package\][^[]*?\nversion\s*=\s*"'"$next"'"/s) ? 0 : 1)' "$file"; then
            return
        fi
        echo "failed to update [workspace.package].version in $file" >&2
        exit 1
    fi
    printf '%s\n' "$after" >"$file"
}

read_workspace_version() {
    local file="$1"
    perl -0ne 'print "$1\n" if /\[workspace\.package\][^[]*?\nversion\s*=\s*"([^"]+)"/s' "$file"
}

update_literal_version_references() {
    local file="$1"
    local previous="$2"
    local next="$3"
    local before
    local after
    before="$(cat "$file")"
    after="$(
        perl -0777 -pe '
            BEGIN {
                $previous = quotemeta($ARGV[0]);
                $next = $ARGV[1];
                shift @ARGV;
                shift @ARGV;
            }
            s/v$previous/v$next/g;
            s/$previous/$next/g;
        ' "$previous" "$next" "$file"
    )"
    if [[ "$before" == "$after" ]]; then
        return
    fi
    printf '%s\n' "$after" >"$file"
}

update_json_package_version_references() {
    local file="$1"
    local previous="$2"
    local next="$3"
    local before
    local after

    command -v node >/dev/null 2>&1 || {
        echo "node is required on PATH to update JSON package versions" >&2
        exit 1
    }

    before="$(cat "$file")"
    after="$(
        node - "$previous" "$next" "$file" <<'JS'
const fs = require("fs");

const [previous, next, file] = process.argv.slice(2);
const input = fs.readFileSync(file, "utf8");
const data = JSON.parse(input);
let changed = false;

function updateVersion(owner, label) {
  if (!owner || !Object.prototype.hasOwnProperty.call(owner, "version")) {
    return;
  }
  if (owner.version === next) {
    return;
  }
  if (owner.version !== previous) {
    throw new Error(`${file}: ${label}.version is ${owner.version}, expected ${previous}`);
  }
  owner.version = next;
  changed = true;
}

updateVersion(data, "root");
if (data.packages && data.packages[""]) {
  updateVersion(data.packages[""], "packages[\"\"]");
}

process.stdout.write(changed ? `${JSON.stringify(data, null, 2)}\n` : input);
JS
    )"
    if [[ "$before" == "$after" ]]; then
        return
    fi
    printf '%s' "$after" >"$file"
}

update_known_mesh_versions() {
    local file="$1"
    local next="$2"
    local before
    local after
    before="$(cat "$file")"
    if perl -0777 -ne '
        BEGIN {
            $next = quotemeta($ARGV[0]);
            shift @ARGV;
        }
        exit((/"$next"/) ? 0 : 1)
    ' "$next" "$file"; then
        return
    fi
    after="$(perl -0777 -pe 's/(fn known_mesh_llm_versions\(\).*?\&\[\n\s*)/${1}"'"$next"'", /s' "$file")"
    if [[ "$before" == "$after" ]]; then
        echo "failed to add $next to known mesh-llm versions in $file" >&2
        exit 1
    fi
    printf '%s\n' "$after" >"$file"
}

update_versioned_path_dependency_versions() {
    local file="$1"
    local next="$2"
    local before
    local after
    before="$(cat "$file")"
    after="$(perl -0777 -pe 's/(\{\s*(?=[^}]*\bpath\s*=)(?=[^}]*\bversion\s*=)[^}]*\bversion\s*=\s*")[^"]+(")/${1}'"$next"'$2/gs' "$file")"
    if [[ "$before" == "$after" ]]; then
        return
    fi
    printf '%s\n' "$after" >"$file"
}

update_gradle_project_version() {
    local file="$1"
    local next="$2"
    local before
    local after
    before="$(cat "$file")"
    after="$(perl -0777 -pe 's/(\nversion\s*=\s*")[^"]+(")/${1}'"$next"'$2/s' "$file")"
    if [[ "$before" == "$after" ]]; then
        if perl -0777 -ne 'exit((/\nversion\s*=\s*"'"$next"'"/s) ? 0 : 1)' "$file"; then
            return
        fi
        echo "failed to update Gradle project version in $file" >&2
        exit 1
    fi
    printf '%s\n' "$after" >"$file"
}

refresh_cargo_lock_versions() {
    local attempt

    for attempt in 1 2 3 4 5; do
        if (cd "$REPO_ROOT" && cargo metadata --format-version 1 >/dev/null); then
            return
        fi

        if [[ "$attempt" -eq 5 ]]; then
            echo "cargo metadata failed after $attempt attempts" >&2
            return 1
        fi

        echo "cargo metadata failed; retrying in $((attempt * 10))s..." >&2
        sleep $((attempt * 10))
    done
}

manifests=()
while IFS= read -r manifest; do
    manifests+=("$manifest")
done < <(
    cd "$REPO_ROOT"
    git ls-files \
        'crates/*/Cargo.toml' \
        'tools/*/Cargo.toml' \
        | sort -u
)

if [[ "${#manifests[@]}" -eq 0 ]]; then
    echo "no Cargo.toml manifests found under crates/ or tools/" >&2
    exit 1
fi

versioned_files=()

workspace_manifest="$REPO_ROOT/Cargo.toml"
require_file "$workspace_manifest"
previous_version="$(read_workspace_version "$workspace_manifest")"
if [[ -z "$previous_version" ]]; then
    echo "failed to read [workspace.package].version from $workspace_manifest" >&2
    exit 1
fi
update_workspace_version "$workspace_manifest" "$version"
update_versioned_path_dependency_versions "$workspace_manifest" "$version"
versioned_files+=("$workspace_manifest")

for relative_manifest in "${manifests[@]}"; do
    manifest="$REPO_ROOT/$relative_manifest"
    require_file "$manifest"
    update_manifest_version "$manifest" "$version"
    update_versioned_path_dependency_versions "$manifest" "$version"
    versioned_files+=("$manifest")
done

kotlin_build_file="$REPO_ROOT/sdk/kotlin/build.gradle.kts"
require_file "$kotlin_build_file"
update_gradle_project_version "$kotlin_build_file" "$version"
versioned_files+=("$kotlin_build_file")

literal_version_files=(
    "crates/mesh-llm-native-runtime/README.md"
    "crates/mesh-llm-sdk/README.md"
    "crates/mesh-llm-ui/package.json"
    "crates/mesh-llm-ui/package-lock.json"
    "sdk/node/package.json"
    "docs/sdk/node.md"
    "docs/sdk/rust.md"
    "docs/sdk/swift.md"
    "docs/SDK.md"
    "sdk/swift/README.md"
    "sdk/swift/scripts/generate-swift-bindings.sh"
    "website/src/docs/pages/CLI.md"
    "docs/design/NATIVE_RUNTIMES.md"
    "sdk/kotlin/README.md"
    "sdk/kotlin/example/example-jvm/build.gradle.kts"
    "crates/mesh-llm-config/src/model/built_in_schema/presentation.rs"
)

for relative_file in "${literal_version_files[@]}"; do
    file="$REPO_ROOT/$relative_file"
    require_file "$file"
    case "$relative_file" in
        crates/mesh-llm-ui/package.json | crates/mesh-llm-ui/package-lock.json | sdk/node/package.json)
            update_json_package_version_references "$file" "$previous_version" "$version"
            ;;
        *)
            update_literal_version_references "$file" "$previous_version" "$version"
            ;;
    esac
    versioned_files+=("$file")
done

known_versions_file="$REPO_ROOT/crates/mesh-llm-config/src/model/built_in_schema.rs"
require_file "$known_versions_file"
update_known_mesh_versions "$known_versions_file" "$version"
versioned_files+=("$known_versions_file")

echo "Refreshing Cargo.lock workspace package versions..."
refresh_cargo_lock_versions

versioned_files+=("$REPO_ROOT/Cargo.lock")

echo "Updated release version to $version:"
for file in "${versioned_files[@]}"; do
    echo "  $file"
done
