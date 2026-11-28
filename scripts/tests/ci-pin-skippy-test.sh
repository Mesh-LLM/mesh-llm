#!/usr/bin/env bash
# Exercise manifest rewriting in a disposable checkout, never the real pin.
set -euo pipefail
scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/scripts" "$work/.cargo"
cp "$scripts/ci-pin-skippy.sh" "$work/scripts/"
# Keep the fixture independent of the product's moving release version.
cat >"$work/scripts/skippy-pin.sh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' 'https://github.com/Mesh-LLM/skippy 1111111111111111111111111111111111111111 0.80.3'
EOF
chmod +x "$work/scripts/skippy-pin.sh"
cat >"$work/Cargo.toml" <<'EOF'
[workspace.dependencies]
skippy-ffi = { version = "=0.80.3", default-features = false, features = ["dynamic-runtime"] }
hf_hub = { package = "skippy-hf-hub", version = "=0.80.3" }
other = { git = "https://github.com/other/repo", rev = "1111111111111111111111111111111111111111", version = "1" }
EOF
cp "$work/Cargo.toml" "$work/original"
sha=2222222222222222222222222222222222222222
"$work/scripts/ci-pin-skippy.sh" "$sha"
[[ "$(grep -c "rev = \"$sha\"" "$work/Cargo.toml")" -eq 2 ]]
if grep -q 'version = "=0.80.3"' "$work/Cargo.toml"; then
    echo 'Old version constraint survived' >&2
    exit 1
fi
grep -Fq 'default-features = false, features = ["dynamic-runtime"]' "$work/Cargo.toml"
grep -Fq 'package = "skippy-hf-hub"' "$work/Cargo.toml"
[[ "$(grep '^other =' "$work/Cargo.toml")" == "$(grep '^other =' "$work/original")" ]]
cp "$work/Cargo.toml" "$work/expected"
"$work/scripts/ci-pin-skippy.sh" "$sha"
cmp "$work/Cargo.toml" "$work/expected"
reject() {
    if "$work/scripts/ci-pin-skippy.sh" "$1" >/dev/null 2>&1; then
        echo "Unexpectedly accepted $1" >&2
        exit 1
    fi
    cmp "$work/Cargo.toml" "$work/expected"
}
reject main
reject ''
reject 'a;b'
touch "$work/.cargo/skippy-local.toml"
reject "$sha"
rm "$work/.cargo/skippy-local.toml"
echo '[workspace]' >"$work/Cargo.toml"
cp "$work/Cargo.toml" "$work/expected"
reject "$sha"
echo 'ci-pin-skippy: ok'
