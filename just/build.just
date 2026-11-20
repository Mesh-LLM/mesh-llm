[unix]
bootstrap-build-tools:
    @scripts/bootstrap-build-tools

[windows]
bootstrap-build-tools:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap-build-tools.ps1

[private]
[unix]
with-lld *COMMAND:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v sccache >/dev/null 2>&1 || {
        echo "Error: sccache is required; install it and retry." >&2
        exit 1
    }
    scripts/cargo-linker --mesh-probe >/dev/null
    exec {{ COMMAND }}

[private]
[windows]
[script("powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File")]
@with-lld *COMMAND:
    $ErrorActionPreference = 'Stop'
    # Look for sccache.exe on PATH directly: Get-Command, on a name it cannot
    # find, first rebuilds the module analysis cache, which takes seconds on a
    # fresh runner. A malformed PATH entry is skipped.
    $sccache = $env:PATH -split ';' | Where-Object { $_ } | Where-Object { Test-Path -LiteralPath "$_\sccache.exe" -PathType Leaf -ErrorAction SilentlyContinue }
    if (-not $sccache) { throw 'sccache is required; install it and retry.' }
    $LASTEXITCODE = 0
    Invoke-Expression '{{ COMMAND }}'
    exit $LASTEXITCODE


# Build the MeshLLM host and console. Skippy is external: link a checkout with
# scripts/skippy-source.sh and supply its packaged runtime at run time.
[unix]
build:
    @scripts/build-host.sh --profile debug

# Fast local iteration build.
[unix]
build-dev:
    @scripts/build-host.sh --profile dev

[windows]
build:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1

# Fast local iteration build.
[windows]
build-dev:
    @powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -BuildProfile dev

# Build the installable host-private OpenAI lifecycle observer exemplar.
build-openai-exchange-exemplar message_format="human":
    @just with-lld cargo build -p mesh-llm-plugin --example openai-exchange-observer --message-format={{ message_format }}

# Produce a local-install archive, with the same plugin.toml/executable layout as releases.
[unix]
package-openai-exchange-exemplar:
    #!/usr/bin/env bash
    set -euo pipefail
    package_root="$(mktemp -d)"
    trap 'rm -rf "$package_root"' EXIT
    plugin_name=openai-exchange-observer
    just build-openai-exchange-exemplar json > "$package_root/build.json"
    # Cargo reports the artifact built by this invocation, including configured
    # target directories and build target triples. Never fall back to a stale binary.
    plugin_executable="$(jq -er -s 'map(select(.reason == "compiler-artifact" and .target.name == "openai-exchange-observer" and .executable != null)) | last | .executable // error("exemplar executable missing from Cargo artifacts")' "$package_root/build.json")"
    mkdir -p "$package_root/$plugin_name" dist
    cp "$plugin_executable" "$package_root/$plugin_name/$plugin_name"
    printf 'name = "%s"\nversion = "1.0.0"\n' "$plugin_name" > "$package_root/$plugin_name/plugin.toml"
    "$plugin_executable" --manifest > "$package_root/$plugin_name/plugin-manifest.json"
    tar -czf "dist/$plugin_name.tar.gz" -C "$package_root" "$plugin_name"

# Run installed-process lifecycle conformance over loopback HTTP and QUIC.
[unix]
test-openai-exchange-conformance: package-openai-exchange-exemplar
    @just with-lld cargo test -p mesh-llm-plugin --example openai-exchange-observer
    @just with-lld cargo test -p mesh-llm-host-runtime --lib --no-default-features installed_ -- --ignored --test-threads=1
