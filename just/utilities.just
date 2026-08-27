# ── Utilities ──────────────────────────────────────────────────

# Measure repository-local Cargo build-cache usage (80 GiB / 14 day defaults).
[unix]
cache-status max_size="80GiB" max_age="14":
    python3 scripts/manage-build-cache.py status --max-size "{{ max_size }}" --max-age "{{ max_age }}"

# Emit Cargo's effective workspace and artifact-directory metadata as JSON.
[unix]
cache-cargo-metadata:
    @cargo metadata --no-deps --format-version 1

# Remove one package from an explicitly validated Cargo target directory.
[unix]
cache-cargo-clean:
    @test -n "${MESH_LLM_CACHE_TARGET_DIR:-}" || { echo "cache-cargo-clean: MESH_LLM_CACHE_TARGET_DIR is unset; this recipe is driven by scripts/manage-build-cache.py" >&2; exit 1; }
    @test -n "${MESH_LLM_CACHE_PACKAGE:-}" || { echo "cache-cargo-clean: MESH_LLM_CACHE_PACKAGE is unset; this recipe is driven by scripts/manage-build-cache.py" >&2; exit 1; }
    @cargo clean --target-dir "$MESH_LLM_CACHE_TARGET_DIR" -p "$MESH_LLM_CACHE_PACKAGE"

# Preview bounded, oldest-first incremental and package-aware Cargo cleanup.
[unix]
cache-prune-dry-run max_size="80GiB" max_age="14":
    python3 scripts/manage-build-cache.py prune --max-size "{{ max_size }}" --max-age "{{ max_age }}"

# Prune only repository-local Cargo artifacts; refuses during compilation.
[unix]
cache-prune max_size="80GiB" max_age="14":
    python3 scripts/manage-build-cache.py prune --execute --max-size "{{ max_size }}" --max-age "{{ max_age }}"

# Update both tracked llama.cpp pin files from the prepared checkout.
llama-update-pin:
    scripts/update-llama-pin.sh

# Render a Markdown summary for a llama.cpp upstream pin change.
llama-summary old new:
    scripts/summarize-llama-upstream.sh "{{ old }}" "{{ new }}"

# Clean Rust, llama.cpp, and UI build artifacts.
[unix]
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf \
        target \
        .deps/llama.cpp/build-stage-abi-* \
        .deps/llama-build/build-stage-abi-* \
        "{{ ui_dir }}/node_modules" \
        "{{ ui_dir }}/dist"
    echo "Cleaned Rust target, llama.cpp build dirs, and UI artifacts"

[windows]
clean:
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "Remove-Item -Recurse -Force target,'.deps/llama.cpp/build-stage-abi-*','.deps/llama-build/build-stage-abi-*','{{ ui_dir }}/node_modules','{{ ui_dir }}/dist' -ErrorAction SilentlyContinue"
    echo "Cleaned Rust target, llama.cpp build dirs, and UI artifacts"

# Clean UI build artifacts (node_modules, dist). Fixes stale pnpm state.
[unix]
ui-clean:
    cd "{{ ui_dir }}" && rm -rf node_modules dist
    echo "Cleaned UI: node_modules + dist removed"

[windows]
ui-clean:
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location '{{ ui_dir }}'; Remove-Item -Recurse -Force node_modules,dist -ErrorAction SilentlyContinue"
    echo "Cleaned UI: node_modules + dist removed"
# Stop mesh-llm processes
stop:
    pkill -f "mesh-llm" 2>/dev/null || true
    echo "Stopped"

# Quick test inference (works with any running server on 8080 or 8090)
test port="9337":
    curl -s http://localhost:{{ port }}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello! Write a haiku about distributed computing."}],"max_tokens":50}' \
        | python3 -c "import sys,json; d=json.load(sys.stdin); t=d['timings']; print(d['choices'][0]['message'].get('content','')[:200]); print(f\"  prompt: {t['prompt_per_second']:.1f} tok/s  gen: {t['predicted_per_second']:.1f} tok/s ({t['predicted_n']} tok)\")"

# Show the local llama.cpp ABI patch queue
diff:
    ls -1 third_party/llama.cpp/patches

# Build the client-only Docker image
[unix]
docker-build-client tag="mesh-llm:client":
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.client -t {{ tag }} .

[windows]
docker-build-client tag="mesh-llm:client":
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:DOCKER_BUILDKIT='1'; docker build -f docker/Dockerfile.client -t '{{ tag }}' ."

# Run the client console image locally
docker-run-client tag="mesh-llm:client":
    docker run --rm -p 3131:3131 -p 9337:9337 -e APP_MODE=console {{ tag }}
