# ── Utilities ──────────────────────────────────────────────────

# Clean Rust and UI build artifacts.
[unix]
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf \
        target \
        "{{ ui_dir }}/node_modules" \
        "{{ ui_dir }}/dist"
    echo "Cleaned Rust target and UI artifacts"

[windows]
clean:
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "Remove-Item -Recurse -Force target,'{{ ui_dir }}/node_modules','{{ ui_dir }}/dist' -ErrorAction SilentlyContinue"
    echo "Cleaned Rust target and UI artifacts"

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
        | jq -r '.choices[0].message.content'

# Build the client-only Docker image
[unix]
docker-build-client tag="mesh-llm:client":
    DOCKER_BUILDKIT=1 docker build -f mesh/deploy/docker/Dockerfile.client -t {{ tag }} .

[windows]
docker-build-client tag="mesh-llm:client":
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:DOCKER_BUILDKIT='1'; docker build -f mesh/deploy/docker/Dockerfile.client -t '{{ tag }}' ."

# Run the client console image locally
docker-run-client tag="mesh-llm:client":
    docker run --rm -p 3131:3131 -p 9337:9337 -e APP_MODE=console {{ tag }}
