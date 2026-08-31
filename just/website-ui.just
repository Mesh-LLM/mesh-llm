# Run the UI dev server with Vite HMR, proxying /api to mesh-llm (default: http://127.0.0.1:3131)
ui-dev api="http://127.0.0.1:3131" port="5173":
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{ ui_dir }}"
    MESH_UI_API_ORIGIN="{{ api }}" VITE_API_URL="{{ api }}" pnpm run dev -- --host 0.0.0.0 --port {{ port }}

# Run the UI dev server proxying to the public meshllm.cloud API
ui-dev-public: (ui-dev "https://public.meshllm.cloud")

# Build the public website into docs/ for static hosting.
website-build:
    cd "{{ website_dir }}" && npm run clean && npm run build
    just crate-docs

# Build Rustdoc for the crates published by the release workflow.
crate-docs:
    scripts/build-crate-docs.sh

# Run the public website dev server on port 8765.
website-dev:
    cd "{{ website_dir }}" && npm run dev

# Remove generated public website output while preserving docs/ source markdown.
website-clean:
    cd "{{ website_dir }}" && npm run clean

# Run UI unit tests (vitest)
ui-test:
    cd "{{ ui_dir }}" && pnpm test
