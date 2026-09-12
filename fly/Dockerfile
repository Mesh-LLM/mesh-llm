# mesh-llm Fly.io console image
#
# Deploy:
#   fly deploy --config fly/console/fly.toml --dockerfile fly/Dockerfile
#   Set MESH_HEADLESS=1 in fly.toml [env] to run API-only (no web UI)

# syntax=docker/dockerfile:1.7

FROM node:24.20.0-alpine AS ui-builder
ARG TARGETARCH
ARG VITE_API_URL=https://public.meshllm.cloud
ARG VITE_APP_VERSION=
ARG VITE_STORAGE_NAMESPACE=mesh-llm-ui-production
ARG NODE_ENV=production
ENV NODE_ENV=$NODE_ENV
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_STORAGE_NAMESPACE=$VITE_STORAGE_NAMESPACE
WORKDIR /app
COPY crates/mesh-llm-ui/package.json crates/mesh-llm-ui/pnpm-lock.yaml crates/mesh-llm-ui/pnpm-workspace.yaml ./
# Install the pnpm pinned by package.json's packageManager field. An unpinned
# `npm install -g pnpm` resolves to pnpm 12, which self-switches to the pinned
# 10.x and then fails with ERR_PNPM_PNPM_ENGINE_IDENTITY_UNVERIFIABLE because a
# pnpm 10 lockfile does not record the @pnpm/exe integrity that pnpm 12 checks.
RUN --mount=type=cache,id=mesh-llm-npm-alpine-node24-${TARGETARCH},target=/root/.npm,sharing=locked \
    npm install -g "$(node -p 'JSON.parse(require("fs").readFileSync("package.json", "utf8")).packageManager')"
RUN --mount=type=cache,id=mesh-llm-pnpm-store-alpine-node24-${TARGETARCH},target=/root/.local/share/pnpm/store,sharing=locked \
    pnpm install --frozen-lockfile
COPY crates/mesh-llm-ui/ ./
RUN VITE_APP_VERSION="${VITE_APP_VERSION:-$(node -p 'JSON.parse(require("fs").readFileSync("package.json", "utf8")).version')}" pnpm run build

FROM debian:bookworm-slim AS builder
ARG TARGETARCH
ARG SCCACHE_VERSION=0.16.0
RUN --mount=type=cache,id=mesh-llm-apt-lists-debian-bookworm-${TARGETARCH},target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,id=mesh-llm-apt-archives-debian-bookworm-${TARGETARCH},target=/var/cache/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git cmake ninja-build pkg-config libssl-dev libdbus-1-dev python3 build-essential
COPY scripts/install-sccache.sh /usr/local/bin/install-sccache
RUN --mount=type=cache,id=mesh-llm-downloads-debian-bookworm-sccache-${TARGETARCH},target=/var/cache/mesh-downloads,sharing=locked \
    TARGETARCH="$TARGETARCH" SCCACHE_VERSION="$SCCACHE_VERSION" install-sccache
RUN --mount=type=cache,id=mesh-llm-rustup-stable-debian-bookworm-${TARGETARCH},target=/root/.rustup/downloads,sharing=locked \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:$PATH"
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
# Copy the whole crates tree. A hand-maintained per-crate list went stale twice
# (#1241, and again once skippy-package-format, skippy-scheduler, and
# mesh-llm-log-store landed), which breaks `cargo build --locked` at deploy time.
# .dockerignore already drops node_modules, dist, and target, so this costs no
# extra context and invalidates the same downstream layers as the list did.
COPY crates/ crates/
COPY tools/xtask/ tools/xtask/
COPY scripts/prepare-llama.sh scripts/build-llama.sh scripts/
COPY scripts/lib/ scripts/lib/
COPY third_party/llama.cpp/upstream.txt third_party/llama.cpp/upstream.txt
COPY third_party/llama.cpp/patches/ third_party/llama.cpp/patches/
RUN --mount=type=cache,id=mesh-llm-sccache-debian-bookworm-gcc12-cpu-${TARGETARCH},target=/root/.cache/sccache,sharing=locked \
    scripts/prepare-llama.sh pinned && \
    SCCACHE_DIR=/root/.cache/sccache \
    MESH_LLM_REQUIRE_SCCACHE=1 \
    LLAMA_STAGE_USE_SCCACHE=1 \
    scripts/build-llama.sh
COPY --from=ui-builder /app/dist crates/mesh-llm-ui/dist
RUN --mount=type=cache,id=mesh-llm-cargo-registry-debian-bookworm-rust-stable-${TARGETARCH},target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,id=mesh-llm-cargo-git-debian-bookworm-rust-stable-${TARGETARCH},target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,id=mesh-llm-cargo-target-debian-bookworm-rust-stable-cpu-${TARGETARCH},target=/src/target,sharing=locked \
    cargo build --release --locked -p mesh-llm && \
    cp target/release/mesh-llm /usr/local/bin/mesh-llm

FROM debian:bookworm-slim
ARG TARGETARCH
RUN --mount=type=cache,id=mesh-llm-apt-lists-debian-bookworm-${TARGETARCH},target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,id=mesh-llm-apt-archives-debian-bookworm-${TARGETARCH},target=/var/cache/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 libdbus-1-3
COPY --from=builder /usr/local/bin/mesh-llm /usr/local/bin/mesh-llm
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV APP_MODE=console
ENV NODE_ENV=production
EXPOSE 3131 9337
ENTRYPOINT ["/entrypoint.sh"]
