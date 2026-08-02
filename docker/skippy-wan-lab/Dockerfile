# syntax=docker/dockerfile:1.7

FROM debian:bookworm-slim AS builder
ARG TARGETARCH
ARG SCCACHE_VERSION=0.16.0

RUN --mount=type=cache,id=mesh-llm-apt-lists-debian-bookworm-${TARGETARCH},target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,id=mesh-llm-apt-archives-debian-bookworm-${TARGETARCH},target=/var/cache/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    git \
    build-essential \
    libdbus-1-dev \
    libssl-dev \
    ninja-build \
    pkg-config \
    python3

COPY scripts/install-sccache.sh /usr/local/bin/install-sccache
RUN --mount=type=cache,id=mesh-llm-downloads-debian-bookworm-sccache-${TARGETARCH},target=/var/cache/mesh-downloads,sharing=locked \
    TARGETARCH="$TARGETARCH" SCCACHE_VERSION="$SCCACHE_VERSION" install-sccache

RUN --mount=type=cache,id=mesh-llm-rustup-stable-debian-bookworm-${TARGETARCH},target=/root/.rustup/downloads,sharing=locked \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
ENV LLAMA_STAGE_BUILD_DIR=".deps/llama.cpp/build-stage-abi-static"
WORKDIR /src

RUN --mount=type=cache,id=mesh-llm-cargo-registry-debian-bookworm-rust-stable-${TARGETARCH},target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,id=mesh-llm-cargo-git-debian-bookworm-rust-stable-${TARGETARCH},target=/root/.cargo/git,sharing=locked \
    cargo install just --locked

COPY . .

RUN --mount=type=cache,id=mesh-llm-sccache-debian-bookworm-gcc12-cpu-${TARGETARCH},target=/root/.cache/sccache,sharing=locked \
    SCCACHE_DIR=/root/.cache/sccache \
    MESH_LLM_REQUIRE_SCCACHE=1 \
    LLAMA_STAGE_USE_SCCACHE=1 \
    just llama-build
RUN --mount=type=cache,id=mesh-llm-cargo-registry-debian-bookworm-rust-stable-${TARGETARCH},target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,id=mesh-llm-cargo-git-debian-bookworm-rust-stable-${TARGETARCH},target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,id=mesh-llm-cargo-target-debian-bookworm-rust-stable-cpu-${TARGETARCH},target=/src/target,sharing=locked \
    just skippy-wan-lab-build-bins && \
    mkdir -p /out && \
    cp /src/target/release/skippy-server /out/skippy-server && \
    cp /src/target/release/skippy-prompt /out/skippy-prompt && \
    cp /src/target/release/metrics-server /out/metrics-server && \
    cp /src/target/release/skippy-model-package /out/skippy-model-package

FROM debian:bookworm-slim AS runtime
ARG TARGETARCH

RUN --mount=type=cache,id=mesh-llm-apt-lists-debian-bookworm-${TARGETARCH},target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,id=mesh-llm-apt-archives-debian-bookworm-${TARGETARCH},target=/var/cache/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    iproute2 \
    iputils-ping \
    jq \
    libdbus-1-3 \
    libgomp1 \
    libssl3 \
    netcat-openbsd \
    python3 \
    tini

COPY --from=builder /out/skippy-server /usr/local/bin/skippy-server
COPY --from=builder /out/skippy-prompt /usr/local/bin/skippy-prompt
COPY --from=builder /out/metrics-server /usr/local/bin/metrics-server
COPY --from=builder /out/skippy-model-package /usr/local/bin/skippy-model-package
COPY docker/skippy-wan-lab/entrypoint.sh /usr/local/bin/skippy-wan-lab

RUN chmod +x /usr/local/bin/skippy-wan-lab

ENV APP_ROLE=stage
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/skippy-wan-lab"]
