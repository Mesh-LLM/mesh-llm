# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git cmake ninja-build pkg-config \
    libssl-dev libdbus-1-dev clang libclang-dev lld mold python3-dev sccache \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain 1.99.0 --profile minimal
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"
RUN cargo install just --version 1.58.0 --locked
RUN curl -LsSf https://astral.sh/uv/0.13.0/install.sh | sh
RUN apt-get update && apt-get install -y --no-install-recommends patchelf \
    && rm -rf /var/lib/apt/lists/*
# One GB10 per job: omit optional multi-GPU NCCL linkage from both engines.
# NCCL is intentionally outside the native runtime's redistribution policy.
RUN apt-get purge -y libnccl-dev libnccl2
ENV SKIPPY_CUDA_ARCHITECTURES=121
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
WORKDIR /workspace
COPY . .
RUN --mount=type=cache,id=agentx-rust199-registry,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,id=agentx-rust199-git,target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,id=agentx-rust199-target,target=/workspace/target,sharing=locked \
    --mount=type=cache,id=agentx-cuda130-single-gpu-native,target=/workspace/.deps,sharing=locked \
    --mount=type=cache,id=agentx-cuda130-sccache,target=/root/.cache/sccache,sharing=locked \
    just agentx-build && just automation-bootstrap && \
    mkdir -p /prepared/target/debug /prepared/target/release /prepared/.deps/agentx-llama-build && \
    cp target/debug/xtask /prepared/target/debug/ && \
    cp target/release/skippy /prepared/target/release/ && \
    cp -a .deps/agentx-llama-build/bin /prepared/.deps/agentx-llama-build/
RUN mv /prepared/target target && mv /prepared/.deps .deps && rmdir /prepared
ARG SOURCE_COMMIT
LABEL org.opencontainers.image.revision=$SOURCE_COMMIT
# Runtime uses the caller's UID; installed tools and the harness remain readable.
RUN chmod o+x /root && chmod -R a+rX /root/.local/share/uv
ENV PATH="/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV HOME=/tmp
ENTRYPOINT ["/usr/bin/timeout", "--signal=TERM", "--kill-after=30"]
