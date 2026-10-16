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
ENV SKIPPY_CUDA_ARCHITECTURES=121
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
WORKDIR /workspace
COPY . .
RUN just agentx-build && just automation-bootstrap
ARG SOURCE_COMMIT
LABEL org.opencontainers.image.revision=$SOURCE_COMMIT
# Runtime uses the caller's UID; installed tools and the harness remain readable.
RUN chmod o+x /root && chmod -R a+rX /root/.local/share/uv
ENV PATH="/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV HOME=/tmp
ENTRYPOINT ["/usr/bin/timeout", "--signal=TERM", "--kill-after=30"]
