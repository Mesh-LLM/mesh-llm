# Full Linux build test: mesh-llm + llama.cpp (CPU/RPC)
# Run from repo root: docker build -f ci/linux-test.dockerfile -t mesh-llm-ci .
#
# NOTE: npm ci may fail behind SSL-intercepting proxies. If so, pre-build the
# UI on the host (npm run build in mesh/crates/mesh-llm-ui/) — the dist/ is COPY'd in.
FROM rust:latest

RUN apt-get update && apt-get install -y cmake pkg-config git && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Clone llama.cpp fork (not in docker context due to .dockerignore)
RUN git clone -b rebase-upstream-master --depth 1 https://github.com/michaelneale/llama.cpp.git

# Build llama.cpp (CPU + RPC, no GPU)
RUN cmake -B llama.cpp/build -S llama.cpp \
    -DGGML_RPC=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_OPENSSL=OFF \
    && cmake --build llama.cpp/build --config Release -j$(nproc)

# Build mesh-llm (UI already built on host via npm run build, dist/ included)
COPY Cargo.toml Cargo.lock ./
COPY mesh/crates/ mesh/crates/
COPY skippy/crates/ skippy/crates/
COPY tools/xtask/ tools/xtask/
RUN cargo build --release -p mesh-llm
RUN cargo test -p mesh-llm

# Verify all binaries
RUN ls -lh target/release/mesh-llm llama.cpp/build/bin/llama-server llama.cpp/build/bin/rpc-server
RUN target/release/mesh-llm --version
RUN target/release/mesh-llm --help | head -5
RUN llama.cpp/build/bin/llama-server --version
