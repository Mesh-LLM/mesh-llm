# Build recipe for Anthropic's OSS Scanner (https://red.anthropic.com/oss-scanner).
#
# Contract:
#   - The repository is the build context and must end up at /src inside the image.
#   - Network access exists ONLY during this build. The security audit afterwards is
#     fully offline, so every crate, tool and test fixture must be fetched here.
#   - The whole build (clone -> finished image) has a 45 minute budget on a
#     16 vCPU / 64 GB / 120 GB machine. Keep it lean: no GPU, no model downloads,
#     no llama.cpp, no npm. Do not add long-running test suites.
#
# The enrolment lives in anthropics/oss-scanner under projects/mesh-llm/.
FROM rust:1-trixie

# The repo's .cargo/config.toml sets rustc-wrapper = "sccache", which is a developer
# convenience and is not installed here; the empty env var overrides it.
ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_TERM_COLOR=never \
    CARGO_BUILD_RUSTC_WRAPPER="" \
    RUST_BACKTRACE=1

# Mirrors the Linux build deps used by .github/workflows/release.yml.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git curl jq lsof \
      build-essential clang lld mold cmake ninja-build pkg-config \
      libssl-dev libdbus-1-dev python3 \
    && rm -rf /var/lib/apt/lists/*

# Scanner contract: the checkout lives inside the image, at /src.
COPY . /src
WORKDIR /src

# Vendor every dependency while the network is still up.
RUN cargo fetch --locked

# Debug build keeps debug info and assertions, which is what the scanner wants.
# mesh/crates/mesh-llm-ui/build.rs falls back to an empty dist when the UI is not built,
# so no Node toolchain is required here.
RUN cargo build --locked -p mesh-llm -p mesh-llm-plugin

# Compile the test binaries so they can be run offline during the audit.
# Keep this to the crates that matter; the full workspace does not fit the budget.
RUN cargo test --locked --no-run -p mesh-llm -p mesh-llm-plugin

# The scanner's agent may use a login shell, which resets PATH and loses cargo.
RUN echo 'export PATH=/usr/local/cargo/bin:$PATH' > /etc/profile.d/10-cargo.sh

# Cheap smoke check only. A failing check must not abort the build.
RUN ./target/debug/mesh-llm --version || echo "WARNING: mesh-llm --version failed"
