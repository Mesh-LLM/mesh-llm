# skippy-api-rs

Rust bindings to `../skippy_api`, developed side by side with it.

| crate        | what it is                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `skippy-sys` | the cxx bridge (`src/lib.rs`), the C++ facade it binds (`cpp/bridge.*`), and the build (`build.rs`) |
| `skippy`     | the safe API: `Runtime`, `Model<'rt>`, `Session<'m>`, `Token`, `Result` types |

## Building

`build.rs` runs CMake on `skippy_api` with `SKIPPY_BUILD_LLAMA=ON`, then
compiles the facade with the generated cxx glue. Inputs, all optional:

```
SKIPPY_API_DIR            default ../../skippy_api (relative to skippy-sys)
SKIPPY_LLAMA_SOURCE_DIR   nearest ancestor .deps/llama.cpp above skippy-sys
SKIPPY_CMAKE_PROFILE      default Release
```

The facade, llama and ggml link statically from Cargo's `OUT_DIR`, including
their system-library dependencies. Consumers need no llama loader paths or
consumer build script. Keep this isolated from the production dynamic runtime:
linking both closures into one executable would introduce two llama copies.

## Testing

```
cargo test                                  # no model needed for the unit tests
SKIPPY_TEST_MODEL=/path/to/embedding.gguf \
SKIPPY_TEST_RERANKER=/path/to/reranker.gguf cargo test
```

Or through the workspace recipe, which passes both paths and turns a missing
model into a failure rather than a skip:

```
just skippy-api-rs-e2e /path/to/embedding.gguf /path/to/reranker.gguf
```

The C++ side has no llama-dependent tests by design; the Rust suite is where
backend coverage lives, in two layers:

- `skippy/tests/integration.rs` — the public API end to end: classification,
  embedding, tokenization, sessions, threads, all three workloads.
- `skippy/tests/end_to_end.rs` — the contracts the C++ suite pins with mocks,
  run against a real GGUF: pooling selection, buffer bounds, batch row layout,
  session geometry, prompt token counts, and pool reuse. Each test names the
  C++ behaviour it replaces.

Tests skip when their model is unset, so a checkout without GGUFs stays green.
`SKIPPY_TEST_REQUIRE_MODELS=1` makes a missing model panic instead; the recipe
sets it whenever both paths are given.
