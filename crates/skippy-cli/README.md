# Skippy CLI

The `skippy` binary owns standalone argument parsing, model preparation and console output. The `skippy-server` library owns service loops and accepts plain Rust options.

Build with `just skippy-build`. Supply a separately packaged native runtime:

```sh
target/debug/skippy --runtime-bundle /path/to/runtime serve-openai --model-path /path/to/model.gguf
```

`serve --config stage.json` and `serve-binary --config stage.json` expose the HTTP and binary worker transports. `example-config` writes one JSON document to stdout without loading a runtime. Diagnostics go to stderr.

SIGINT and SIGTERM request shutdown through the service's existing shutdown future. HTTP listeners use Axum graceful shutdown; successful draining of in-flight model requests is a separate acceptance requirement.

Runtime management uses `--runtime-cache`, then `SKIPPY_NATIVE_RUNTIME_CACHE_DIR`, then the platform cache directory under `skippy/native-runtimes`. `SKIPPY_NATIVE_RUNTIME_BUNDLE_DIR` adds explicit bundle roots. Mesh environment variables do not select standalone runtime storage.

```sh
skippy runtime import /path/to/verified-bundle --dry-run
skippy runtime import /path/to/verified-bundle
skippy runtime list
skippy runtime install --manifest /path/to/runtime-catalog.json
skippy runtime install --manifest-url https://example.org/runtime-catalog.json
skippy runtime import-legacy /path/to/old-cache --dry-run
```

Catalog installation requires exactly one explicit catalog file or URL. It uses the selected Skippy runtime cache and explicit bundle roots; Mesh catalog and discovery environment settings are ignored. Downloaded archives retain the shared installer checksum and compatibility checks.

Import copies verified payloads and leaves the source untouched. Legacy import reports every entry as JSON and returns a nonzero exit status if any entry failed.

For explicit workers, `plan-split` runs the shared topology planner, enforces the release certification roster, and admits every native stage before writing any configs:

```sh
skippy --runtime-bundle /path/to/runtime plan-split \
  --model-path /path/to/model.gguf --model-id local-model \
  --worker 127.0.0.1:9400 --worker 127.0.0.1:9401 --output-dir new-plan
skippy --runtime-bundle /path/to/runtime serve-binary --config new-plan/stage-1.json
skippy --runtime-bundle /path/to/runtime serve-binary --config new-plan/stage-0.json \
  --openai-bind-addr 127.0.0.1:9337
```

Start downstream workers first. Each worker must have the exact verified source files at the paths recorded in its config. For workers on different machines, use their reachable addresses in the plan. The plan directory contains both stage configs and their admission descriptors and must not already exist. The initial direct-GGUF planner defaults to one lane, a 512-token context and CPU execution (`--n-gpu-layers 0`).

Generated stage files are the artifact of record: workers load those configs and do not re-admit the diagnostic `admissions.json`. Regenerate a plan when changing it rather than editing stage files. The diagnostic envelope records `certification: certified` and each stage admission.

The graph configuration identity binds the requested GPU-layer policy, context and lane count; it does not attest the actual selected device. Use a recognizable family name in `--model-id` to enable the topology planner's family-specific rules. A filename-derived default may select only intrinsic topology rules; native admission and certification still apply. Split planning currently accepts GGUF files, including the first shard of a multipart model, but not safetensors directories.

Model downloads use `models --cache-dir`, then `SKIPPY_MODEL_CACHE_DIR`, then the platform cache directory under `skippy/models`. Hub endpoint and token settings remain standard Hugging Face settings. Model commands do not load a native runtime.

```sh
skippy models pull org/repo@revision:Q4_K_M --sha256 EXPECTED_SHA256 --size-bytes EXPECTED_BYTES
skippy models list
```

Pull resolves an immutable Hub revision, downloads the selected artifact's file set and reports each file's SHA-256 and the primary path. Optional size and SHA-256 pins apply to the primary file and are checked even for cache hits. Without an expected digest, a reported digest records downloaded content rather than asserting an independently pinned checksum. Use the returned GGUF path with `serve-openai --model-path` or `plan-split --model-path`.

`skippy models remove org/repo --dry-run` previews removal of **all local revisions** of that repository; omit `--dry-run` to remove them. Other repositories remain untouched and no remote Hub deletion is performed. Stop serving that model and stop external Hub downloads into the same cache before removal. Skippy pull/remove commands serialize their mutations with a cache lock; other Hub clients do not participate in it. Repository-root symlinks are rejected and nested symlinks are unlinked without following their targets.

The standalone default model cache is separate from Mesh's cache; select an existing cache explicitly when sharing is intended. Each downloaded file is rehashed, but sidecars without catalog metadata receive only a measured digest, not independent checksum verification. Removing an absent repository succeeds with status `not-found`.

For an orderly split shutdown, signal stage 0 first and wait for it to exit before stopping downstream workers. Stage 0 stops accepting OpenAI requests and drains existing responses while its worker connections remain available. Stopping a downstream worker first interrupts requests that still depend on it.
