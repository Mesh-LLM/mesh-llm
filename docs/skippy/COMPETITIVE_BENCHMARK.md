# Mesh versus llama.cpp competitive benchmark

`evals/skippy-competitive-benchmark.py` is the reproducible entrypoint for the
cross-platform scheduler benchmark. It compares the release `skippy-server`
OpenAI surface with the repository's pinned raw llama.cpp baseline without
changing serving code during a run.

The checked-in contract covers:

- CUDA and Metal;
- Llama 3.2 1B dense, DeepSeek Coder V2 Lite MoE, and Falcon-H1 recurrent;
- llama-benchy `pp=512`, `tg=8/64/256`, exact output length;
- 256 deterministic prompts derived from the Thoughtworks agentic coding
  trajectories dataset;
- offered concurrency `1/2/4/8/16/32/64/128/256`; and
- raw results, parity evidence, CSV tables, SVG charts, `REPORT.md`, and a
  SHA-256 inventory.

The Thoughtworks subset is pinned to the MIT-licensed
`swe-smith-claude-3-7-sonnet` source. The eight selected trajectory identities,
dataset commit, dataset checksum, and generated manifest checksum live in
`evals/skippy-competitive-benchmark.json`. The exact llama-benchy tokenizer
directory digest for every model is pinned there as well. Corpus, tokenizer,
and model bytes are never checked into the repository.

## Inspect the matrix

`plan` is side-effect free:

```bash
python3 evals/skippy-competitive-benchmark.py plan
```

The full matrix contains 432 arm-level cells. Filters are repeatable:

```bash
python3 evals/skippy-competitive-benchmark.py plan \
  --platform metal \
  --model falcon-h1-recurrent \
  --workload thoughtworks
```

## Acquire pinned inputs

Run `prefetch` outside the timed benchmark window. It invokes `hf download` at
the checked-in revisions, invokes `hf cache verify`, verifies each required
file's SHA-256, regenerates the prompt manifest, then verifies its SHA-256 and
row provenance.

```bash
python3 evals/skippy-competitive-benchmark.py prefetch \
  --model-root /path/to/competitive/models \
  --dataset-root /path/to/competitive/dataset \
  --manifest /path/to/competitive/thoughtworks-256.json
```

The manifest generator requires DuckDB in the selected Python environment.
The model and dataset SHA-256 checks are authoritative for the selectively
downloaded files; `hf cache verify` additionally validates every locally
present Hub file without requiring unrelated quantizations from the same repo.
In the two-machine lab, download from Studio54 with
`HF_HOME=/Volumes/External/models/huggingface`; Micstudio reads the shared NFS
cache and must not download the same input concurrently.

The llama-benchy tokenizer directories are inputs rather than generated report
data. Put the pinned directories under `<tokenizer-root>/<model-key>`; `run`
hashes every relative file and fails before timing if a directory differs from
the checked-in digest.

## Run one hardware platform

Build `skippy-server` in release mode before measuring. Run the same entrypoint
on each hardware host and point both at one artifact root. The runner records
the exact Git heads and SHA-256 values of both binaries and refuses to resume
into a directory created by different binaries or config.

```bash
python3 evals/skippy-competitive-benchmark.py run \
  --platform metal \
  --output /path/to/artifact \
  --model-root /path/to/competitive/models \
  --tokenizer-root /path/to/competitive/tokenizers \
  --manifest /path/to/competitive/thoughtworks-256.json \
  --mesh-root "$PWD" \
  --mesh-binary "$PWD/target/release/skippy-server" \
  --native-dir "$PWD/.deps/llama-build/build-stage-abi-static-metal" \
  --llama-root /path/to/pinned/llama.cpp \
  --llama-binary /path/to/pinned/llama.cpp/build/bin/llama-server \
  --benchy /path/to/llama-benchy
```

Use `--platform cuda` and that host's native runtime directory for the CUDA
run. `--model` and `--workload` may be repeated for a narrowed diagnosis.
Resume is on by default. A cell is skipped only when its completion marker
matches the config, manifest, and relevant binary hashes.

Synthetic cells use four active execution lanes. Thoughtworks cells use two
active lanes and a 16K unified context so the same model/profile fits the
16 GB CUDA acceptance host. Offered concurrency still reaches 256: requests
beyond the active lanes exercise waiting admission, queueing, and scheduler
behavior. The trace arms alternate raw/Mesh and Mesh/raw across the concurrency
ladder to reduce time-order bias.

On Linux CUDA, `--mesh-adaptive` adds a staged Mesh arm whose committed active
generation permit count starts at one. Under sustained queued load it tentatively
probes one additional lane, keeps it only when useful-token retirement improves
without hardware service-time or p95 latency pressure, and otherwise drains back
to the last committed limit. Failed generations force immediate rollback or
backoff, and a cooldown periodically re-probes after workload changes. Use
`--comparison-backend vllm` and/or `--comparison-backend sglang` for optional
external comparisons. vLLM serves the pinned GGUF with the pinned tokenizer
and requires `vllm-gguf-plugin` in the vLLM virtual environment plus the
pinned per-model configs under `--vllm-hf-config-root`.
SGLang also serves the same pinned GGUF and tokenizer by default; an alternate
per-model input may be supplied with `--sglang-model-root` when an architecture
cannot load that GGUF. Missing optional runtimes or SGLang model inputs are recorded in
`comparisons/<platform>/availability.json` and skipped without weakening the
required raw llama.cpp/fixed Mesh matrix.
Pass `--require-comparison-backends` for a controlled comparison that must
produce every requested external-backend number; preflight then fails before
timing if either runtime or any model input is missing. The next controlled
CUDA rerun uses this flag with both vLLM and SGLang.

The independent `nightly-competitive-benchmark.yml` workflow runs trusted
`main` on the approved Linux GPU runner when
`MESH_NIGHTLY_COMPETITIVE_ENABLED=1`. Its `MESH_NIGHTLY_COMPETITIVE_*`
repository variables point at pre-baked native libraries, model/tokenizer
inputs, Thoughtworks manifest, pinned llama.cpp checkout/server, and
llama-benchy executable. The runner installs and downloads nothing during a
timed run. Partial evidence uploads even when a cell fails, and the rendered
summary identifies a promotion candidate only after correctness, full-load
completion, and positive mean throughput deltas over fixed Mesh in both
synthetic and Thoughtworks workloads.

Mesh arms pin `--generation-queue-capacity 256` and
`--generation-admission-timeout-secs 600`; these are benchmark overrides, not
production defaults. Active `--generation-concurrency` remains equal to the
manifest's KV-backed lane count, so waiting requests do not create native/KV
lanes.

Do not time benchmarks while model downloads, builds, or unrelated GPU work are
active. Record the host-isolation decision with the artifact when the hardware
is shared.

## Produce the artifact

After both platforms finish:

```bash
python3 evals/skippy-competitive-benchmark.py report \
  --artifact /path/to/artifact
```

The report phase does not start a server. It writes:

```text
artifact/
  benchmark-config.json
  provenance/<platform>.json
  plans/<platform>.json
  data/<platform>/<model>/<arm>/...
  trace/<platform>/<model>/c-<concurrency>/<arm>/...
  summary/
    synthetic.csv
    thoughtworks.csv
    parity.csv
    charts/*.svg
    REPORT.md
  artifact-sha256.txt
```

Throughput is marked competitive only when the paired c1 deterministic
continuation matches exactly and both arms completed every request in the
cell. Failed parity, HTTP overload, timeouts, or incomplete waves remain in the
artifact but are labeled diagnostic.
