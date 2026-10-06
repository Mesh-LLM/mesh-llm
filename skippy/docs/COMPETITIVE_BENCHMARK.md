# Mesh versus llama.cpp competitive benchmark

The native `automation replay-matrix competitive-plan`, `competitive-run` and
`competitive-report` commands own generic planning, prepared execution and
reporting. Input acquisition/build preparation remains a separate migration
frontdoor; the native run consumes independently prepared pinned inputs. It compares the release `skippy` CLI
OpenAI surface with the repository's pinned raw llama.cpp baseline without
changing serving code during a run.

The checked-in contract covers:

- CUDA and Metal;
- Llama 3.2 1B dense, DeepSeek Coder V2 Lite MoE, Falcon-H1 recurrent,
  and Granite 4.0 H 1B hybrid;
- llama-benchy `pp=512`, `tg=8/64/256`, exact output length;
- 256 deterministic prompts derived from the Thoughtworks agentic coding
  trajectories dataset;
- offered concurrency `1/2/4/8/16/32/64/128/256`; and
- raw results, parity evidence, CSV tables, SVG charts, `REPORT.md`, and a
  SHA-256 inventory.

The Thoughtworks subset is pinned to the MIT-licensed
`swe-smith-claude-3-7-sonnet` source. The eight selected trajectory identities,
dataset commit, dataset checksum, and generated manifest checksum live in
`skippy/skippy/evals/skippy-competitive-benchmark.json`. The exact llama-benchy tokenizer
directory digest for every model is pinned there as well. Corpus, tokenizer,
and model bytes are never checked into the repository.

## Inspect the matrix

Native `competitive-plan` is side-effect free and classifies the required
raw-versus-Mesh matrix without starting a backend. Its schema version 2 binds
`config_sha256` to the exact config file bytes (`source_bytes_sha256`), rather
than the legacy logical JSON hash. The prepared native run/report interfaces are described below; acquisition and
build-to-prepared-input composition remain separate open migration work.

```bash
cargo xtool automation replay-matrix competitive-plan --config skippy/evals/skippy-competitive-benchmark.json
```

The full required raw-versus-Mesh matrix contains 576 arm-level cells. Optional
comparison arms add cells only where their pinned model inputs are available.
Filters are repeatable:

```bash
cargo xtool automation replay-matrix competitive-plan --config skippy/evals/skippy-competitive-benchmark.json \
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
python3 skippy/evals/skippy-competitive-benchmark.py prefetch \
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

`scripts/materialize-competitive-inputs.sh` materializes every pinned input —
GGUF models, tokenizer directories, per-model vLLM `config.json` files, the
Thoughtworks parquet, and the deterministic prompt manifest — from Hugging Face
into the HF cache and verifies each digest before linking a runnable input tree:

```bash
scripts/materialize-competitive-inputs.sh --root /path/to/competitive
```

The Python environment needs `transformers` v5 (tokenizer re-exports) and
`duckdb` (manifest generation). deepseek/falcon tokenizer directories are
transformers v5 re-exports of the pinned `vllm_hf_config` revisions — verified
byte-identical to the checked-in digests — and granite's directory is the
pinned snapshot minus `README.md`; the script header documents each provenance.

Granite's alternate container digest is derived only after the model-specific
value check succeeds. In a Python environment with `gguf`, `numpy`,
`safetensors`, and `torch`:

```bash
python3 skippy/evals/skippy-granite-tensor-equivalence.py \
  --gguf /path/to/granite-4.0-h-1b-bf16.gguf \
  --safetensors /path/to/model.safetensors
```

The verifier covers every source and GGUF tensor, including the official MLP
split, q/k permutation, BF16-to-F32 scalar promotion, and Mamba `A_log`
conversion, then prints the canonical BF16 tensor digest pinned in the config.

## Run one hardware platform

Build the standalone Skippy CLI plus adjacent native runtime through the current
Skippy release product recipes. Prepare the source-pinned raw llama.cpp baseline and
llama-benchy separately, outside measurement. Do not substitute an externally
launched llama-server for Mesh's runtime; the raw executable is a comparison
arm only. Verify the checked-in llama.cpp revision, model/tokenizer/container
pins and the actual backend build/version custody before preparing the input.
A supplied executable/version SHA records a declaration and observation; it
does not authenticate the producer checkout or prove source revision by itself.

Write a prepared input JSON using the closed schema below and absolute paths.
Each artifact is `{ "path": "/absolute/path", "sha256": "64-lowercase-hex" }`.
Hashes must come from the independently verified producer/input inventory; never
change a source pin to make local bytes pass. File artifacts use file-byte SHA;
runtime/tokenizer/comparison artifacts use the existing native sorted tree digest.
HF config is specifically the pinned `config.json` **file**, while its original
parent directory is projected into vLLM's `--hf-config-path`. The supported
materializer symlink is verified by canonical file bytes, not a directory hash.

| Input field | Contract |
| --- | --- |
| `config`, `config_sha256` | Absolute checked-in config and exact raw-byte SHA |
| `platform` | One of `metal`, `cuda`, `rocm`; use a distinct artifact root per platform |
| `models` | Selected configured keys, pinned GGUF artifact and prepared backend map |
| `workloads` | `synthetic`, `thoughtworks`, or both |
| `optional_arms`, `required_comparisons` | Selected `vllm`/`sglang`; required names must also be selected |
| `adaptive` | Adds the `mesh-adaptive` backend without expanding fixed capacity |
| `manifest`, `benchy` | Pinned trace-manifest path and synthetic benchy file artifact when selected |
| `output` | Absolute owned artifact directory |
| `timeout_seconds`, `cell_timeout_seconds`, `request_timeout_seconds` | Absolute matrix/cell/request budgets; matrix at least13s, cell at least10s, request positive and less than cell |
| `resume`, `force` | Mutually exclusive; resume revalidates complete cells, force quarantines prior cell bytes |

Each selected model entry has `key`, `model` (GGUF artifact) and `backends`.
`llama` and `mesh` are required. Each backend has `executable` (file artifact),
`version_sha256` (SHA of trimmed concatenated actual stdout+stderr version bytes),
`cwd` (absolute directory), `runtime`, `tokenizer`, `hf_config`, `comparison_model`
(each nullable artifact), and `match_kv_capacity` (boolean). Mesh requires its
runtime tree. Synthetic requires its configured pinned tokenizer. vLLM/SGLang
require the pinned tokenizer; vLLM additionally requires the config file.
An alternate container requires the exact configured directory and declared
canonical tensor-equivalence pins. Supply every selected model/backend; no
implicit on-demand download/build occurs during the run. Native validation
refuses missing inputs, unexpected fields and changed hashes before launch.

```bash
cargo xtool automation replay-matrix competitive-run \
  --input /path/to/competitive/prepared-metal.json
```

The immutable ladder remains `1/2/4/8/16/32/64/128/256`; each platform root
contains its selected complete model/workload roster. Source/config/prepared
selection, command capacity and completion-bound artifacts are correlated for
resume. Incomplete or mismatched cells refuse resume. Changing selection or
source context requires a new artifact root; `force` preserves old cell evidence
in quarantine rather than admitting it as completed measurement.

Supply `resume: false` for a new run and `resume: true` explicitly to reuse an
existing root. This required input has no implicit default. A cell is skipped
only when completion, config, manifest, consumed artifacts and prepared binary
provenance all correlate; `force` and `resume` cannot both be true.

Synthetic cells use four active execution lanes. Thoughtworks runtime shape is
pinned per family: dense Llama and Granite use 131,072 total KV tokens and 16
live sequences, DeepSeek uses 16,384/8, and Falcon uses 16,384/2. Every backend
inside a family row receives the same context, total KV-token budget, and live
sequence cap. Offered concurrency still reaches 256: requests beyond the active
lanes exercise waiting admission, queueing, and scheduler behavior. The trace
arms alternate raw/Mesh and Mesh/raw across the concurrency ladder to reduce
time-order bias. Reports keep separate tables and charts per family; they do
not compare or aggregate absolute throughput across models.

Set `adaptive: true` and provide a `mesh-adaptive` backend to add a staged Mesh arm whose committed active
generation permit count starts at one. Under sustained queued load it tentatively
probes one additional lane, keeps it only when useful-token retirement improves
without hardware service-time or p95 latency pressure, and otherwise drains back
to the last committed limit. Failed generations force immediate rollback or
backoff, and a cooldown periodically re-probes after workload changes. Select
`vllm` and/or `sglang` in `optional_arms` for optional
external comparisons. vLLM serves the pinned GGUF with the pinned tokenizer by
default and requires `vllm-gguf-plugin` in the vLLM virtual environment plus
the pinned per-model `hf_config` file artifact in its prepared backend. A native
alternate container is supplied as that backend's `comparison_model` tree artifact.
SGLang likewise defaults to the pinned GGUF; its optional per-model alternate
is supplied as its backend's `comparison_model` tree artifact. Alternate containers are accepted only
when their directory digest, source revision, and canonical tensor-equivalence
digest are pinned in the model config. Granite uses the official BF16 GGUF for
raw/Mesh and its value-equivalent official BF16 safetensors for vLLM/SGLang.
Capacity matching accounts for model-specific vLLM cache page geometry:
Granite pins the observed vLLM 0.27.1 ratio of 2,560 cache blocks per 131,072
total tokens, rather than treating its internally enlarged Mamba page as a
generic 16-token attention block.
Unavailable optional engines are recorded in `matrix-plan.json`'s `availability`
field and skipped without weakening required raw/fixed Mesh cells. Supplied
artifacts whose pins fail verification refuse preflight; they are not silently
reclassified as missing. The native owner emits no legacy comparisons directory.
Put every required optional arm in `required_comparisons` for a controlled comparison that must
produce every requested external-backend number; preflight then fails before
timing if either runtime or any model input is missing. A controlled CUDA comparison can require both vLLM and SGLang this way.

Mesh arms pin `--generation-queue-capacity 256` and
`--generation-admission-timeout-secs 600`; these are benchmark overrides, not
production defaults. Active `--generation-concurrency` remains equal to the
manifest's KV-backed lane count, so waiting requests do not create native/KV
lanes.

Do not time benchmarks while model downloads, builds, or unrelated GPU work are
active. Record the host-isolation decision with the artifact when the hardware
is shared.

## Produce the artifact

After a native platform run, reporting is already emitted. To rebuild its report
without launching servers:

```bash
cargo xtool automation replay-matrix competitive-report \
  --artifact /path/to/native/platform-artifact
```

This consumes only the native schema2 plan, exact raw config snapshot and
completion-bound cells; legacy Python archives remain separate historical
artifacts. A partial report omits unmarked cells and says `report_complete=false`;
a marked invalid cell refuses reporting. Root/cell/worker/report output paths
refuse links and special files. Optional/model/platform capability and actual
performance still require executable qualification.

The report phase does not start a server. It writes:

```text
artifact/
  benchmark-config.source.json
  matrix-plan.json
  matrix-results.json
  cells/<model>/<workload>/tg-<tokens>-c-<concurrency>/<arm>/...
  summary/
    synthetic.csv
    thoughtworks.csv
    parity.csv
    charts/*.svg
    REPORT.md
  artifact-sha256.txt
```

Throughput is marked competitive only when the paired per-cell deterministic
continuation matches exactly and both arms completed every request in the
cell. Failed parity, HTTP overload, timeouts, or incomplete waves remain in the
artifact but are labeled diagnostic.

Native synthetic cells own separate retained servers/warmups and deterministic
continuation probes. This changes warmup/cache amortization from the historical
Python sweep, so compare fresh measured baselines. Fixture success establishes
local orchestration only. The local-layout adapter below is implemented in the
native draft pending execution. Acquisition and independent build custody
remain open migration work; schema documentation does not certify them.

## Compose a local prepared layout

The native adapter composes the acquired local layout into the run input:

```bash
cargo xtool automation replay-matrix competitive-prepare \
  --input /path/to/competitive/layout-request.json
cargo xtool automation replay-matrix competitive-run \
  --input /path/to/competitive/prepared-run.json
```

The layout request has these required fields: `config`, `platform`, `model_keys`,
`workloads`, `model_root`, `tokenizer_root`, `mesh_root`, `llama_root`,
`mesh_binary`, `llama_binary`, `native_runtime`, `manifest` (nullable), `benchy`
(nullable), `optional_backends` (map), `required_comparisons`, `adaptive`,
`output`, `prepared_output`, `timeout_seconds`, `cell_timeout_seconds`,
`request_timeout_seconds`, `preparation_timeout_seconds`, `resume` and `force`.
Paths are absolute; output parents must already exist. `prepared_output` must
be a fresh file, while `output` is the eventual owned matrix directory. The
preparation budget is 30..3600 seconds and includes local Git/version processes
and cleanup reserves. No model serving, downloads or builds occur here.

The model layout is `<model_root>/<key>/<configured filename>` and tokenizer
layout `<tokenizer_root>/<key>`. Optional map keys are `vllm`/`sglang`; values
contain `executable`, `cwd`, nullable `hf_config_root`, nullable
`comparison_model_root` and `match_kv_capacity`. Config files are resolved as
`<hf_config_root>/<key>/config.json`; alternate model trees as
`<comparison_model_root>/<key>`. Unsupported optional platform/model arms retain
availability exclusions. A required comparison refuses a missing or
platform-ineligible backend for a configured supported row; explicitly
source-pinned unsupported model rows remain exclusions. Already
materialized root symlinks resolve to their canonical pinned tree; internal
tree safety and source pin checks still apply.

The adapter verifies the configured raw llama.cpp checkout HEAD, observes the
Mesh HEAD and actual bounded complete version output, hashes prepared artifacts
and verifies configured model/tokenizer/config/container pins before writing.
A run rechecks its recorded checkout identities and file/tree pins before launch;
resume binds the source context in its immutable matrix plan. `source_context`
is tagged `local_checkout_and_artifact_observation_no_build_attestation`. A
checkout HEAD beside a binary does not prove that binary was built from it.
Independent release/build custody and real model/platform acceptance remain
required operator evidence. Acquisition/materialization stays with its existing
native HF owners and retained trajectory reader; this adapter does not replace
them with a general download/build engine.
