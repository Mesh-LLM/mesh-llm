# Mesh versus llama.cpp competitive benchmark

The native `automation replay-matrix competitive-plan`, `competitive-run` and
`competitive-report` commands own generic planning, prepared execution and
reporting. Native `competitive-inputs-prefetch` acquires source-pinned inputs,
and `competitive-prepare` composes a local prepared layout. Independent build
custody and real-family benchmark qualification remain separate requirements. It compares the release `skippy` CLI
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
`skippy/evals/skippy-competitive-benchmark.json`. The exact llama-benchy tokenizer
directory digest for every model is pinned there as well. Corpus, tokenizer,
and model bytes are never checked into the repository.

## Inspect the matrix

Native `competitive-plan` is side-effect free and classifies the required
raw-versus-Mesh matrix without starting a backend. Its schema version 2 binds
`config_sha256` to the exact config file bytes (`source_bytes_sha256`), rather
than the legacy logical JSON hash. The native acquisition, local-layout preparation and run/report interfaces are
described below; executable observations do not authenticate a build producer.

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

Run native acquisition outside measurement. It resolves configured model artifacts
at manual cadence, downloads selected immutable files, checks their byte pins,
exports declared tokenizer layouts, and uses the native trajectory reader and
manifest owner for the Thoughtworks selection. It owns a fresh private cache and
output tree; it does not require a Python Transformers or DuckDB environment.

Build the acquisition helper and prepared reader through existing Just recipes:

```bash
just competitive-inputs-helper-build
just ci-automation-contracts
```

The helper is `target/debug/model-package-competitive-inputs`; the second recipe
also builds `target/debug/trajectory-reader` with its Parquet input feature and
runs the automation contracts. Use the same-source xtask from
`just automation-bootstrap`. Verify these executable bytes before invoking them.

Create an acquisition request with the closed schema below. All file paths are
absolute. `output_directory` must be fresh with an existing canonical parent.
`model_keys: []` selects all four configured families; nonempty keys select only
those rows, and unknown or duplicate keys refuse. There is no implicit repinning.

| Field | Contract |
| --- | --- |
| `schema_version` | `1` |
| `config`, `config_sha256` | Checked-in benchmark config path and exact byte SHA |
| `model_manifest`, `model_manifest_sha256` | Absolute source artifact manifest path and exact byte SHA, required by the native prefetch authority phase |
| `model_keys` | Empty for all configured families, or explicit selected keys |
| `output_directory` | Fresh absolute input tree |
| `timeout_seconds`, `maximum_bytes` | Positive budget up to 86400 seconds and total admitted bytes up to 1 TiB |
| `credential_file` | Private credential file path, or null for anonymous access; never a token in argv |
| `skip_dataset`, `skip_tokenizers`, `skip_vllm_configs` | Explicit selection flags, normally false |
| `export_sha256` | Accepted derived tokenizer tree SHA per selected family when exporting |
| `semantic_cases` | Per-family source-tokenizer reference cases for non-Granite exports |

Each semantic case supplies `text`, `add_special_tokens`, `decode_ids`,
`skip_special_tokens`, `expected_ids`, and `expected_decoded_sha256`. These are
independent source reference observations, not values fabricated from the export.
Granite uses the complete pinned snapshot minus README files. Other families use
the native tokenizer engine and retain the exact declared chat-template bytes.
Token ID/decoder cases and byte identity do not qualify chat-template rendering
or claim byte identity with historical Transformers exports. Review the accepted
export pins and real-family behavior before measuring a derived layout.

For a reviewed request at `/absolute/acquisition-request.json`, supply observed
helper and reader byte hashes:

```bash
just competitive-inputs-prefetch /absolute/acquisition-request.json \
  /absolute/repo/target/debug/model-package-competitive-inputs HELPER_SHA256 \
  /absolute/repo/target/debug/trajectory-reader READER_SHA256 \
  /absolute/fresh-acquisition-evidence 3600
```

`prefetch.json` must be request-correlated and `MATERIALIZED` with null error for
the complete acquisition. `MATERIALIZED_SELECTED` records explicitly skipped
groups and is not full-input acceptance. For a readerless request with
`skip_dataset: true`, call the same native command directly and omit `--reader`
and `--reader-sha256`; the remaining helper/request/evidence/budget flags still
apply. Unsupported families, missing pins, dataset/manifest provenance drift,
byte drift, timeout or cancellation refuse while retaining partial evidence.

The acquired layout contains `models/<key>/<configured-basename>`,
`tokenizers/<key>`, `vllm-configs/<key>/config.json`, and
`thoughtworks/manifest.json` for selected groups. Use its
`derived-benchmark-config.json` when exports declare new accepted tokenizer pins;
keep the original config and acquisition receipt as lineage. The native
`competitive-prepare` interface below consumes those local layouts and verifies
file/tree identities before constructing the run request.

Qualify the full acquisition-to-native-manifest composition explicitly with the
same-source xtask executable before retiring the transitional materializer:

```bash
just competitive-inputs-prefetch-fixture /absolute/repo/target/debug/xtask
```

That inert fixture establishes local orchestration, export and reader composition;
it does not establish real model performance or the external HF service behavior.

Granite's alternate container digest is derived only after this optional,
model-specific reference evaluation succeeds. It is not a build, quality or
benchmark-run dependency. Prepare its separate locked Python3.12 environment
explicitly; dependency resolution alone does not qualify model values.

```bash
MESH_RESEARCH_ROOT=/absolute/path/to/mesh-llm-research
uv sync --locked --no-python-downloads --project "$MESH_RESEARCH_ROOT/granite-reference" --python python3.12
"$MESH_RESEARCH_ROOT/granite-reference/.venv/bin/python" -I "$MESH_RESEARCH_ROOT/granite-reference/skippy-granite-tensor-equivalence.py" \
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
produce every configured supported external-backend number. A missing or
platform-ineligible prepared backend for a supported row refuses before timing;
explicit source-pinned unsupported model rows remain declared exclusions. A
controlled CUDA comparison can require both vLLM and SGLang this way.

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
    parity.json
    report.json
    promotion.json
    charts/*.svg
    REPORT.md
  artifact-sha256.txt
```

Throughput is marked competitive only when the paired per-cell deterministic
continuation matches exactly and both arms completed every request in the
cell. Failed parity, HTTP overload, timeouts, or incomplete waves remain in the
artifact but are labeled diagnostic.

`summary/report.json` retains the accepted cell rows, capacity policies, paired
parity gates, source config SHA, requested cell count and `report_complete`.
Its `promotion` field and `summary/promotion.json` contain the same decisions,
grouped separately by hardware platform and model; `REPORT.md` renders them.

An optional or adaptive arm is a promotion candidate only when it and fixed
Mesh both cover the complete configured synthetic concurrency/output-token
cross-product and the complete Thoughtworks concurrency roster. Every compared
cell must be accepted and complete with positive finite throughput. Every
configured synthetic output size must also pass its concurrency-one paired
exact-continuation gate against raw llama.cpp. A filtered or incomplete run
cannot satisfy the full promotion roster merely because all observed rows pass.

For each workload, the reducer averages the per-cell percentage gain over fixed
Mesh. Both means must be strictly positive. Capacity policy must be consistent
within each arm; candidate and fixed Mesh policies remain separately recorded.
Among eligible arms, the highest sum of the two means wins, with lexical arm
order breaking equal-gain ties. Missing cells or gates, failed parity, mixed
within-arm capacity policies, or a nonpositive mean produce an explicit hold;
if no arm is eligible, the group has no winner. This records a candidate from
supplied benchmark evidence and never automatically promotes a serving policy.
It establishes no performance result until an actual qualified run provides the
measurements.

Native synthetic cells own separate retained servers/warmups and deterministic
continuation probes. This changes warmup/cache amortization from the historical
Python sweep, so compare fresh measured baselines. Fixture success establishes
local orchestration only. Native acquisition and the local-layout adapter below
own input and prepared-request composition. Independent build custody and
real-family/tokenizer/platform qualification remain operator requirements;
schema documentation and inert fixtures do not certify those external values.

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
native HF owners and native trajectory reader; this adapter does not replace
them with a general download/build engine.
