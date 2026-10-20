# Experimental Mesh WASM peer probe (v0)

This is an **artifact-only research component**, not peer attestation. It reuses
Mesh's `mesh-llm-plugin` IPC runtime (`plugin` subcommand). The actual bundled
`controller.wasm` is compiled from `controller.wat` at build time and executed
by Wasmi. Its only imports are `probe.next_action` and `probe.run_tool`; it has
no WASI, filesystem, shell, or network imports. The native plugin broker owns
all observations and model calls. A malicious B host can forge a coherent
broker, run a clean decoy, or proxy the exchange, and still obtain a positive
artifact-only result.

## Source intents and rounds

The default is **one intent per round, at most two rounds**. Round 1 asks for the
probe component's process and an opaque executable handle. After inspecting
that response, A chooses round 2 from a small set: an executable sample for
most seeds, model metadata for every third seed, or model metadata when the
process observation was unavailable. Round 2 has a fresh nonce and carries
only the necessary prior handle, not the whole earlier chat. The optional
`bundled` mode asks process plus the seed's second intent in one round on the
same fixture and sample range. It measures the one-intent versus two-intent
small-model trade-off. These are research comparisons, not production peer
transport.

The structured obligations are authoritative even if A's model rewrites the
question or falls back to a template. A records its reference bytes separately;
no expected digest is sent to B.

| Intent | Question obligation | B tool | A check |
| --- | --- | --- | --- |
| `P` | Identify the **probe component process**, PID, start time, executable handle, source. | `process` | Require a coherent process record; this remains B-reported. |
| `E` | Sample the prior executable handle at `offset` for `length` bytes using the round-2 nonce. | `executable` | Recompute nonce-bound SHA-256 against A-pinned bytes and check PID, start-time and handle linkage; a changed process is explicitly unavailable. |
| `M` | Report active model metadata or explicit unsupported status. | `model` | Compare only if A has independent metadata. The local broker has no active-model linkage and returns unsupported. |

For example, `experiment --seed 0 --broker fixture` produces questions of
this form (IDs, range, nonce, and values are chosen deterministically from the
seed, so see the command output for their exact values). Repeating a seed
reuses its nonce; this deterministic evaluation mode is **not** a production
freshness source:

> Round 1: Identify this probe component's process and executable handle.
>
> Round 2: For executable `fixture-exe-1`, sample offset `...` for 64 bytes
> using the fresh nonce `...`. Say if it is no longer the same process.

The native planner accepts only strict JSON actions naming one tool from that
round's menu, `finish`, or explicit `unavailable`. One malformed or disallowed
action can be retried. Exhausted retries yield `invalid_model_output`. The WASM
guest loops over the model-selected actions and bounded native host calls. B
returns the ordered host-recorded tool records and answer. A one-intent round
ends with a host-generated trace summary after its chosen observation, so a
small model does not need a redundant second `finish` choice; bundled rounds
still require model-selected actions until completion. A's model appraises
the answer and trace, but code decides binding, ranges, reference matches, and
final outcome. A matching executable sample is at most `partial_consistent`,
never full serving-chain validation.

The sample formula is
`SHA-256("mesh-probe-range-v1" || 0x00 || nonce[32] || u64_be(offset) || u32_be(length) || bytes)`.
Offsets are zero-based. B gets neither the reference bytes nor expected digest.

## Run locally

From the repository root, with Rust on `PATH`:

```sh
just with-lld cargo run -p mesh-llm-peer-probe -- experiment --seed 0 --broker fixture
just with-lld cargo run -p mesh-llm-peer-probe -- experiment --seed 0 --broker local
just with-lld cargo run -p mesh-llm-peer-probe -- experiment --seed 0 --broker fixture --mode bundled
just with-lld cargo run -p mesh-llm-peer-probe -- plugin --manifest
```

`fixture` is simulated. `local` samples the **probe executable itself**, not
the Mesh serving process, and A reads the same local file for this loopback
experiment. Both use loopback transport, **without peer authentication**. The
plugin subcommand exposes `agentic_probe_v0` on Mesh's existing plugin IPC,
but no peer challenge/response routing or serving-exchange binding is wired.
Install/enable it only as an opt-in experimental plugin; do not treat its
manifest as a shipped Mesh feature.

`plugin --manifest` prints `{}` because Mesh's package manifest carries static
config and permissions; the operation is declared in the runtime manifest
sent during plugin initialization. A package test checks that runtime entry.

The default model is a scripted test double. For a live model, supply both
`--endpoint http://127.0.0.1:9447/v1` and `--model-id <id>` to `experiment` or
`plugin`. The endpoint must be loopback and `/models` must mark that ID `local`.
This is a client-side routing check, not cryptographic proof that the model
really ran locally or that the service will not proxy it.

For an operator-run, OpenAI-compatible **direct local** model server whose
`/models` list has no Mesh `local` marker, add `--unverified-loopback` to
`experiment` or `evaluate`. The server must still be on loopback and advertise
the selected ID. This explicit mode labels every transcript
`operator_supplied_loopback_unverified`; the operator must establish that the
server uses local weights and does not proxy paid or remote inference. The
`plugin` command keeps the Mesh-local marker requirement.

## Evaluation

```sh
just with-lld cargo run -p mesh-llm-peer-probe -- evaluate \
  --endpoint http://127.0.0.1:9447/v1 \
  --small '<local-small-ID>' --medium '<local-medium-ID>' --larger '<local-larger-ID>' \
  --seeds 0,1,2 --repeat 2
```

For a direct local server, append `--unverified-loopback` and use its listed
model IDs. A small first pass is `--seeds 0 --repeat 1`; it still runs all six
broker scenarios in both challenge modes for each supplied size.

Omit a size that is unavailable: it is reported `untested_missing_model`, not
substituted. Each supplied model receives the same seeds, six broker scenarios,
both challenge modes and budgets across repeats. Output includes complete
round transcripts, task completion, unsupported observations, judge-reported
unsupported claims, outcome confusion matrices, latency, model/tool calls, and
a direct-tool fixed baseline per transcript. The second intent is sampled from
the seed, though sequential mode can pivot to model metadata after an
unavailable process observation; the transcript shows this difference. This is
opt-in evaluation, not deterministic CI. Fixture
`clean_decoy_or_forged_broker` deliberately presents coherent clean
observations despite a deceptive ground truth; a positive appraisal there
demonstrates the trust ceiling.

The defaults cap each B round at 6 tool calls, 8 planner calls, 16 KiB total
planner input/output plus trace, 4 KiB per model output, 120 seconds wall time, and 10,000 WASM fuel
units. The verifier also rejects more than 6 total observed tool calls. Native
model HTTP calls have a 35-second timeout. A's question and judge outputs are
also byte-limited. The WASM guest has no linear memory; its imports cannot
receive paths or arbitrary tool arguments.

## Verify

```sh
ulimit -n 8192
just with-lld cargo test -p mesh-llm-peer-probe
just with-lld cargo test -p mesh-llm-plugin
just with-lld cargo fmt --all -- --check
```

Limitations: observations originate from B's broker; no independent
OS-trusted measurement, B peer signature, challenge transport, serving request
receipt, model-file handle, or release-reference distribution exists here.
Only the A-side loopback runner compares an executable sample with bytes it
holds. A later peer protocol needs authenticated binding and independent
references before any wider assurance claim.
