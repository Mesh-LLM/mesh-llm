# Running Big Models With Skippy Splits

Skippy is Mesh LLM's embedded staged runtime. It lets the mesh run models that
do not fit on one machine by loading package-backed layer stages across
selected peers.

## Mental model

1. The coordinator resolves the requested model or layer package.
2. The topology planner picks peers and contiguous layer ranges.
3. Downstream/final stages load first.
4. Stage 0 becomes routable only after every required stage reports ready.
5. OpenAI clients keep using the normal mesh endpoint at
   `http://localhost:9337/v1`.

If one node can load the full model, Mesh LLM prefers the single-node path.
Splitting is used when the model physically needs a split or when an explicit
split run asks for it.

## Use a published layer package

Layer packages are durable Hugging Face repos with a `model-package.json`
manifest and GGUF fragments. Prefer immutable refs for production runs:

```bash
mesh-llm serve --model hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers@<revision> --split
```

Named or moving refs are useful while testing:

```bash
mesh-llm serve --model hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers --split
mesh-llm serve --model hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers:main --split
```

Other peers join the mesh normally:

```bash
mesh-llm serve --join <token>
```

## Two-node split smoke test

Use the same layer-package model on every serving node. Each node resolves the
package and downloads only the shared artifacts plus the layer files needed for
its assigned stage.

```bash
# node A: starts the private mesh and becomes the coordinator
mesh-llm serve \
  --model meshllm/Qwen3-8B-Q4_K_M-layers \
  --split \
  --max-vram 5 \
  --bind-port 7842 \
  --port 9447 \
  --console 3232

# node B: joins with the token printed by node A
mesh-llm serve \
  --model meshllm/Qwen3-8B-Q4_K_M-layers \
  --split \
  --max-vram 5 \
  --join <token> \
  --bind-port 7843 \
  --port 9447 \
  --console 3232
```

For hosts with more than one network interface, add `--bind-ip <lan-ip>` on
each node so the invite token and gossip advertise the routable address.

Once both stages are ready:

```bash
curl -sS http://127.0.0.1:3232/api/status | jq '{state:.node_state, ready:.llama_ready, peers:(.peers|length), stages:.runtime.stages}'
curl -sS http://127.0.0.1:9447/v1/models | jq '.data[].id'
curl -sS http://127.0.0.1:9447/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer mesh' \
  -d '{"model":"meshllm/Qwen3-8B-Q4_K_M-layers","messages":[{"role":"user","content":"Reply with OK"}],"max_tokens":16}'
```

## Try Inkling Q2 text splits (experimental)

Inkling is available as an immutable layer package for operators who want to
evaluate the text path before it is promoted to the customer support matrix:

```text
meshllm/inkling-UD-Q2_K_XL-layers@9b4b91a7ddd978dd7a01679bc977f6e53777f2c7
```

The package is about 296.5 GiB and contains 66 model layers plus shared and
projector artifacts. Each node materializes only its assigned layer range, but
the participating nodes still need enough aggregate GPU memory and per-host
system memory for the model, KV cache, runtime workspaces, and headroom. Start
every node with the same pinned package and context allocation:

```bash
# first node
mesh-llm serve \
  --model meshllm/inkling-UD-Q2_K_XL-layers@9b4b91a7ddd978dd7a01679bc977f6e53777f2c7 \
  --split \
  --ctx-size 131072 \
  --bind-port 7842

# each additional node
mesh-llm serve \
  --model meshllm/inkling-UD-Q2_K_XL-layers@9b4b91a7ddd978dd7a01679bc977f6e53777f2c7 \
  --split \
  --ctx-size 131072 \
  --bind-port 7842 \
  --join <token>
```

Prefer directly reachable, low-latency UDP paths. If a cloud provider remaps the
container UDP port to a different public port, confirm that the invite advertises
the reachable public endpoint. Iroh owns path selection and may use or upgrade
between relay and direct paths; split admission does not second-guess that choice
with path-kind or RTT gates.

Inkling uses the same fixed raw-f32 activation wire as every other model and a
Q4_0 K/V cache. Historical f16 and q8 failures remain compression research
evidence. The published package has no default speculative strategy,
and live native MTP and multimodal serving are not yet operator claims.

PR #1118 has exercised ordinary all-CUDA Mesh planning on a direct roughly 5 ms
Iroh/QUIC path using one 4 x 96 GB node and one 48 GB node. Automatic placement
produced ranges `0..65 / 65..66`, four lanes, and a 131,072-token allocation;
a short exact-answer request completed at 14.98 generated tokens/s. This is a
runnable research topology, not a recommendation that the highly imbalanced
range is optimal: the 65-layer head reserved about 589 GiB of CUDA host compute
workspace. Do not size a host from package bytes and VRAM alone. Prefer multiple
nearby nodes with enough system-memory headroom for a more balanced plan, and
inspect `GET /api/runtime/stages` before inference.

The same run completed two sequential OpenAI tool loops, each with two native
structured tool calls, two intervening pressure turns, and final recall. Exact
prompt replay restored all 3,531 prompt tokens and the native log scan found no
fatal KV/decode/slot/eviction error. Both overlapping phases missed the full
harness bar: one failed while opening a direct prediction-return sink; the other
completed its tool behavior but reported zero changed-tail cached tokens. The
separate same-prefix phase reported the same cache miss. Treat concurrent
admission and suffix cache reuse as active validation gaps; the sequential tool
and exact-cache results do not waive them.

Use streaming for a cold long-context request. Inkling Q2 prefill at this scale
can exceed the OpenAI frontend's 300-second non-streaming backend deadline; a
non-streaming request then returns HTTP 504 even though native prefill is still
healthy. Streaming establishes the response before prefill and also propagates
client cancellation to the generation worker.

Do not treat the 131,072-token allocation as a completed 128K workload proof.
A 480,000-character repository prompt kept native prefill active for 3,429
seconds without a fatal native-log pattern, but the two SSH-launched Mesh
processes ended together before an SSE data event was delivered. The client saw
an empty HTTP 200 stream with no content or usage. That probe is inconclusive;
an operator-facing long-context claim still requires a completed response with
reported prompt-token usage and correct far-prefix recall.

## Lock node order and layer ranges

Maintainer and benchmark runs can replace automatic placement with an exact,
fail-closed topology. Create the same JSON file on every serving node:

```json
{
  "version": 1,
  "model": "hf://meshllm/example-layers@immutable-revision",
  "manifest_sha256": "<sha256 of model-package.json>",
  "stages": [
    {
      "node": "micstudio.local",
      "layer_start": 0,
      "layer_end": 31
    },
    {
      "node": "studio54-3.local",
      "layer_start": 31,
      "layer_end": 47
    }
  ]
}
```

`node` accepts either a full iroh endpoint id or an advertised hostname. Node
selectors must resolve uniquely among eligible split participants. Ranges are
half-open: `layer_start` is inclusive and `layer_end` is exclusive. They must be
non-empty, contiguous, cover `0..layer_count`, and assign each node once. The
model and manifest digest must match the resolved package.

Pass the lock together with `--split` on every node:

```bash
mesh-llm serve \
  --model hf://meshllm/example-layers@immutable-revision \
  --split \
  --split-topology-lock /path/to/topology-lock.json
```

The normal context, KV-cache, headroom, and VRAM checks still apply. Startup
fails if the locked ranges do not fit. Once running, membership changes do not
replace the stages or collapse the model to a local fallback. If a locked stage
is lost, the topology becomes unavailable and is withdrawn after the normal
stage-loss grace period.

Use `skippy-model-package preflight <package-dir> --verify-sha256` to obtain the
manifest digest for a local package. Confirm the realized assignments through
`GET /api/runtime/stages`.

## Use a local GGUF

Direct GGUFs still work:

```bash
mesh-llm serve --gguf ~/models/model.gguf
```

Internally, direct GGUF serving materializes through the same package-backed
stage machinery as a synthetic single-stage package. That keeps the runtime path
consistent without requiring you to publish a package repository first.

## Check readiness

```bash
curl -s http://localhost:3131/api/status | jq .
curl -s http://localhost:9337/v1/models | jq '.data[].id'
mesh-llm doctor split --model-ref meshllm/Qwen3-8B-Q4_K_M-layers --port 3131
```

The stage runtime status is exposed through the management API and web console.
The OpenAI model list should include the full model id once stage 0 is ready.
The split doctor explains which peers are eligible, which peers were excluded,
and the exact next step when the coordinator sees only itself as a valid split
participant.

For maintainer debugging, add `--output-dir <dir>`. The doctor bundle includes
`split-readiness.json`, management API snapshots for runtime/stage/llama status,
plugin startup/provider/endpoint snapshots, `skippy-diagnostics.json`, and the
active instance's `skippy-native.log` when the local runtime directory can be
matched to the console port.

On Windows, collect a shareable diagnostic bundle from already-running nodes:

```powershell
.\contrib\windows\CollectSplitDiagnostics.ps1 `
  -Model meshllm/Qwen3-8B-Q4_K_M-layers `
  -ConsoleUrls http://127.0.0.1:3131 `
  -ApiUrls http://127.0.0.1:9337/v1
```

## Cache behavior

Mesh sets Skippy materialization under the Mesh LLM cache by default:

```text
<user-cache>/mesh-llm/skippy-stages
```

Layer-package downloads use Skippy's Hugging Face package cache unless
`SKIPPY_HF_PACKAGE_CACHE` overrides it. Materialized stage GGUFs are derived
cache, not the durable package format.

Preview cache cleanup:

```bash
mesh-llm models prune
```

Apply cleanup:

```bash
mesh-llm models prune --yes
```

`models prune` protects active or pinned materialized stages and removes only
eligible derived cache entries.

## Verify a package before rollout

For a brand-new model family or a large sharded GGUF candidate, start with the
[new model onboarding checklist](skippy/NEW_MODEL_ONBOARDING.md) before adding a
support-matrix entry.

Package-only verification checks resolution, artifact integrity, and local stage
materialization:

```bash
mesh-llm models certify hf://meshllm/Qwen3-8B-Q4_K_M-layers --package-only --report-out cert.json
```

Use package-only certification as the rollout preflight for published package
refs. It should fail before a split model becomes routable when package
resolution, manifest shape, artifact size/SHA, missing stage files,
tokenizer/projector sidecars, or local materialization are not clean enough for
serving. For a local package directory, run the package-local preflight first:

```bash
skippy-model-package preflight ./model-package --stages 2 --verify-sha256
```

Runtime verification additionally checks a running OpenAI-compatible endpoint:

```bash
mesh-llm models certify hf://meshllm/Qwen3-8B-Q4_K_M-layers \
  --api-base http://127.0.0.1:9337 \
  --json
```

Runtime certification hits `/v1/models`, `/v1/chat/completions`, and
`/v1/responses` and requires real text-bearing responses.

## Peer artifact transfer

For split runs, a worker may fetch missing package artifacts from the
coordinating mesh node before falling back to normal local/Hugging Face package
resolution. This is not a discovery protocol and does not gossip local package
inventory.

Peer artifact transfer is disabled by default on public meshes. Use it only for
trusted or lab deployments:

```bash
MESH_LLM_ARTIFACT_TRANSFER=trusted mesh-llm serve --model hf://meshllm/<repo>@<revision> --split
MESH_LLM_ARTIFACT_TRANSFER=open mesh-llm serve --model hf://meshllm/<repo>@<revision> --split
```

Only immutable `hf://namespace/repo@revision` package refs are eligible for peer
transfer. Received artifacts are size/SHA-256 verified and installed atomically.

## More details

- [LAYER_PACKAGE_REPOS.md](LAYER_PACKAGE_REPOS.md) explains how to contribute packages.
- [specs/layer-package-repos.md](specs/layer-package-repos.md) is the manifest spec.
- [skippy/FAMILY_STATUS.md](skippy/FAMILY_STATUS.md) lists certified families.
- [skippy/TOPOLOGY_PLANNER.md](skippy/TOPOLOGY_PLANNER.md) documents topology planning, including latency-aware physical stage counts.
