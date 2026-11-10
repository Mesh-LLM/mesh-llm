# Local image safety plugins

`get-images` screens outgoing image-generation text locally with YuFeng XGuard
Reason 0.6B. Setting `screen_received_images = true` instead loads Nemotron 3.5
Content Safety for **both** outgoing text and incoming images. `serve-images`
always uses Nemotron for text before generation and images before release.
Each process owns an embedded Skippy backend, with no listener and no mesh
inference RPC. Safety prompts, model identities and templates cannot be selected
by an inference request.

The host requires these plugins at the image request/release boundaries. Plugin
installation is an honest-binary protection, not proof that a remote permissionless
operator ran a classifier. Classifier errors and unavailable or unrecognized
results fail closed. These general safety models are imperfect; this integration
does not claim validated detection of every form of child sexual exploitation.

## Build and provision

Build the two binaries with `just build-image-guards`; the Cargo package is
`mesh-llm-image-guards`. These are separate plugin packages, not bundled or
automatically installed with the host. They use the same packaged
dynamic Skippy native runtime as the host. `get-images --manifest` and
`serve-images --manifest` emit `plugin-manifest.json`, including the settings
schema required by the installer. No Python or
separate inference server is needed.

Install the plugin before configuring its custom settings. A local release
archive contains a directory named `get-images` (or `serve-images`) with its
executable, `plugin-manifest.json`, and `plugin.toml`. The latter declares the
matching `name` and build `version`. This is the same package layout used by
the repository's `package-openai-exchange-exemplar` recipe.

```sh
mesh-llm plugins install --archive ./get-images.tar.gz --name get-images --version 0.78.0
mesh-llm plugins install --archive ./serve-images.tar.gz --name serve-images --version 0.78.0
```

Choose the role needed by that node. The installer records the schema so the
normal strict configuration validator can check the settings. Example provider
config, using the installed executable:

```toml
[[plugin]]
name = "serve-images"

[plugin.settings]
model_path = "/absolute/path/to/Nemotron-3.5-Content-Safety.Q8_0.gguf"
projector_path = "/absolute/path/to/Nemotron-3.5-Content-Safety.mmproj-f16.gguf"
runtime_selection = "cuda"
n_gpu_layers = -1
ctx_size = 8192
```

For a requester, change the name to `get-images`. The default uses the
YuFeng weights below and no projector. For optional received-image screening set
`screen_received_images = true` and supply the Nemotron weights **and** projector.
Only the selected model is loaded. For GPU inference, explicitly select `cuda`
on Linux with NVIDIA GPUs or `metal` on macOS, and retain `n_gpu_layers = -1`
to request all language-model layers on the GPU. This also requests GPU execution
for Nemotron's vision encoder/projector. An exact GPU runtime ID is also
supported. Without an explicit selection, the resolver recommends a compatible
backend and may choose CPU. Native bundle discovery uses
`MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR` and `MESH_LLM_NATIVE_RUNTIME_CACHE_DIR`, just
like the host; the child verifies and loads its own ABI-compatible library set.
Native runtime acquisition is explicit and never triggers a plugin download.

The installed package's enabled flag controls whether this configuration starts
it. Enable a previously disabled package with `mesh-llm plugins enable serve-images`
(or `get-images`). For development, an explicit `command` may select a different
binary; install its matching schema first. Such an explicit command runs
independently of the installed package's enabled flag. Missing or unavailable
guards refuse image requests even though an optional installed plugin does not
prevent the host itself from starting.

For a branch-local native runtime, build from the same source and start the host
with that bundle root in its environment (inherited by its plugins):

```sh
just release-runtime-build cuda  # Linux NVIDIA; use metal on macOS
MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$PWD/dist/native-runtimes" mesh-llm serve
```

GPU settings request placement; they do not enforce a GPU-only execution policy.
Before deploying, verify the native logs report all language-model layers
offloaded and a GPU backend and compute buffers for the vision encoder/projector,
without a CPU vision fallback. Confirm per-process GPU activity during a benign
text and image smoke call. `status.ready` verifies classifier execution, not GPU
placement. Stop the deployment if these GPU checks fail.

Without `model_path`, the plugin requests its fixed asset recipe through the
host's shared Hugging Face cache. Configure a managed `[[models]]` entry whose
`model` is the upstream model ID, whose `provider` is this plugin name, and whose
`source.repo` and `source.revision` match the catalog below. There is no fallback
to an unapproved repository or revision. For preprovided paths, both file size and
SHA256 must match the same pinned recipe. Model and projector files must remain
under the operator's trusted local control after verification.

Once the host is running, benign smoke calls use the existing local management
API (replace the port for the selected test instance):

```sh
mesh-llm plugins call get-images status --port 3131
mesh-llm plugins call get-images screen_text --port 3131 --input '{"text":"Generate an image of a cat sitting on a blue blanket."}'
mesh-llm plugins call serve-images status --port 3131
```

`status.ready` must be true. The CLI unwraps the plugin operation's structured
JSON result. `screen_image` takes `{"data_url":"data:image/png;base64,..."}`;
send it using `--input -` and stdin to avoid shell argument limits. Only benign
fixtures are needed to verify model execution and malformed-result refusals.

Readiness remains false while downloading, hashing, opening, or failing a
self-test. It becomes true only after an actual benign native text inference,
and, for Nemotron, verification of the native rendered image/policy ordering plus
inference on a generated solid-color image. Resident models
use one execution slot, a bounded queue/context/completion budget, cancellation,
and no disk prompt cache. Oversized text fails rather than being truncated.
Images must be inline PNG, JPEG or WebP, at most 4 MiB encoded, 4096 pixels on
either side and a bounded decoded allocation. This initial slice accepts only
opaque, static images; transparent and animated images are rejected to avoid
differences between the classifier's view and the displayed artifact. Remote
URLs are never fetched. Image decoding runs off the async worker under the same
single execution permit as inference, retained by the decoder on cancellation.

## Host boundaries

The host gates image-declared concrete providers at its OpenAI gateway,
including passive client ingress and authenticated peer ingress. A local request
to a local image provider needs both roles. Image-output virtual models and
pipelines are rejected in this slice; select the concrete model directly.
Requester protection applies through this gateway. Applications that use a
direct transport SDK need their own requester-side checks.

Independent requester image screening also covers peers that advertise text-only
output: an opted-in requester buffers and verifies every generation response.
Default requesters rely on the serving node's image checks. Older hosts do not
enforce these guard requirements; capability discovery does not establish that a
peer uses an updated host or runs the checks. Use independent requester image
screening when that trust is unavailable.

Guarded replies are buffered in full, including SSE, before approved bytes are
released. Explicit blocks return HTTP 403; unavailable guards, malformed results,
timeouts, and framing or resource-limit failures return HTTP 503.

Serving integrations must keep their backend private and return inline image
bytes from memory. They must withhold unapproved images from files, public
artifact URLs, shared caches, and raw logs. The host quarantine controls its own
wire and logging boundaries; backend side effects occur before it sees the
response. No remote execution attestation is implied by capability discovery.

## Pinned artifacts and licenses

Community GGUF conversions are pinned by repository commit and independently
verified by the following SHA256 digests. Replacing these requires a reviewed
catalog change, not request parameters.

| Model | Repository revision | File | SHA256 |
|---|---|---|---|
| YuFeng | `mradermacher/YuFeng-XGuard-Reason-0.6B-GGUF@a457e581bb00997ff1eb1f9ae0bf21488c6a632c` | `YuFeng-XGuard-Reason-0.6B.Q8_0.gguf` (804,751,488 bytes) | `3c0d0f65c71945c2ac6c306aa031b77ee8a65874d30c010ccd0ddbcf51306f23` |
| Nemotron | `mradermacher/Nemotron-3.5-Content-Safety-GGUF@4fcb2495f250aa0dba62d7a3b1a18bc6fb5f002a` | `Nemotron-3.5-Content-Safety.Q8_0.gguf` (4,130,408,096 bytes) | `217c703e74074fa57116fee18eb4531e4c7cbbf1e2ed7d626076004900cb7884` |
| Projector | Same Nemotron revision | `Nemotron-3.5-Content-Safety.mmproj-f16.gguf` (851,252,000 bytes) | `ae8dd1fb405d8acf4ea41bc0172ed72c5de283e0c35b594eb3718d498b2e86d0` |

`assets/yufeng.jinja` is the official `tokenizer_config.json` template from
[`Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B@9016029653fa2994e8efe5f2c007fc2ac172287d`](https://huggingface.co/Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B/tree/9016029653fa2994e8efe5f2c007fc2ac172287d),
licensed Apache-2.0. Its one-token fixed taxonomy is preserved. `ma` (minor abuse
and exploitation), `cm` (corruption of minors) and `pc` (pornographic contraband)
block; all other documented labels allow. The `pc`/`cm` labels are broader than
the target harm and can produce false positives. Unknown labels fail closed.

`assets/nemotron.jinja` derives from the official template at
[`nvidia/Nemotron-3.5-Content-Safety@35645ed3543b7e7ffaed2e788699e57a5051497c`](https://huggingface.co/nvidia/Nemotron-3.5-Content-Safety/tree/35645ed3543b7e7ffaed2e788699e57a5051497c),
under its OpenMDW-1.1 and Gemma terms. Nemotron uses the fixed
`mesh-child-safety.v2` custom policy (`policy.rs`) in binary-rating mode, with
category reporting and thinking disabled. Conflicting safe/S7 output remains
unavailable. Its only template adaptation preserves the upstream image-before-prefix
layout when Skippy flattens content parts into a media marker before Jinja. The
one-image operation uses a fixed caption and a private `<mesh_safety_image>`
marker; requests cannot select this template mode. Text retains the upstream
layout. The strict parser requires complete stop/EOS termination and accepts only
the documented rating shapes; rationale, truncation and malformed/conflicting
output never permit release. Guard responses and logs
contain verdict codes, never model rationale or screened material.
