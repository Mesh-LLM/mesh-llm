<a id="private-meshes"></a>

# Meshes for Your Own Machines

Publishing a mesh to Nostr makes its discovery information public, whether or
not it has a name. A name helps people find a mesh; it does not restrict access.
Private membership requires enforced admission control. An unpublished mesh is
not automatically private.

This guide separates public discovery, restricted membership, and private
request routing.

Traffic between those Mesh nodes is end-to-end
encrypted by QUIC. If iroh uses a relay, the relay forwards encrypted packets
and cannot read prompts, responses, or split-model activations.

To keep some requests on your own machines while they also serve a shared
mesh, such as the public one, use [private routing](/docs/pages/private-routing/).

## Discovery and admission

A **private mesh** restricts membership through enforced
[admission control](#restrict-membership). Public discovery and private membership
are independent: publication reveals discovery information; admission controls
who may participate.

A mesh created without `--publish` is **unpublished**, not necessarily
access-controlled. Named discovery (`--discover`) and public discovery (`--auto`)
are ways to find a mesh; neither defines who is allowed to participate.

A **default join token** supplies endpoint identity and connection addresses,
not bearer authorization. `--join` and `--join-file` can target published or
unpublished meshes. For controlled private deployments, configure and verify
owner/trust policy rather than relying on the mesh name or token secrecy. Signed
bootstrap tokens bind policy but do not by themselves imply owner allowlisting.
See [ownership and admission control](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/MESHES.md#ownership-and-admission-control).

## Restrict membership

Configure owner allowlisting on each participating node. The default trust policy
is `off`; `require-owned` checks for an owner identity but does not limit admission
to owners you trust. Use `allowlist` with your trusted owner IDs for that boundary.
A discovery name or a join token is not a substitute.

See [ownership and admission control](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/MESHES.md#ownership-and-admission-control)
for owner setup and commands.

## Connect without public discovery

Start a node without `--discover`, `--auto`, or `--publish`, then share its join
token out of band. Recipients use `--join <token>` or `--join-file <path>`.
Configure admission control separately if membership must be restricted.
See the [token-based setup](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/MESHES.md#create-an-unpublished-mesh-and-join-by-token).

## Connect through named public discovery

The following commands publish discovery information to Nostr. Anyone who learns
the name can find the mesh; admission policy determines whether they can join.

### Start the first serving node

```sh
mesh-llm serve --discover=my-mesh --model unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
```

### Add another serving machine

Install Mesh on the second machine, then use the same mesh name:

```sh
mesh-llm serve --discover=my-mesh --model <model-ref>
```

### Join as an API-only client

Use this for a laptop that should send requests but not serve a model:

```sh
mesh-llm client --discover=my-mesh
```

### Check that peers are visible

Open the console:

```text
http://localhost:3131
```

Or check status:

```sh
curl -s http://localhost:3131/api/status | jq .
```

Named meshes are useful for lab machines, office workstations, or a home cluster where you want your own machines to find each other by name.

## Serve one model to independent meshes

To offer one local model to separate meshes, use a memberships manifest:

```sh
mesh-llm serve --gguf /models/model.gguf --memberships-config ./memberships.toml
```

Each membership has its own endpoint, peers, trust settings, and execution/cache
namespace. The process loads the model once and shares its execution capacity.
Repeated `--join` values still identify bootstrap peers for a single mesh.
See [One model across independent meshes](/docs/pages/shared-serving/) for the
manifest, trust defaults, and local lifecycle controls.

## Control an owned node

Mesh membership and private owner-control are separate
boundaries. To manage a model on one remote node you own, read that target's
endpoint token locally and transfer it out of band:

```sh
mesh-llm runtime bootstrap --json
```

From a controlling node authenticated as the same owner:

```sh
mesh-llm runtime load-model \
  --endpoint '<control-endpoint>' --model '<canonical-model-ref>'
mesh-llm runtime ensure-model \
  --endpoint '<control-endpoint>' --model '<canonical-model-ref>'
mesh-llm runtime unload-model \
  --endpoint '<control-endpoint>' --model '<canonical-model-ref>'
mesh-llm runtime drain-model \
  --endpoint '<control-endpoint>' --instance-id '<instance-id>'
```

The endpoint token pins exactly one target. These commands travel over
`mesh-llm-control/2`, require same-owner authentication, and create session-only
intents. They do not persist TOML, use public gossip, or silently fall back to
mesh request streams. Public mixed-version join and routing continue
independently; an older target that lacks owner lifecycle support returns a
typed unsupported result.

See [Runtime Lifecycle](/docs/pages/runtime-lifecycle/#owner-control-lifecycle)
for command and drain semantics.
