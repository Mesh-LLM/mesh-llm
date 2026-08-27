# Private Meshes

A private mesh is a named group of your own machines. Use the same name on each
machine you want to connect. Traffic between those Mesh nodes is end-to-end
encrypted by QUIC. If iroh uses a relay, the relay forwards encrypted packets
and cannot read prompts, responses, or split-model activations.

## Start the first serving node

```sh
mesh-llm serve --discover my-private-mesh --model unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
```

## Add another serving machine

Install Mesh on the second machine, then use the same mesh name:

```sh
mesh-llm serve --discover my-private-mesh --model <model-ref>
```

## Join as an API-only client

Use this for a laptop that should send requests but not serve a model:

```sh
mesh-llm client --discover my-private-mesh
```

## Check that peers are visible

Open the console:

```text
http://localhost:3131
```

Or check status:

```sh
curl -s http://localhost:3131/api/status | jq .
```

Private meshes are useful for lab machines, office workstations, or a home cluster where you want your own machines to find each other by name.

## Control an owned node

Public/private mesh membership and private owner-control are separate
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
`mesh-llm-control/1`, require same-owner authentication, and create session-only
intents. They do not persist TOML, use public gossip, or silently fall back to
mesh request streams. Public mixed-version join and routing continue
independently; an older target that lacks owner lifecycle support returns a
typed unsupported result.

See [Runtime Lifecycle](/docs/pages/runtime-lifecycle/#owner-control-lifecycle)
for command and drain semantics.
