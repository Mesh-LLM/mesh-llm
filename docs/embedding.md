# Embedding mesh-llm in a Rust application

mesh-llm exposes a small, curated Rust API for running an in-process mesh
node from your own binary. Use it when you want to ship mesh-llm as part of
a larger Rust application rather than shelling out to the `mesh-llm` CLI.

The full surface lives at [`mesh_llm::embed`][embed-module]. Everything
else in the `mesh-llm` crate (and the underlying `mesh-llm-host-runtime`
crate) is implementation detail and may change in any release.

[embed-module]: https://docs.rs/mesh-llm/latest/mesh_llm/embed/index.html

## What's in scope

`NodeBuilder` and `NodeHandle` cover the mesh layer:

- iroh endpoint binding
- peer membership and gossip
- invite-token generation and joining
- relay registration (including [`--relay-auth`][relay-auth-pr] for gated
  iroh relays)
- model advertisement

[relay-auth-pr]: https://github.com/Mesh-LLM/mesh-llm/pull/641

## What's not in scope (yet)

The embed surface deliberately does **not** start:

- the HTTP proxy (`/v1/*` OpenAI-compatible API)
- the management console (`/api/*`)
- the TUI
- local llama.cpp / skippy inference
- auto-discovery / auto-mode loops
- model downloads

If you want the full runtime, including those, run the `mesh-llm` binary as
a subprocess. Open an issue with your use case if you need an embed surface
for any of them.

## Quickstart

```rust,no_run
use std::collections::HashMap;
use mesh_llm::embed::{NodeBuilder, NodeRole, QuicBindSelection};

# async fn run() -> anyhow::Result<()> {
let mut relay_auths = HashMap::new();
relay_auths.insert(
    "https://gated.example/".to_string(),
    "<nip98-bearer-or-static-token>".to_string(),
);

let handle = NodeBuilder::new()
    .role(NodeRole::Client)
    .relays(["https://gated.example/"])
    .relay_auths(relay_auths)
    .bind(QuicBindSelection::default())
    .max_vram_gb(Some(0.0))
    .enumerate_host(true)
    .start()
    .await?;

handle.start_accepting();
handle.set_display_name("my-app".to_string()).await;

let invite = handle.invite_token();
println!("share with peers: {invite}");

// later
drop(handle);
# Ok(())
# }
```

## Stability

`mesh_llm::embed::*` is the stable Rust embed surface. Internal modules
(everything else in the crate) may change without notice. If you find
yourself reaching into internals, please open an issue so we can surface
the missing capability on the embed module.

## Relationship to `mesh-llm-client`

- `mesh-llm-client` is a thin **client-only** crate: it speaks the mesh
  protocol to an existing mesh-llm host and exposes chat / completions /
  responses APIs. It does not run a mesh node.
- `mesh_llm::embed` runs a real mesh node in-process: it joins the gossip
  mesh, advertises models, accepts inbound peers, and can produce invite
  tokens.

Pick `mesh-llm-client` if you only need to *talk to* a mesh; pick
`mesh_llm::embed` if you need to *be part of* the mesh.
