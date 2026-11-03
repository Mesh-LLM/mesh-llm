# Offline host fixtures

These executables exercise real Mesh host/plugin IPC, peer transport and inference
lifecycle hooks without creating payment wallet keys or contacting a blockchain. Mesh
still creates real local node identities and private invitations. The backend
returns an explicitly labelled canned response and declared fixture token counts.
The wallet uses a shared local SQLite fixture chain and fabricates delivery evidence;
it must never be installed in a funded or production node. None of these binaries
is included in the production plugin archives.

Run `host-smoke --mesh-bin /absolute/path/to/mesh-llm --output-dir /absolute/private/smoke`.
It requires the three fixture executables and `pay-stablecoin` beside `host-smoke`.
The Mesh binary needs its compatible native runtime bundle even for the
model-free on-demand buyer. Use a complete product with adjacent
`native-runtimes/`, or supply a verified matching bundle through inherited
`MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR`. A host-only `just mesh` build is insufficient
on its own. The harness disables runtime downloads; no model weights are needed.
It spawns two actual Mesh nodes and checks initial admission, two decoded requests,
exact settlement and cleared debt using the real stablecoin engine. It sends one
buffered request and one SSE request with terminal usage. The test uses
only local fixture endpoints and sends no transactions. Before starting either
node, it packages the companion plugins and installs the wallet and payment
engine into both isolated stores so strict configuration validation uses their
installed manifests. Only the seller installs the backend; the buyer must route
inference remotely rather than serving it from an enabled local fixture.
Node identity paths are isolated from the operator's normal Mesh identity.
