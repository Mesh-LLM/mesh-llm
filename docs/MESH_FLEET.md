# Fast Mesh-LLM fleet launcher

`./scripts/mesh-fleet.sh` starts a private Mesh-LLM fleet from macOS across a
Mac seed and Windows join/client peers. It uses the Mac seed's invite-token
workflow, streams the token through SSH stdin, and never saves the token in the
fleet environment file.

## Fast default

```bash
cp scripts/mesh-fleet.env.example scripts/mesh-fleet.env
chmod 600 scripts/mesh-fleet.env
./scripts/mesh-fleet.sh plan
./scripts/mesh-fleet.sh start
```

`start` launches all enabled remote peers concurrently right after the seed
returns its token. It does not wait for each remote console/API to become ready.
Use `--wait-ready` only for slower diagnostics or certification flows.

```bash
./scripts/mesh-fleet.sh --wait-ready status
./scripts/mesh-fleet.sh logs win3
./scripts/mesh-fleet.sh stop all
```

## Per-node offload options

Every node receives a generated `offload.toml` with shared defaults plus any
`NODE_<NAME>_*` overrides:

- `GPU_LAYERS`, `DEVICE`, `LLAMA_FLAVOR`, `TENSOR_SPLIT`, `MAX_VRAM`
- `KV_OFFLOAD`, `KV_CACHE_POLICY`, `CACHE_TYPE_K`, `CACHE_TYPE_V`, `MMAP`
- `CTX_SIZE`, `BIND_PORT`, `CONSOLE`, `API_PORT`, `EXTRA_ARGS`

For cross-host layer splitting, all serving nodes must run compatible Mesh-LLM
versions and use one immutable package-backed model reference. Local GGUF files
are appropriate for single-node serving, not package transfer across split
nodes.

The supplied example leaves Windows nodes disabled until the matching `0.71.0`
Windows executable/runtime bundle is present. It also leaves `win2` disabled:
the current `192.168.1.113` identity matches the active `win` builder, so
running both would start competing nodes on the same host unless deliberately
configured with separate resources and ports.

## Two webchat paths on an operator node

Set `NODE_<NAME>_WEB_UI=1` on exactly the operator/builder node. The fleet
launcher then omits `--headless`, so the Windows build serves its embedded UI at
`http://<node>:<console-port>/`. That UI contains both:

1. **Normal Mesh-LLM chat** — the ordinary local `/v1/responses` chat flow.
2. **Flushnet tool webchat** — activated only when a labelled access code is in
   the user message; the UI then routes through the authenticated canonical
   Flushnet Gateway tool loop.

All other mesh workers should remain `WEB_UI=0` so they start with `--headless`
and spend resources on inference rather than the browser console. The Windows
UI-capable build must use `scripts/build-windows.ps1` without
`MESH_LLM_SKIP_UI=1` and without Cargo `--no-default-features`.

## Windows builder: UI-capable artifact

Use the dedicated wrapper for the Windows operator/builder artifact:

```powershell
.\scripts\build-windows-webchat.ps1 -Backend cpu -BuildProfile release
```

It clears `MESH_LLM_SKIP_UI`, uses the normal Cargo default features (which
include `web-ui`), and verifies both compiled UI routes before reporting
success. Do **not** use `--no-default-features` for the operator/builder binary.
