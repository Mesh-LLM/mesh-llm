# Agent Notes

## Repo Overview

This repo (`mesh-llm`) contains mesh-llm — a Rust binary that pools GPUs over QUIC for distributed LLM inference using llama.cpp.

## Key Docs

| Doc | What it covers |
|---|---|
| `README.md` | Usage, install, CLI flags, examples |
| `CONTRIBUTING.md` | Build from source, dev workflow, UI dev |
| `RELEASE.md` | Release process (build, bundle, tag, GitHub release) |
| `ROADMAP.md` | Future directions |
| `PLAN.md` | Historical design notes and benchmarks |
| `mesh-llm/TODO.md` | Current work items and backlog |
| `mesh-llm/README.md` | Rust crate overview and file map |
| `mesh-llm/docs/DESIGN.md` | Architecture, protocols, features |
| `mesh-llm/docs/TESTING.md` | Test playbook, scenarios, remote deploy |
| `mesh-llm/docs/MoE_PLAN.md` | MoE expert sharding design |
| `mesh-llm/docs/MoE_DEPLOY_DESIGN.md` | MoE auto-deploy UX |
| `mesh-llm/docs/MoE_SPLIT_REPORT.md` | MoE splitting validation results |
| `fly/README.md` | Fly.io deployment (console + API apps) |
| `relay/README.md` | Self-hosted iroh relay on Fly |

## Building

Always use `just`. Never build manually.

```bash
just build    # llama.cpp fork + mesh-llm + UI
just bundle   # portable tarball
just stop     # kill mesh/rpc/llama processes
just test     # quick inference test against :9337
just auto     # build + stop + start with --auto
just ui-dev   # vite dev server with HMR
```

See `CONTRIBUTING.md` for full dev workflow.

## Project Structure

- `mesh-llm/src/` — Rust source
- `mesh-llm/ui/` — React web console (shadcn/ui patterns, see https://ui.shadcn.com/llms.txt)
- `mesh-llm/docs/` — Design and testing docs
- `fly/` — Fly.io deployment (console + API client apps)
- `relay/` — Self-hosted iroh relay
- `evals/` — Benchmarking and evaluation scripts

## Module Structure Rules

The crate root should stay minimal.

- Keep `mesh-llm/src/lib.rs` and `mesh-llm/src/main.rs` as the only root `.rs` files unless there is a strong reason otherwise.
- New code should go into an existing domain directory when possible.

Use semantic ownership for module placement.

- `mesh-llm/src/cli/` — Clap types, command parsing, command dispatch, and user-facing command handlers.
- `mesh-llm/src/runtime/` — top-level process orchestration and startup/runtime coordination.
- `mesh-llm/src/network/` — request routing, proxying, tunneling, relay/discovery networking, request-affinity logic, and endpoint rewrite support.
- `mesh-llm/src/inference/` — model-serving logic, election, launch, pipeline, and MoE behavior.
- `mesh-llm/src/system/` — machine-local environment and platform concerns such as hardware detection, benchmarking, self-update, and local system integration.
- `mesh-llm/src/models/` — model catalog, resolution, downloads, local model storage, and model metadata.
- `mesh-llm/src/mesh/` — peer membership, gossip, identity, peer state, and mesh node behavior.
- `mesh-llm/src/plugin/` — plugin host, plugin runtime, transport, config, and MCP bridge support.
- `mesh-llm/src/api/` — management API surface and route handling.
- `mesh-llm/src/protocol/` — wire protocol types, encoding/decoding, and conversions.

CLI ownership rule.

- All command handlers belong under `mesh-llm/src/cli/`, usually `mesh-llm/src/cli/commands/`.
- Domain modules should not own Clap parsing or top-level command dispatch.
- Domain modules may expose reusable functions that CLI handlers call.

Do not introduce generic buckets.

- Avoid directories or modules named `app`, `utils`, `misc`, `common`, or similar catch-alls.
- Name modules after the responsibility they own.

Keep shared code honest.

- If code is only used by one subsystem, keep it inside that subsystem.
- Only move code to a shared module when it is truly cross-domain.
- Do not create shared helpers prematurely.

Prefer semantic grouping over symmetry.

- Do not create one directory per file just for visual symmetry.
- A single `foo.rs` file is already a Rust module; use a directory only when `foo` has meaningful substructure.

Minimize crate-root re-exports.

- Root re-exports are acceptable as temporary compatibility shims during refactors.
- New code should prefer importing from the owning module directly.
- Remove transitional re-exports once call sites have been updated.

When to split a file.

- Split a file when it contains multiple separable responsibilities, when navigation becomes difficult, or when tests naturally cluster by concern.
- Do not split purely to reduce line count if the code still represents one coherent object or subsystem.

Naming rule.

- File and module names should describe responsibility, not implementation detail.
- Prefer names like `affinity`, `discovery`, `transport`, `maintenance`, `warnings`.
- Avoid vague names like `helpers`, `stuff`, `logic`, or `manager` unless the abstraction is genuinely that broad.

Current structure notes.

- Request-affinity code belongs with networking/routing behavior, not `system/`.
- Plugin MCP support belongs inside `mesh-llm/src/plugin/`, not as a separate root module.
- Model command handlers belong in `mesh-llm/src/cli/commands/`; `mesh-llm/src/models/` should stay domain-focused.

## Key Source Files

- `mesh-llm/src/main.rs` — CLI args, orchestration: `run_auto()`, `run_idle()`, `run_passive()`
- `mesh-llm/src/mesh.rs` — `Node` struct, gossip, mesh_id, peer management
- `mesh-llm/src/election.rs` — Host election, tensor split calculation
- `mesh-llm/src/proxy.rs` — HTTP proxy: request parsing, model routing, response helpers
- `mesh-llm/src/api.rs` — Management API (:3131): `/api/status`, `/api/events`, `/api/discover`, `/api/join`
- `mesh-llm/src/nostr.rs` — Nostr discovery, `score_mesh()`, `smart_auto()`
- `mesh-llm/src/download.rs` — Model catalog (`MODEL_CATALOG`), HuggingFace downloads
- `mesh-llm/src/moe.rs` — MoE detection, expert rankings, split orchestration
- `mesh-llm/src/launch.rs` — llama-server/rpc-server process management

## Plugin Protocol Compatibility

When iterating on the plugin protocol, always consider protocol compatibility.

- If a protocol change may be breaking, explicitly ask the developer whether the change is intended to be breaking.
- If the change is not intended to be breaking, the previous version of the plugin protocol must continue to be supported.
- Do not silently ship plugin protocol changes that strand older plugins or hosts without confirming that outcome is acceptable.

## UI Notes

For changes in `mesh-llm/ui/`, use components and compose interfaces consistently with shadcn/ui patterns. Prefer extending existing primitives in `ui/src/components/ui/` over ad-hoc markup.

## Testing

Read `mesh-llm/docs/TESTING.md` before running tests. It has all test scenarios, remote deploy instructions, and cleanup commands.

## Formatting

Before committing Rust changes, format only the changed Rust files from the repo root, for example with `cargo fmt --all -- path/to/file.rs`, and include those formatting changes in the commit.

## Pull Requests

Pull request titles and descriptions should be user-focused by default.

- Title PRs around the user-visible change or capability, not the implementation detail.
- Start the description with what the user can now do, see, or understand after the change.
- Keep architectural refactors, internal state reshaping, and code-organization notes out of the opening summary unless they directly change user behavior.
- If there are important architectural changes, add a separate `## Architecture` section.
- If there are protocol or compatibility implications, add a separate `## Protocol` section that clearly calls out compatibility, migration, or breaking-change impact.
- If the PR changes CLI behavior or touches user-facing CLI flows, include example commands and representative output in the PR description.
- If the PR changes the UI, include at least one screenshot in the PR description.
- Validation and screenshots should stay separate from the user-facing summary.

### Deploy to Remote

```bash
just bundle
# scp bundle to remote, tar xzf, codesign -s - the three binaries
```

### Cleanup

```bash
pkill -f mesh-llm; pkill -f rpc-server; pkill -f llama-server
```

## Deploy Checklist — MANDATORY

**Every deploy to test machines MUST follow this checklist.**

### Before starting nodes
1. **Bump VERSION** in `main.rs` so you can verify the running binary is new code.
2. `just build && just bundle`
3. Kill ALL processes on ALL nodes — `pkill -9 -f mesh-llm; pkill -9 -f llama-server; pkill -9 -f rpc-server`
4. Verify clean — `ps -eo pid,args | grep -E 'mesh-llm|llama-server|rpc-server' | grep -v grep` must be empty.
5. Deploy bundle — scp + tar + codesign on remote nodes.
6. Verify version — `mesh-llm --version` on every node.

### After starting nodes
7. Verify exactly 1 mesh-llm process per node.
8. Verify child processes (at most 1 rpc-server + 1 llama-server per mesh-llm).
9. `curl -s http://localhost:3131/api/status` returns valid JSON on every node.
10. Check `/api/status` peers for new version string.
11. Verify expected peer count.
12. Test inference through every model in `/v1/models`.
13. Test `/v1/` passthrough on port 3131.

### Common failures
- **nohup over SSH doesn't stick** — use `bash -c "nohup ... & disown"`, verify process survives disconnect.
- **Duplicate processes** — always kill-verify-start.
- **codesign changes the hash** — don't compare local vs codesigned remote.

## Releasing

See `RELEASE.md` for the full process.

Current release flow:

1. Build and verify locally:
   ```bash
   just build
   just bundle
   ```
2. Release from a clean local `main` branch:
   ```bash
   just release v0.X.Y
   ```
   This bumps the version, refreshes `Cargo.lock` without upgrading dependencies, commits as `v0.X.Y: release`, pushes `main`, and then pushes only the new release tag.
3. Pushing a `v*` tag triggers `.github/workflows/release.yml`, which builds the release artifacts on Linux CPU, Linux CUDA, and macOS and creates the GitHub release automatically.

## Credentials

Test machine IPs, SSH details, and passwords are in `~/Documents/private-note.txt` (outside the repo). **Never commit credentials to any tracked file.**

## What NOT to add

- **No `api_key_token` feature** — explicitly rejected, removed in v0.26.0
- **No credentials in tracked files** — IPs, passwords, SSH commands belong in `~/Documents/private-note.txt` only
