# Repo Agent Skills

Skippy lives in its own repository (the one pinned in the root `Cargo.toml`).
Work on Skippy itself, its llama.cpp patches, staged serving, model packages,
quantization jobs and Skippy benchmarks is done there, with that repository's
guidance. This repository keeps only what MeshLLM owns: the host, its
deployment, the mesh, agents, plugins, CI and releases.

`tools/xtask` (`cargo xtool`, `cargo xtask`) was removed from this repository.
`hf-layer-package-jobs`, `kv-tool-loop-stability` and `release-notes` still
name it; check that a command or path exists before following one.

Skills under `.agents/skills/` are the canonical repo-local skills and are
auto-picked-up by agents working in this repo. Each is a focused how-to, but
not every one is current (see above); deeper reference lives in `mesh/docs/`.

## Operator Workflows

| Skill | Use when |
|---|---|
| [deploy-macos](deploy-macos/SKILL.md) | Install/launch mesh-llm on a macOS node (release install or dev-build bundle, codesign/quarantine, verify serving) |
| [deploy-linux-gpu](deploy-linux-gpu/SKILL.md) | Install/launch mesh-llm on a remote Linux GPU node (Vast.ai/RunPod/self-managed CUDA, supervisor/systemd, verify serving) |
| [deploy-windows](deploy-windows/SKILL.md) | Install/launch mesh-llm on Windows (install.ps1 via `irm \| iex`, flavor selection CUDA/ROCm/Vulkan/CPU, contrib helper scripts, PowerShell gotchas) |
| [mesh-join](mesh-join/SKILL.md) | Create/join/publish meshes: invite tokens, `--auto`, named meshes, client-only nodes, NAT/bind issues, multi-node verification |
| [connect-agents](connect-agents/SKILL.md) | Point Goose/Claude Code/OpenCode/Pi or any OpenAI client at a running mesh; tool-call validation; blackboard |

Ground rules baked into all of these:

- The bundle/release is a **single `mesh-llm` binary** with the embedded staged
  runtime. No `rpc-server`, no `llama-server`, no `.dylib` set.
- mesh-llm downloads models itself — pass `--model <ref>`, never pre-download.
- `--headless` only hides the web UI; it is not a backgrounding mechanism.
- Prefer `mesh-llm stop` over `pkill`.

Related docs: `mesh/docs/USAGE.md` (install/service/storage), `mesh/docs/CLI.md`
(commands and model refs), `mesh/docs/MESHES.md` (mesh workflows),
`mesh/docs/AGENTS.md` (agent clients).

The other directories beside these operator workflows are maintainer-facing
skills for release validation, release notes, benchmarks, CI, plugins,
configuration and telemetry. Plugin-shipped skills install
via `mesh-llm skills install`.
