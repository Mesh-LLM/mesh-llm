# Repo Agent Skills

Skills under `.skills/` are auto-picked-up by agents working in this repo.
Each is a focused, current how-to; deeper reference lives in `docs/`.

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

Related docs: `docs/USAGE.md` (install/service/storage), `docs/CLI.md`
(commands and model refs), `docs/MESHES.md` (mesh workflows),
`docs/AGENTS.md` (agent clients), `docs/SKIPPY_SPLITS.md` (big-model splits).

Maintainer-facing skills (skippy internals, patch queues, benchmarks, lab) live
in `.agents/skills/`; plugin-shipped skills install via `mesh-llm skills install`.
