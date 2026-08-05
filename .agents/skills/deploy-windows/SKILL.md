---
name: deploy-windows
description: Use this skill when installing, deploying, launching, serving, or troubleshooting mesh-llm on a Windows machine — PowerShell install via install.ps1, flavor selection (CUDA/ROCm/Vulkan/CPU), source builds, the contrib helper scripts, and verifying it serves.
metadata:
  short-description: Deploy mesh-llm on a Windows node
---

# deploy-windows

Use this when standing up mesh-llm on Windows. Counterpart to `deploy-macos`
and `deploy-linux-gpu`. Same single-binary embedded-runtime architecture; the
binary is `mesh-llm.exe` and release archives are `.zip`
(`mesh-llm-x86_64-pc-windows-msvc[-<flavor>].zip`).

Related skills/docs:

- `mesh-join` — creating/joining meshes (tokens, NAT, multi-node)
- `connect-agents` — pointing Goose/Claude Code/OpenCode/Pi at a running mesh
- `docs/USAGE.md` — install details; `docs/CLI.md` — full command reference
- `contrib/windows/README.md` — local PowerShell helper scripts

## The one rule that matters most

**mesh-llm resolves and downloads the model itself.** Pass `--model <ref>` and
it fetches the GGUF on first use. Do NOT pre-download with `hf` CLI tools.
(Only `--gguf` takes a local file path you manage yourself.)

## Install (PowerShell)

```powershell
irm https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.ps1 | iex
```

Force a flavor non-interactively:

```powershell
$env:MESH_LLM_INSTALL_FLAVOR = "vulkan"
irm https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.ps1 | iex
```

Facts:

- Flavors: `cuda-blackwell`, `cuda`, `rocm`, `vulkan`, `cpu`. **No Metal on
  Windows.** The installer probes `nvidia-smi` (incl. compute capability for
  Blackwell), ROCm tooling, and `vulkaninfo`, then recommends; when
  input/output is redirected (scripted/SSH) it takes the recommendation
  without prompting.
- Installs to `%LOCALAPPDATA%\mesh-llm\bin` (override:
  `-InstallDir` / `MESH_LLM_INSTALL_DIR`) and prepends it to the **user**
  `Path` unless `-NoPathUpdate`. Open a new shell, or use the full path, after
  install.
- CUDA bundles ship their CUDA DLLs alongside `mesh-llm.exe` — no system CUDA
  toolkit install is required to run.
- Other knobs: `-PreRelease` / `MESH_LLM_INSTALL_PRERELEASE=1`,
  `MESH_LLM_REQUIRE_CHECKSUM=1` (makes a missing `.sha256` sidecar fatal;
  default is warn-and-continue).

### `irm | iex` gotchas (learned the hard way — PR #828)

- A missing checksum sidecar on older releases **warns and continues**; that
  is expected, not a failure.
- On Windows PowerShell 5.1, network errors during sidecar download can
  surface as vague response-less `WebException`s rather than clean 404s. The
  installer handles this; if you're debugging a fork/older script, know that
  `iex` also breaks `[ValidateSet]` params (param init to `""` fails
  validation before the script body runs). When `irm | iex` misbehaves,
  fall back to downloading the script and running it as a file:
  `irm <url> -OutFile install.ps1; .\install.ps1 -Flavor vulkan`.

## Build from source (dev)

From a repo checkout (needs Rust, CMake + MSVC, Node, `just`):

```powershell
just build                 # auto-detects cuda / rocm / vulkan / cpu
just build backend=vulkan  # override backend
```

Output: `target\release\mesh-llm.exe` (for `just release-build`) or
`target\debug\` for `just build`. Windows release archives use the dedicated
`release-build-*-windows` / `release-bundle-*-windows` recipes. On native
Windows, `just check-release` skips the Bash-only parity checks.

## Launch

```powershell
mesh-llm serve --model unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL --auto
```

- Same surface as other platforms: `--auto` joins the public mesh; bare
  `serve --model` creates a private mesh and prints an invite token
  (see `mesh-join`). API on `:9337`, console on `:3131`.
- Pin a specific GPU with `--device` (e.g. `--device Vulkan1`,
  `--device cuda:0`); list devices with `mesh-llm gpus`.
- There is **no service install on Windows** (no launchd/systemd equivalent
  in `install.ps1`). Run it in a terminal, or wrap it yourself (Task
  Scheduler / NSSM) — startup models go in `%USERPROFILE%\.mesh-llm\config.toml`
  and bare `mesh-llm serve` reads them.

### Repo helper scripts (dev checkouts)

`contrib\windows\` wraps a local build (falls back to `mesh-llm` on `Path`):

```powershell
.\contrib\windows\StartMeshServer.ps1 -Model Qwen2.5-3B-Instruct-Q4_K_M -Device Vulkan1
.\contrib\windows\StartChat.ps1 -Model Qwen2.5-3B-Instruct-Q4_K_M
.\contrib\windows\CollectSplitDiagnostics.ps1 -Model <ref> -ConsoleUrls http://127.0.0.1:3131 -ApiUrls http://127.0.0.1:9337/v1
```

The diagnostics collector attaches to running nodes and zips redacted API
payloads, GPU/process facts, and `skippy-native.log` tails — useful when
filing split/runtime issues from a Windows box.

## Verify

```powershell
mesh-llm --version
curl.exe -s http://localhost:9337/v1/models
curl.exe -s http://localhost:3131/api/status
curl.exe -s http://localhost:9337/v1/chat/completions -H "Content-Type: application/json" -d '{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}'
```

Use `curl.exe` explicitly — bare `curl` in PowerShell aliases to
`Invoke-WebRequest` with different argument semantics. Model load takes time;
poll `/v1/models` before concluding failure.

## Logs, state, stop

- Runtime/instance state: `%USERPROFILE%\.mesh-llm\runtime\<pid>\` — embedded
  native logs at `logs\skippy-native.log`.
- Config: `%USERPROFILE%\.mesh-llm\config.toml`; identity: `.mesh-llm\key`.
- Stop: `mesh-llm stop` (preferred). Emergency:
  `Stop-Process -Name mesh-llm -Force`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `irm \| iex` dies before any output | Old script / `[ValidateSet]`-style param bug | Update; or download script to a file and run it |
| "could not download checksum sidecar" hard failure | Old installer + release without `.sha256` | Update installer; missing sidecar should warn-and-continue |
| `mesh-llm` not found after install | New `Path` not in current shell | Open a new terminal or use `%LOCALAPPDATA%\mesh-llm\bin\mesh-llm.exe` |
| CUDA flavor won't start on new GPUs | Blackwell needs its own bundle | Install `cuda-blackwell` flavor (or let detection pick it) |
| GPU not used | Wrong flavor or device | `mesh-llm gpus`; reinstall correct flavor; `--device <id>` |
| curl JSON errors in PowerShell | `curl` is an IWR alias | Use `curl.exe`, or `Invoke-RestMethod` |
| Empty `/v1/models` | Model still downloading/loading | Wait; check `skippy-native.log` |
