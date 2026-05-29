# Windows helpers

These optional PowerShell helpers wrap a local Windows build of `mesh-llm`.

Build first from the repository root:

```powershell
just build backend=vulkan
```

Start a local server:

```powershell
.\contrib\windows\StartMeshServer.ps1 -Model Qwen2.5-3B-Instruct-Q4_K_M -Device Vulkan1
```

Chat with that server:

```powershell
.\contrib\windows\StartChat.ps1 -Model Qwen2.5-3B-Instruct-Q4_K_M
```

Collect split diagnostics from one or more already-running nodes:

```powershell
.\contrib\windows\CollectSplitDiagnostics.ps1 `
  -Model meshllm/Qwen3-8B-Q4_K_M-layers `
  -ConsoleUrls http://127.0.0.1:3131 `
  -ApiUrls http://127.0.0.1:9337/v1
```

The collector writes a timestamped folder and zip containing redacted
management API payloads, `/v1/models`, GPU/process facts, optional probe
results, and recent `skippy-native.log` tails. It attaches to running nodes; it
does not start or stop mesh processes.

The scripts default to `target\release\mesh-llm.exe` when it exists, otherwise
they fall back to `mesh-llm` on `PATH`.
