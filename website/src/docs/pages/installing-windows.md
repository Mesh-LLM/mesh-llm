---
title: Installing on Windows
---

# Installing on Windows

Install Mesh on every Windows machine that should serve a model or call into a mesh.

## Quick install

Open PowerShell and run:

```powershell
irm https://meshllm.cloud/install.ps1 | iex
```

Open a new terminal after install if the installer added Mesh to your `PATH`.

Check the install:

```powershell
mesh-llm --version
```

## Flavor selection

The installer auto-detects your GPU and selects the best bundle. Supported Windows flavors:

| Flavor | Hardware | Notes |
|---|---|---|
| `cuda` | NVIDIA (all architectures) | Selected when NVIDIA detected |
| `rocm` | AMD GPUs with HIP runtime | Use when HIP runtime is available |
| `vulkan` | Vulkan-capable GPUs | Useful when CUDA/ROCm not available |
| `cpu` | Any Windows x86_64 | Slowest, useful for API-only nodes |

Force a specific flavor:

```powershell
& ([scriptblock]::Create((irm https://meshllm.cloud/install.ps1))) -Flavor cuda
```

## What the installer does

The installer detects your Windows hardware, selects the matching release bundle, downloads the Mesh release, installs the `mesh-llm` binary, and adds `%LOCALAPPDATA%\mesh-llm\bin` to your user `PATH` when needed.

## Advanced install

Install the latest prerelease:

```powershell
& ([scriptblock]::Create((irm https://meshllm.cloud/install.ps1))) -PreRelease
```

Install to a custom location:

```powershell
& ([scriptblock]::Create((irm https://meshllm.cloud/install.ps1))) -InstallDir "$HOME\bin"
```

## Next step

Run the [Quickstart](/docs/pages/quickstart/) to start a private node and open the console.

## See also

- [Installing on macOS](/docs/pages/installing-macos/)
- [Installing on Linux](/docs/pages/installing-linux/)
- [Hardware support](/docs/pages/hardware-support/)
- [Updating Mesh](/docs/pages/updating-mesh/)
