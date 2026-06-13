---
title: Installing on Linux
---

# Installing on Linux

Install Mesh on every Linux machine that should serve a model or call into a mesh.

## Quick install

```sh
curl -fsSL https://meshllm.cloud/install.sh | bash
```

Open a new terminal after install if the installer added Mesh to your `PATH`.

Check the install:

```sh
mesh-llm --version
```

## Flavor selection

The installer auto-detects your GPU and selects the best bundle. Supported Linux flavors:

| Flavor | Hardware | Notes |
|---|---|---|
| `cuda` | NVIDIA (all architectures) | Selected when NVIDIA tooling or devices detected |
| `rocm` | AMD GPUs with ROCm/HIP | Use when ROCm is installed |
| `vulkan` | Vulkan-capable GPUs | Useful when CUDA/ROCm not available |
| `cpu` | Any Linux x86_64 or ARM64 | ARM64 bundles are CPU-only |

Force a specific flavor:

```sh
curl -fsSL https://meshllm.cloud/install.sh | bash -s -- --flavor cuda
```

## What the installer does

The installer detects your Linux hardware, selects the matching release bundle, downloads the Mesh release, installs the `mesh-llm` binary, and adds `~/.local/bin` to your user `PATH` when needed.

## Advanced install

Install the latest prerelease:

```sh
curl -fsSL https://meshllm.cloud/install.sh | bash -s -- --pre-release
```

Install to a custom location:

```sh
curl -fsSL https://meshllm.cloud/install.sh | bash -s -- --install-dir "$HOME/bin"
```

## Next step

Run the [Quickstart](/docs/pages/quickstart/) to start a private node and open the console.

## See also

- [Installing on macOS](/docs/pages/installing-macos/)
- [Installing on Windows](/docs/pages/installing-windows/)
- [Hardware support](/docs/pages/hardware-support/)
- [Updating Mesh](/docs/pages/updating-mesh/)
