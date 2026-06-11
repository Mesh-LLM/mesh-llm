---
title: Installing on macOS
---

# Installing on macOS

Install Mesh on every Mac that should serve a model or call into a mesh.

## Quick install

```sh
curl -fsSL https://meshllm.cloud/install.sh | bash
```

Open a new terminal after install if the installer added Mesh to your `PATH`.

Check the install:

```sh
mesh-llm --version
```

## Homebrew

If you use Homebrew:

```sh
brew install mesh-llm/tap/mesh-llm
```

## What the installer does

The installer detects your Mac hardware, selects the `metal` bundle (optimized for Apple Silicon), downloads the matching Mesh release, installs the `mesh-llm` binary, and adds `~/.local/bin` to your user `PATH` when needed.

## Force a flavor

The macOS installer auto-detects `metal`. Force a different flavor when auto-detection is wrong or you want to test a specific backend:

```sh
curl -fsSL https://meshllm.cloud/install.sh | bash -s -- --flavor vulkan
```

Available macOS flavors:

| Flavor | Use case |
|---|---|
| `metal` | Apple Silicon (default, recommended) |
| `vulkan` | Vulkan-capable GPUs via MoltenVK |
| `cpu` | CPU-only (slowest, useful for API-only nodes) |

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

- [Installing on Linux](/docs/pages/installing-linux/)
- [Installing on Windows](/docs/pages/installing-windows/)
- [Hardware support](/docs/pages/hardware-support/)
- [Updating Mesh](/docs/pages/updating-mesh/)
