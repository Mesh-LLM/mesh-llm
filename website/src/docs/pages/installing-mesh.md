---
title: Installing Mesh
---

# Installing Mesh

Mesh runs on macOS, Linux, and Windows. Choose your platform for detailed install instructions.

## Choose your platform

- [Installing on macOS](/docs/pages/installing-macos/) &mdash; Apple Silicon (Metal), Homebrew, flavors
- [Installing on Linux](/docs/pages/installing-linux/) &mdash; CUDA, ROCm, Vulkan, CPU
- [Installing on Windows](/docs/pages/installing-windows/) &mdash; CUDA, ROCm, Vulkan, CPU

## What the installer does

The installer detects the best release bundle for your machine, downloads the matching Mesh release, installs the `mesh-llm` binary, and adds the install directory to your user `PATH` when needed.

Default install locations:

| Platform | Default location |
|---|---|
| macOS/Linux | `~/.local/bin` |
| Windows | `%LOCALAPPDATA%\mesh-llm\bin` |

## Verify the install

```sh
mesh-llm --version
```

## Next step

Run the [Quickstart](/docs/pages/quickstart/) to start a private node and open the console.

## See also

- [Hardware support](/docs/pages/hardware-support/)
- [Updating Mesh](/docs/pages/updating-mesh/)
