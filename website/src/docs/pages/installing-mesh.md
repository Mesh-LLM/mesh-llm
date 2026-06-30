---
title: Installing Mesh
---

# Installing Mesh

Mesh runs on macOS, Linux, and Windows. Choose your platform for detailed install instructions.

## Choose your platform

- [Installing on macOS](/docs/pages/installing-macos/) (Apple Silicon, Homebrew)
- [Installing on Linux](/docs/pages/installing-linux/) (platform details)
- [Installing on Windows](/docs/pages/installing-windows/) (platform details)

## What the installer does

The installer downloads the `mesh-llm` executable and adds the install directory to your user `PATH` when needed. After install, run `mesh-llm setup` to finish runtime configuration and service setup.

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

Run `mesh-llm setup` to finish machine setup, then follow the [Quickstart](/docs/pages/quickstart/) to start a private node and open the console.

## Uninstall

Preview what Mesh would remove:

```sh
mesh-llm uninstall --dry-run
```

Remove the executable, setup-owned service files, and native-runtime cache:

```sh
mesh-llm uninstall --yes
```

Uninstall preserves `~/.mesh-llm` configuration and identity data by default.
Use `--purge-config` only when you intentionally want to remove that data too.

## See also

- [Hardware support](/docs/pages/hardware-support/)
- [Updating Mesh](/docs/pages/updating-mesh/)
