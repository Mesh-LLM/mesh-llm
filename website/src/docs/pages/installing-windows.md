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

## What the installer does

The installer downloads the `mesh-llm` executable and adds `%LOCALAPPDATA%\mesh-llm\bin` to your user `PATH` when needed. After install, run `mesh-llm.exe setup` to finish runtime configuration.

## Next step

Run `mesh-llm.exe setup` to finish machine setup. See the [CLI guide](/docs/pages/CLI/) for the setup flags.

## Uninstall

```powershell
mesh-llm.exe uninstall --dry-run
mesh-llm.exe uninstall --yes
```

On Windows, uninstall removes the executable and native-runtime cache. It
preserves `%USERPROFILE%\.mesh-llm` unless you pass `--purge-config`.

## Advanced install

Install the latest prerelease:

```powershell
& ([scriptblock]::Create((irm https://meshllm.cloud/install.ps1))) -PreRelease
```

Install to a custom location:

```powershell
& ([scriptblock]::Create((irm https://meshllm.cloud/install.ps1))) -InstallDir "$HOME\bin"
```

## See also

- [Installing on macOS](/docs/pages/installing-macos/)
- [Installing on Linux](/docs/pages/installing-linux/)
- [Hardware support](/docs/pages/hardware-support/)
- [Updating Mesh](/docs/pages/updating-mesh/)
