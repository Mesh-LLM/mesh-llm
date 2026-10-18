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

The installer downloads the `mesh-llm` executable and adds `%LOCALAPPDATA%\mesh-llm\bin` to your user `PATH` when needed. After install, run `mesh-llm.exe setup` to finish runtime configuration and, if you want it, the background service.

`mesh-llm serve` is a foreground process attached to the current terminal.
When setup installs the optional background service, a per-user scheduled task
named `mesh-llm` starts it at your logon, with no console window and no
administrator rights. It restarts the node after a failure (after 5 seconds,
then doubling, up to five restarts in a row) and writes its output to
`%LOCALAPPDATA%\mesh-llm\logs\service.log`, keeping the part before in
`service.previous.log` once it passes 10 MiB. The task runs while you are logged
in, not before logon. Do not start a second foreground copy on the same ports;
stop the task with `schtasks /End /TN mesh-llm` before switching to foreground
operation, and start it again with `schtasks /Run /TN mesh-llm`.

## Next step

Run `mesh-llm.exe setup` to finish machine setup. See the [CLI guide](/docs/pages/CLI/) for the setup flags.

## Uninstall

```powershell
mesh-llm.exe uninstall --dry-run
mesh-llm.exe uninstall --yes
```

On Windows, uninstall ends and deletes the `mesh-llm` scheduled task when
present, removes its definition and logs, and removes the native-runtime cache
and the executable. It preserves `%USERPROFILE%\.mesh-llm` unless you pass
`--purge-config`.

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
