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

The installer downloads the `mesh-llm` executable and adds `%LOCALAPPDATA%\mesh-llm\bin` to your user `PATH` when needed. In an interactive PowerShell session it then runs `mesh-llm.exe setup` itself, which installs the recommended runtime and asks whether to install the background service (default Yes). Otherwise, or with `-NoSetup`, it prints the setup command to run next.

`mesh-llm serve` is a foreground process attached to the current terminal.
When setup installs the optional background service, a per-user scheduled task
named `mesh-llm` starts it at your logon, with no console window and no
administrator rights. It restarts the node after a failure (after 5 seconds,
then doubling, up to five restarts in a row) and writes its output to
`%LOCALAPPDATA%\mesh-llm\logs\service.log`, keeping the part before in
`service.previous.log` once it passes 10 MiB. The task runs while you are logged
in, not before logon. Setup starts the task as soon as it installs it, so a
node is usually running on the console and API ports when setup finishes. Check
setup's summary: without a desktop session (setup run over SSH, for example)
the task is installed but cannot start, and setup says it starts at your next
logon. It does, or start it with `schtasks /Run /TN mesh-llm` once you are
signed in to the desktop; until then no node holds the ports.

Do not start a second foreground copy on the same ports: load models into the
running node with `mesh-llm load <model>`, or stop the task with
`schtasks /End /TN mesh-llm` before switching to foreground operation, and start
it again with `schtasks /Run /TN mesh-llm`. Ending the task also ends its node.
Use `schtasks /End` rather than `mesh-llm stop` for this node: `stop` has to
force-end a node that has no console window, and the task treats that as a
failure and starts it again.

## Next step

If setup did not run during install, run `mesh-llm.exe setup` to finish machine setup. See the [CLI guide](/docs/pages/CLI/) for the setup flags. Then follow the [Quickstart](/docs/pages/quickstart/) from its step 3.

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
