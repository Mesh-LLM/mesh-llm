# Install

Install Mesh on every machine that should serve a model or call into a mesh.

## Recommended install

macOS or Linux:

```sh
curl -fsSL https://mesh-llm.cloud/install.sh | bash
```

Windows PowerShell:

```powershell
irm https://mesh-llm.cloud/install.ps1 | iex
```

Open a new terminal after install if the installer added Mesh to your `PATH`.

Check the install:

```sh
mesh-llm --version
```

## What the installer does

The installer:

- detects the best release bundle for this machine
- downloads the matching Mesh release
- installs the `mesh-llm` binary
- adds the install directory to your user `PATH` when needed

Default install locations:

| Platform | Default location |
|---|---|
| macOS/Linux | `~/.local/bin` |
| Windows | `%LOCALAPPDATA%\mesh-llm\bin` |

## Force a flavor

Most users should let the installer auto-detect. Force a flavor when auto-detection is wrong, when you are preparing a machine image, or when you intentionally want CPU/Vulkan instead of a vendor GPU backend.

macOS/Linux:

```sh
curl -fsSL https://mesh-llm.cloud/install.sh | bash -s -- --flavor vulkan
```

Windows PowerShell:

```powershell
& ([scriptblock]::Create((irm https://mesh-llm.cloud/install.ps1))) -Flavor vulkan
```

Supported release flavors:

| Platform | Flavors |
|---|---|
| macOS Apple Silicon | `metal` |
| Linux x86_64 | `cuda-blackwell`, `cuda`, `rocm`, `vulkan`, `cpu` |
| Linux ARM64 | `cpu` |
| Windows x86_64 | `cuda-blackwell`, `cuda`, `rocm`, `vulkan`, `cpu` |

## Advanced install

Install the latest prerelease:

```sh
curl -fsSL https://mesh-llm.cloud/install.sh | bash -s -- --pre-release
```

Windows PowerShell:

```powershell
& ([scriptblock]::Create((irm https://mesh-llm.cloud/install.ps1))) -PreRelease
```

Install somewhere else:

```sh
curl -fsSL https://mesh-llm.cloud/install.sh | bash -s -- --install-dir "$HOME/bin"
```

Windows PowerShell:

```powershell
& ([scriptblock]::Create((irm https://mesh-llm.cloud/install.ps1))) -InstallDir "$HOME\bin"
```

## Next step

Run the [Quickstart](/docs/pages/quickstart/) to start a private node and open the console.
