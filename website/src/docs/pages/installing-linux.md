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

## Native packages and containers

Versioned Ubuntu 24.04 `.deb` and Arch Linux `.pkg.tar.zst` files are published
as [Mesh packaging release assets](https://github.com/Mesh-LLM/mesh-packaging/releases).
Choose the file whose distro, architecture, and backend suffix matches the host.
These are directly downloadable release assets, not apt or pacman repositories;
download the package's matching `.sha256` sidecar and verify it before installing
with `apt install` or `pacman -U`.

For Ubuntu, replace the placeholders with the version, architecture, and backend
from the selected release asset. Architecture is `amd64` or `arm64`; copy the
exact backend suffix, such as `cpu`, `vulkan`, or `cuda12.9.2`, from the release:

```sh
PACKAGE='mesh-llm-<version>-ubuntu-<architecture>-<backend>.deb'
RELEASE_URL='https://github.com/Mesh-LLM/mesh-packaging/releases/download/packaging-v<version>'
curl -fLO "$RELEASE_URL/$PACKAGE"
curl -fLO "$RELEASE_URL/$PACKAGE.sha256"
sha256sum --check "$PACKAGE.sha256"
sudo apt install "./$PACKAGE"
```

For Arch Linux, use the matching `amd64` package and backend suffix:

```sh
PACKAGE='mesh-llm-<version>-arch-amd64-<backend>.pkg.tar.zst'
RELEASE_URL='https://github.com/Mesh-LLM/mesh-packaging/releases/download/packaging-v<version>'
curl -fLO "$RELEASE_URL/$PACKAGE"
curl -fLO "$RELEASE_URL/$PACKAGE.sha256"
sha256sum --check "$PACKAGE.sha256"
sudo pacman -U "./$PACKAGE"
```

The same CPU, Vulkan, CUDA, and ROCm package variants are available as OCI
images from [`ghcr.io/mesh-llm/mesh-llm`](https://github.com/orgs/Mesh-LLM/packages/container/package/mesh-llm).
Immutable tags include the Mesh version, distro, architecture, and backend. See
the [packaging matrix](https://github.com/Mesh-LLM/mesh-packaging/blob/main/docs/matrix.md)
for the supported rows and exact tag scheme.

Alpine packages are not published because Mesh release archives currently use
glibc rather than musl.

## What the installer does

The installer downloads the `mesh-llm` executable and adds `~/.local/bin` to your user `PATH` when needed. After install, run `mesh-llm setup` to finish runtime configuration and, if you want it, the background service.

## Next step

Run `mesh-llm setup` to finish machine setup. See the [CLI guide](/docs/pages/CLI/) for the setup flags.

## Uninstall

```sh
mesh-llm uninstall --dry-run
mesh-llm uninstall --yes
```

On Linux, uninstall disables the per-user systemd unit when present, removes
setup-owned service files, removes the native-runtime cache, and removes the
executable last. It preserves `~/.mesh-llm` unless you pass `--purge-config`.

## Advanced install

Install the latest prerelease:

```sh
curl -fsSL https://meshllm.cloud/install.sh | bash -s -- --pre-release
```

Install to a custom location:

```sh
curl -fsSL https://meshllm.cloud/install.sh | bash -s -- --install-dir "$HOME/bin"
```

## See also

- [Installing on macOS](/docs/pages/installing-macos/)
- [Installing on Windows](/docs/pages/installing-windows/)
- [Hardware support](/docs/pages/hardware-support/)
- [Updating Mesh](/docs/pages/updating-mesh/)
