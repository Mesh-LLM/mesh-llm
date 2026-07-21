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

Mesh publishes a versioned Homebrew formula for Apple Silicon with each
[packaging release](https://github.com/Mesh-LLM/mesh-packaging/releases). The
formula is a release asset rather than a tap, so download it before installing:

```sh
brew tap-new mesh-llm/release
curl -fL https://github.com/Mesh-LLM/mesh-packaging/releases/latest/download/mesh-llm.rb \
  -o "$(brew --repository mesh-llm/release)/Formula/mesh-llm.rb"
brew install mesh-llm/release/mesh-llm
```

Homebrew requires formulae to live in a tap. The first command creates a local
tap; it does not clone or depend on a public `Mesh-LLM/homebrew-tap` repository.
To update later, download the latest formula to the same path and run
`brew upgrade mesh-llm`.

The formula downloads the checksummed Metal release archive from
`Mesh-LLM/mesh-llm`. Intel macOS is not available through Homebrew because the
upstream release does not include an x86_64 macOS archive.

## What the installer does

The installer downloads the `mesh-llm` executable and adds `~/.local/bin` to your user `PATH` when needed. After install, run `mesh-llm setup` to finish runtime configuration and, if you want it, the background service.

## Next step

Run `mesh-llm setup` to finish machine setup. See the [CLI guide](/docs/pages/CLI/) for the setup flags.

## Uninstall

```sh
mesh-llm uninstall --dry-run
mesh-llm uninstall --yes
```

On macOS, uninstall boots out the per-user launchd agent when present, removes
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

- [Installing on Linux](/docs/pages/installing-linux/)
- [Installing on Windows](/docs/pages/installing-windows/)
- [Hardware support](/docs/pages/hardware-support/)
- [Updating Mesh](/docs/pages/updating-mesh/)
