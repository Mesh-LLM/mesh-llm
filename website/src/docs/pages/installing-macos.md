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

Mesh publishes a versioned Homebrew formula for Apple Silicon through the
canonical [`Mesh-LLM/tap`](https://github.com/Mesh-LLM/homebrew-tap) tap:

```sh
brew install Mesh-LLM/tap/mesh-llm
```

The fully qualified formula name adds the tap automatically. Update later with
`brew update` followed by `brew upgrade mesh-llm`.

The formula downloads the checksummed Metal release archive from
`Mesh-LLM/mesh-llm`. `Mesh-LLM/mesh-packaging` produces and validates each
formula before the tap publishes it. Intel macOS is not available through
Homebrew because the upstream release does not include an x86_64 macOS archive.

## What the installer does

The installer downloads the `mesh-llm` executable and adds `~/.local/bin` to your user `PATH` when needed. After install, run `mesh-llm setup` to finish runtime configuration and, if you want it, the background service.

`mesh-llm serve` is a foreground process attached to the current terminal.
When setup installs the optional background service, launchd owns the same
logical runtime's startup, restart, and logs. Do not start a second foreground
copy on the same ports; stop or disable the setup-managed launchd agent before
switching to foreground operation.

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
- [Runtime Lifecycle](/docs/pages/runtime-lifecycle/)
