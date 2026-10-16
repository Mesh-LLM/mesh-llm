# Update

Update Mesh with one command:

```sh
mesh-llm update
```

Restart the local node after updating:

```sh
mesh-llm stop
mesh-llm serve --auto
```

For a private mesh, restart with the same mesh name and model you used before:

```sh
mesh-llm serve --discover my-private-mesh --model unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
```

## Release checks

At startup, mesh-llm checks for a newer release and shows it as a notice and in the web console. To turn these checks off, for example when a package manager updates mesh-llm, set this in `~/.mesh-llm/config.toml`:

```toml
[updates]
check = false
```

`MESH_LLM_NO_UPDATE_CHECK=1` does the same for one process and overrides the setting; `0` turns the checks back on. `mesh-llm update` and `--auto-update` are not affected.

## Switch flavors

Use `--flavor` when you want to switch the installed release bundle:

```sh
mesh-llm update --flavor cuda
mesh-llm update --flavor rocm
mesh-llm update --flavor vulkan
mesh-llm update --flavor cpu
```

Apple Silicon:

```sh
mesh-llm update --flavor metal
```

To re-detect the best flavor for the current machine:

```sh
mesh-llm update --detect-flavor
```

`--detect-flavor` cannot be combined with `--flavor`.

## Install a specific version

```sh
mesh-llm update --version v0.X.Y
mesh-llm update --version v0.X.Y --flavor vulkan
```

Update serving nodes one at a time when a mesh is actively handling traffic. Mixed-version meshes are expected to keep operating during rolling updates.
