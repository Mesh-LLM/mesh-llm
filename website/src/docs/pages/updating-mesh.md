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

## Switch flavors

Use `--flavor` when you want to switch the installed release bundle:

```sh
mesh-llm update --flavor cuda
mesh-llm update --flavor cuda-blackwell
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
