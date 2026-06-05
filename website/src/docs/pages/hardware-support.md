# Hardware Support

Mesh can run on one machine or across several machines. The release flavor controls which local runtime backend is used.

## Which flavor should I use?

| Machine | Recommended flavor | Install behavior |
|---|---|---|
| Apple Silicon Mac | `metal` | macOS installer selects it automatically. |
| Linux NVIDIA, Blackwell | `cuda-blackwell` | Linux installer detects Blackwell when possible. |
| Linux NVIDIA, pre-Blackwell | `cuda` | Linux installer selects CUDA when NVIDIA tooling or devices are detected. |
| Linux AMD | `rocm` | Use when ROCm/HIP is installed and supported by the GPU. |
| Linux Vulkan-capable GPU | `vulkan` | Useful when CUDA/ROCm are not available. |
| Linux ARM64 | `cpu` | Published ARM64 Linux bundle is CPU-only. |
| Windows NVIDIA | `cuda` or `cuda-blackwell` | Windows installer detects NVIDIA when possible. |
| Windows AMD | `rocm` | Use when the Windows HIP runtime is available. |
| Any supported OS | `cpu` | Slowest, but useful for testing and API-only workflows. |

## Model fit

VRAM requirements are not exact. Context size, runtime overhead, other GPU memory use, platform differences, and concurrency all matter.

Use [Choose a model](/docs/pages/choose-a-model/) for starting points. If a model fails to load, try a smaller model or smaller quant first.

## Add capacity

Add another machine when:

- one machine cannot fit the model you want
- you want a second machine to serve a different model
- you want a laptop to use a workstation through a local API

Start every serving machine with the same private mesh name:

```sh
mesh-llm serve --discover my-private-mesh --model <model-ref>
```

Join from an API-only laptop:

```sh
mesh-llm client --discover my-private-mesh
```
