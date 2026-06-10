# Hardware Support

Mesh can run on one machine or across several machines. The release flavor controls which local runtime backend is used.

## Which flavor should I use?

| Machine | Recommended flavor | Install behavior |
|---|---|---|
| Apple Silicon Mac | `metal` | macOS installer selects it automatically. |
| Linux NVIDIA (all architectures) | `cuda` | Linux installer selects CUDA when NVIDIA tooling or devices are detected. |
| Linux AMD | `rocm` | Use when ROCm/HIP is installed and supported by the GPU. |
| Linux Vulkan-capable GPU | `vulkan` | Useful when CUDA/ROCm are not available. |
| Linux ARM64 | `cpu` | Published ARM64 Linux bundle is CPU-only. |
| Windows NVIDIA | `cuda` | Windows installer detects NVIDIA when possible. |
| Windows AMD | `rocm` | Use when the Windows HIP runtime is available. |
| Any supported OS | `cpu` | Slowest, but useful for testing and API-only workflows. |

The `cuda` flavor covers all NVIDIA GPUs. The legacy `cuda-blackwell` name was introduced when Blackwell hardware required a separate CUDA toolkit (12.8) due to an nvcc/driver cubin incompatibility. In the runtime source at [`crates/mesh-llm-native-runtime/src/flavor.rs`](https://github.com/Mesh-LLM/mesh-llm/blob/main/crates/mesh-llm-native-runtime/src/flavor.rs), `cuda-blackwell` and `blackwell` are parsed as aliases for the same `Cuda` backend kind — there has never been a separate Blackwell runtime backend. As the mesh runtimes are consolidated into a unified release lane, these legacy aliases will be removed. Users currently selecting `cuda-blackwell` should use `cuda` instead.

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
