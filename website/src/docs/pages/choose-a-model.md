# Choose a Model

Start with a model that fits comfortably. After you see console chat working, move up to larger models or add more machines.

## Gemma 4 starting points

These Unsloth Gemma 4 GGUF refs are starting points, not guarantees. Fit depends on context size, runtime overhead, platform, other GPU memory use, and concurrency.

| Available VRAM | Try first |
|---:|---|
| 8GB | `unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL` |
| 12GB | `unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL` |
| 16GB | `unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q3_K_XL` |
| 24GB | `unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M` |
| 64GB+ | `unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL` |

If loading fails, try the next smaller row or a smaller quant.

## Inspect before serving

```sh
mesh-llm models show unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
```

Search the catalog:

```sh
mesh-llm models search gemma
```

## Serve the selected model

```sh
mesh-llm serve --discover my-private-mesh --model unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
```

## When to add machines

Add another machine when:

- the model does not fit on one device
- you want another machine to serve a different model
- you want an API-only laptop to route to a workstation

For multi-machine large-model serving, use catalog entries with layer packages. Layer packages let Mesh place parts of a supported model across available machines while requests still go through `http://localhost:9337/v1`.
