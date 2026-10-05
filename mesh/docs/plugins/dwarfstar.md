# DwarfStar (ds4) engine plugin

[DwarfStar for Mesh](https://github.com/Mesh-LLM/ds4-plugin) runs
[antirez/ds4](https://github.com/antirez/ds4) models (DeepSeek V4 Flash and
others) as an alternative engine to Mesh's built-in runtime. Mesh starts and
stops the bundled engine; clients use Mesh's normal OpenAI-compatible API.

**Early access: Apple Silicon, Mesh v0.77.0+.** Models are large: the DeepSeek
V4 Flash Q2 weights are ~81 GiB — that much disk for the first download, and
that much free unified memory to serve them, because the engine maps the
weights into memory. Pick a model that fits both.

## Setup

```sh
mesh-llm plugins install Mesh-LLM/ds4-plugin
```

Add to `~/.mesh-llm/config.toml`:

```toml
[runtime]
mode = "on_demand"   # don't also load a built-in model next to DwarfStar

[[plugin]]
name = "ds4-plugin"
args = ["serve", "--model", "ds4f-q2"]
```

Then run `mesh-llm serve`. The first start downloads the model (progress in the
terminal); later starts reuse it. Once loaded, the model appears in
`curl http://127.0.0.1:9337/v1/models`.

Other model names, using existing weights (`--weights /path.gguf`), context
size and storage location are in the
[plugin README](https://github.com/Mesh-LLM/ds4-plugin#readme). DwarfStar runs
on one node; Mesh does not split it across machines.
