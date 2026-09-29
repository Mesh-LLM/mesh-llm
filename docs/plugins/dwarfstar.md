# DwarfStar (ds4) engine plugin

[DwarfStar for Mesh](https://github.com/Mesh-LLM/ds4-plugin) runs DeepSeek V4
Flash through [antirez/ds4](https://github.com/antirez/ds4), an alternative
inference engine to Mesh's built-in Skippy/llama.cpp runtime. Mesh launches the
plugin; the plugin starts and stops its bundled DwarfStar server. Clients use
Mesh's normal OpenAI-compatible API for chat, streaming and tool calls.

**Early access: Apple Silicon, tested with Mesh v0.77.0.** Flash Q2 needs about
81 GiB of disk space for weights and a 96 GB or larger Mac for resident
inference. Leave memory for macOS, context and other applications.

This is a single-node model execution engine exposed through Mesh, not a way to
split DwarfStar model layers across machines. You do not need to build the
engine or launch a separate server.

## Install and choose weights

```sh
mesh-llm plugins install Mesh-LLM/ds4-plugin
```

The package includes the plugin and native engine. The macOS binaries are
ad-hoc signed, not Apple-notarized. Installing does not download model weights.

If you already have compatible DeepSeek V4 Flash Q2 weights, use their path in
the configuration below. Otherwise, review the model size and
[weight licence](https://huggingface.co/antirez/deepseek-v4-gguf), then download
explicitly:

```sh
~/.mesh-llm/plugins/installed/ds4-plugin/ds4-plugin catalog
~/.mesh-llm/plugins/installed/ds4-plugin/ds4-plugin download --model ds4f-q2 --directory "$HOME/Models/ds4" --accept-download
```

The downloader requires curl, resumes interrupted transfers and verifies
SHA-256. Use the completed weight-file path it prints, not a `.partial` file.

## Configure and start

`on_demand` starts Mesh without also loading a built-in model at startup
when no model is explicitly supplied on the Mesh command line. DwarfStar
still loads its configured weights when the plugin starts.

Add this to `~/.mesh-llm/config.toml`, replacing the absolute weight path. If
`[runtime]` already exists, edit its `mode` instead of adding another section.

```toml
[runtime]
mode = "on_demand"

[[plugin]]
name = "ds4-plugin"
args = ["serve", "--weights", "/absolute/path/to/model.gguf", "--context", "4096"]
```

```sh
mesh-llm serve
```

The model loads when Mesh starts the plugin, **not** on the first chat request.
Once ready, check discovery and send a request:

```sh
curl http://127.0.0.1:9337/v1/models
curl http://127.0.0.1:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello"}]}'
```

Point your OpenAI-compatible client at `http://127.0.0.1:9337/v1` and select
`deepseek-v4-flash`. Upstream also lists a PRO alias for the same loaded model;
it is not a second model.

Ctrl+C in the Mesh terminal stops the plugin and its backend. Remove the
`ds4-plugin` configuration entry and restart Mesh to stop loading it on future
launches. Changes to `--weights` or `--context` also require restarting Mesh.
Weights remain in your model directory.

See the [plugin README](https://github.com/Mesh-LLM/ds4-plugin#readme) for current
platform support, download recovery, offline installation and development
instructions.
