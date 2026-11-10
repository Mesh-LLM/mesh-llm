# Proposed catalog entry: powermetrics-plugin

Adds [powermetrics-plugin](https://github.com/Mesh-LLM/powermetrics-plugin)
to the plugin catalog, so it can be found with `mesh-llm plugins search` and
installed by name now that releases no longer bundle plugins (#2977). It
stays outside `recommended` and adds no set.

```sh
mesh-llm plugins install powermetrics-plugin
```

This directory is a reviewable proposal. Merging it into GitHub does not
publish anything: the live catalog is the Hugging Face
[`meshllm/plugin-catalog` dataset](https://huggingface.co/datasets/meshllm/plugin-catalog/tree/main),
and [plugins.additions.jsonl](plugins.additions.jsonl) is the line to append
to its `plugins.jsonl`.

## Release pin

| Plugin | Release pin | Targets |
| --- | --- | --- |
| `powermetrics-plugin` | [v0.2.0](https://github.com/Mesh-LLM/powermetrics-plugin/releases/tag/v0.2.0) | Apple Silicon macOS, Linux x64, Linux ARM64, Windows x64 |

The four SHA-256 pins are those of the release's versioned archives
(`powermetrics-plugin-v0.2.0-<target>.tar.gz`, `.zip` on Windows), as GitHub
reports them for that release. The entry name is the plugin's own name, the
one its `[[plugin]]` config entry uses.

## What it needs

- A host with the `local_inference_observation.v1` and `plugin_status.v1`
  capabilities (#2608, #2601). The plugin refuses to initialize on an older
  one. v0.78.1 has neither, so 0.79 is the first release it runs on. The
  entry declares no host version bound: source builds with those capabilities
  still report `0.78.0`, so a numeric bound would not be truthful.
- The observation feed turned on for it. Installing alone produces no
  figures:

  ```toml
  [[plugin]]
  name = "powermetrics-plugin"
  allow_inference_observation = true
  ```

- A power reading the plugin supports: `nvidia-smi` on any OS, IOReport
  through macmon on Apple Silicon (no sudo), ADLX on Windows or amdgpu hwmon
  on Linux for AMD. Intel GPUs are not covered yet.

## The price panel on 0.79

v0.2.0 reads the node's own prices from the `payment` block of `/v1/models`,
which #2654 removed. On a 0.79 host its Energy page still shows the power,
energy and cost figures, and leaves the price comparison panel hidden.
[powermetrics-plugin#9](https://github.com/Mesh-LLM/powermetrics-plugin/pull/9)
reads the prices from `pay-lightning` instead; it is merged and will ship as
v0.2.1 after the 0.79 release, and this pin moves to it then.
