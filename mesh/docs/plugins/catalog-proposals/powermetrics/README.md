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

## Publication blocker: hosts older than 0.79

The catalog is shared by every host version, and its entries cannot say which
hosts they need. A v0.78.1 host can therefore install this entry by name, but
the plugin then refuses to initialize there (it needs the capabilities below),
and v0.78.1 treats a configured plugin that fails to start as required, so
the node stops at startup. 0.79 starts such a plugin as optional instead. A
numeric bound would not help: v0.78.1 does not read one, and source builds
with the capabilities still report `0.78.0`.

**Publish this entry only once that is handled**: catalog entries that carry
a host floor older hosts respect, or a decision that v0.78.1 is no longer
supported for catalog installs.

## Release pin

| Plugin | Release pin | Targets |
| --- | --- | --- |
| `powermetrics-plugin` | [v0.2.1](https://github.com/Mesh-LLM/powermetrics-plugin/releases/tag/v0.2.1) | Apple Silicon macOS, Linux x64, Linux ARM64, Windows x64 |

The four SHA-256 pins are those of the release's versioned archives
(`powermetrics-plugin-v0.2.1-<target>.tar.gz`, `.zip` on Windows), as GitHub
reports them for that release. The entry name is the plugin's own name, the
one its `[[plugin]]` config entry uses.

## What it needs

- A host with the `local_inference_observation.v1` and `plugin_status.v1`
  capabilities (#2608, #2601). v0.78.1 has neither, so 0.79 is the first
  release it runs on.
- The observation feed turned on for it. Installing alone produces no
  figures, and a node that started the plugin without this grant has to be
  restarted after it is set: the plugin stops measuring when its subscription
  is refused at startup.

  ```toml
  [[plugin]]
  name = "powermetrics-plugin"
  allow_inference_observation = true
  ```

- A power reading the plugin supports: `nvidia-smi` on any OS, IOReport
  through macmon on Apple Silicon (no sudo), ADLX on Windows or amdgpu hwmon
  on Linux for AMD. Intel GPUs are not covered yet.

## The Energy page

The page shows the board power, the idle draw and both energy rates. The cost
figures appear only when the plugin is given an electricity price, for
example `args = ["--electricity-price-per-kwh", "0.25", "--currency", "EUR"]`
in the entry above. The price comparison panel reads the node's own prices
from `pay-lightning` (0.79 moved them there, #2654), so it shows only on a
node with `pay-lightning` and a wallet plugin bound to it.

Checked on Windows 11 with the published 0.79.0 host: built from main at
powermetrics-plugin#9 (v0.2.1 adds only the version bump on top), the plugin
reports `ready` from the `adlx-apu` source and counts a request's output
tokens.
