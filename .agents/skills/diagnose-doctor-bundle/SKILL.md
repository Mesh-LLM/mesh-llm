---
name: diagnose-doctor-bundle
description: Use when diagnosing a mesh-llm doctor zip bundle, support bundle, runtime logs, GPU inventory, plugin inventory, API snapshots, or crash report, and the user wants likely causes, evidence, and suggested fixes.
---

# Diagnose Doctor Bundle

Use this skill to turn a `mesh-llm doctor` zip into a practical support
diagnosis. Prefer evidence from the bundle over guesses, and separate facts
from likely causes.

## Workflow

1. Locate the bundle path from the user or attached files. If no path or file is
   available, ask for the doctor zip.
2. Run the bundled helper:

   ```bash
   python3 .agents/skills/diagnose-doctor-bundle/scripts/summarize_bundle.py /path/to/mesh-llm-doctor.zip
   ```

   Use `--json` when you need structured output for further processing.
3. Inspect raw bundle files when the summary points at a problem:

   ```bash
   unzip -l /path/to/mesh-llm-doctor.zip
   unzip -p /path/to/mesh-llm-doctor.zip manifest.json
   unzip -p /path/to/mesh-llm-doctor.zip system.json
   unzip -p /path/to/mesh-llm-doctor.zip runtime/logs/<name>
   ```

4. Produce a diagnosis with:
   - observed facts: version, OS/arch, detected flavor, CPU/memory, GPUs,
     plugins, selected runtime, API reachability, manifest warnings
   - likely root cause(s), each tied to evidence from files or log snippets
   - suggested fixes or next commands, ordered by confidence and impact
   - what is still unknown and what extra artifact would settle it

## What To Check

- `manifest.json`: warnings, selected runtime, included files, truncated logs.
- `system.json`: flavor mismatch, low available memory, CPU/OS/arch details,
  runtime port.
- `gpus.json`: no GPUs, backend detection mismatch, low VRAM,
  CUDA/ROCm/Metal/Vulkan errors, unavailable devices.
- `plugins.json`: installed versions, disabled plugins, inactive runtime plugins,
  missing commands, unexpected plugin versions.
- `runtime/owner.json` and `runtime/instances.json`: current vs stale runtime,
  wrong API port, dead owner, unexpected binary path/version.
- `api/*.json`: console unreachable, `/v1/models` empty, status errors,
  model inventory mismatch.
- `runtime/logs/*`: panics, `ERROR`, OOM, port bind failures, model load
  failures, GGUF/package errors, skippy stage failures, network join failures,
  plugin process failures.

## Response Style

Lead with the highest-confidence finding. Do not dump the whole bundle summary.
Use short evidence snippets with file names. Avoid claiming a root cause when
the bundle only shows a symptom.

Recommended shape:

```text
Likely cause: ...
Evidence:
- manifest.json: ...
- runtime/logs/mesh-llm.log: ...

Suggested fix:
1. ...
2. ...

Still unknown:
- ...
```

## Safety

Doctor bundles include the resolved `config.toml` when available and avoid
environment variable values, but logs and config can still contain user prompts,
local paths, hostnames, model names, or tokens emitted by external tools. Do not
paste long logs back to the user. Quote only the minimal evidence needed for the
diagnosis.
