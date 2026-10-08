# Native manual family parity

Inventory and retained certification use the closed native JSON frontdoor below. Source preparation, acquisition and family promotion keep their existing owning gates.

Prepare the current pinned native checkout using the existing prepare/build owners. A manual inventory still requires current prepared source markers and native classification/runtime admission. Use the already built xtask executable to obtain its SHA-256, and the actual current controller Git HEAD for `authority.controller.revision` and `authority.base`; `authority.controller.root` and `authority.root` identify that exact checkout. These are local inspection bindings, not independently trusted producer custody.

Write an input JSON with this closed schema (replace the absolute paths and digest/revision placeholders):

```json
{
  "authority": {
    "controller": {"root": "/absolute/mesh-llm", "revision": "CURRENT_40_HEX_HEAD", "executable_sha256": "CURRENT_64_HEX_XTASK_SHA"},
    "root": "/absolute/mesh-llm", "base": "CURRENT_40_HEX_HEAD"
  },
  "cache_root": "/absolute/huggingface/hub",
  "mode": "inventory",
  "statuses": [], "families": [], "llama_models": [], "priorities": [],
  "limit": null, "missing_only": false, "local_only": false,
  "policy": {}, "run": null, "admission_seconds": 300
}
```

```sh
cargo xtool automation replay-matrix parity-local --input /absolute/parity-local.json
```

Inventory reports all current admitted classifications and local diagnostics. `missing_only` and `local_only` are mutually exclusive inventory views; priorities filter the inventory view. Status/family/model/priority lists and limit govern the prepared invocation selection; empty statuses mean candidate, candidate_stateful, candidate_multimodal and certified. Non-runnable declared statuses remain visible. Unpinned cache rows carry `observed_offline_cache_not_provider_pin`; missing/corrupt rows do not produce invocations. Source pin precedence can deliberately differ from a historical repo alias.

For execution set mode to `run` and supply the existing retained execution settings object as `run` (without its `plan` field). The frontdoor supplies the admitted plan and refuses a caller override:

```json
{
  "harness_sha256": "CURRENT_64_HEX_HARNESS_SHA",
  "evidence": "/absolute/fresh-parity-evidence",
  "max_seconds": 3600,
  "candidate_seconds": 600,
  "stop_on_failure": false,
  "stage_build_dir": null,
  "path": "/usr/local/bin:/usr/bin:/bin"
}
```

The harness is always the admitted source root’s scripts/family-certify.sh; it must match harness_sha256. evidence must be fresh and absolute, max_seconds must be 4..86400, and candidate_seconds 1..43200. The private child environment is owned by the retained runner; path is the explicit executable search path and stage_build_dir optionally identifies an existing absolute build directory. No argument aliases or implicit Python subprocess are provided. Policy overrides are JSON fields, including ctx_size, n_gpu_layers (supports -1), prompt, skip_build, skip_state, state_payload_kind, prefix_token_count, cache_hit_repeats, borrow_resident_hits, cache_decoded_result_hits and startup_timeout_secs. Dense cache kind defaults from the source manifest; recurrent prefixes default kv-recurrent. Context increases as necessary for prefix reuse. Real run output/receipts are evidence of the retained process and schema checks, never automatic family promotion.

Continue using native `models parity-download --cadence manual` for acquisition (explicit immutable pins and discovery scope), and existing source/package admission for validate. This frontdoor never downloads, invents a new model-preparation engine or silently excludes unpinned declared rows. Complete shard/runtime/platform qualification and family promotion remain their owning runbook gates.

Nonstandard installed toolkit locations are explicit `run.toolkit_dirs` fields (or top-level `toolkit_dirs` for parity-local-run):

```json
"toolkit_dirs": {
  "CUDA_PATH": "/absolute/installed-cuda",
  "HIP_PATH": "/absolute/installed-hip",
  "ROCM_PATH": "/absolute/installed-rocm",
  "LLVMInstallDir": "/absolute/installed-llvm",
  "VULKAN_SDK": "/absolute/installed-vulkan-sdk"
}
```

Include only locations used by your operator profile. Omitted fields use the retained owner's existing system discovery; no ambient toolkit root is implicitly forwarded. If an ambient variable is set, declare the same canonical directory explicitly or clear that variable; conflicting, relative, missing or special-file roots refuse before evidence creation or child launch. Ordinary directory symlinks are supported through their canonical target. The exact five names are closed; arbitrary path/library/build-root aliases are rejected.

The actual retained child receives canonical directory values. Effective-profile and partial/final receipts record requested/canonical paths, root metadata and the actual owning Windows link-search subdirectory observations, with Unix inode/device identity where available. Those observations are rechecked immediately before and after each harness execution; changed root/search directory identity refuses and preserves logs. This proves stable observed local directory references at those boundaries, not installed library bytes, compiler authenticity, hostile concurrent replacement protection, actual linker/runtime performance or Windows host qualification. Existing typed stage_build_dir and explicit path remain authoritative. Toolkit installation and real family certification are separate operator gates.
