# Rebase onto main bd5514dd8

## Completed integration, 2026-10-01

- All 73 commits replayed; merge base is `bd5514dd8de8db64d42975d04de0b0ad0583c7ba`.
- Replayed HEAD before integration commits: `cc3d01913a3b05fe9c3d607a380c846e08691af0`.
- Final HEAD: the receipt commit containing this section, resolvable with
  `git log -1 --format=%H -- .omo/evidence/rebase-main-bd5514dd8.md`.
  Final code/ledger HEAD before that receipt is `cf5b438cf`.
- Added code/ledger commits: `c470f6bae` immutable composer tool delivery;
  `35d6d4578` arm64 Swift parity; `56064d336` Windows row-trigger regression;
  `1f3437b01` Laya pins/resolution; `46ef49d75` source version fixtures;
  `cf5b438cf` source-backed migration ledgers. Final receipt commit follows.
- No push, force, skipped replay, amend, new dependency, CI dispatch or deliberate
  external network operation. Cargo used offline resolution. Cargo.lock merged
  without conflict and required no manual resolution.

### Conflict resolutions

| File | Resolution |
| --- | --- |
| `tools/xtask/src/workflow_checks.rs` | Kept branch deletion. Ported release version/source-only distinction and sccache/no-Cargo checks to `ci_validation/release_dispatch.rs` and new `release_compilation.rs`, with seven Rust tests. Local composite actions are traversed recursively. |
| `scripts/verify-swift-xcframework.py` | Kept deletion; Rust full-mode contract now requires arm64 only on all four platforms. |
| `scripts/tests/test_verify_swift_xcframework.py` | Kept deletion; Rust declaration and lipo tests reject unsupported x86_64 while complete arm64 fixtures pass. |
| `.github/actions/restore-smoke-inputs/action.yml` | Kept Rust extraction/projection/composition; `smoke_inputs` already writes explicit LF bytes, retaining main's Windows carriage-return intent. |
| `scripts/tests/test_ci_artifact_actions.py` | Kept branch caller-switch tests, then restored main's two arm64-only Swift cache assertions lost during conflict resolution. |

### Immutable automation delivery

The existing release host producer jobs build automation once per platform and
upload `automation-<OS>-<arch>-<source SHA>`. Each of the eight release composers
already declares its matching host producer in `needs`; the Rust guard checks
that edge and upload/restore ownership. `restore-automation` verifies executable
and source checksums, compares the source SHA, and exports
`MESH_LLM_AUTOMATION_BIN`. It invokes no Cargo. The shared upload/restore actions
are reusable for L2; no L2 caller changed. The local adapter QA executed the real
restore Bash with valid bytes and rejected modified executable bytes.

The guard also exposed ARM64 artifact smoke without sccache, CUDA cleanup's
failure/cancellation Cargo path, and publish's unconditional Rust manifest
validation behind conditionally initialized sccache. Their setup was corrected.
Laya's new shell caller named the deleted model resolver; it now uses the existing
Rust `models resolve` owner without changing the new Python parity driver.

### Main Python census dispositions, all 38 paths

| Path | Final disposition |
| --- | --- |
| `evals/agentic-replay.py` | Retained main progress/heartbeat, local-model digest preflight and ready runtime context polling. Still-Python DuckDB execution owner; Rust invocation already forwards expected SHA-256. |
| `evals/agentic_replay_evidence.py` | Retained exactly-one `models` entry and `context_length` projection; no Rust execution/evidence replacement exists. |
| `scripts/ci-laya-smoke.py` | New main file retained and registered as migration debt. |
| `scripts/cleanup-self-hosted.py` | Main upload-gated canary-preflight cleanup retained; L2 caller unchanged. |
| `scripts/llama-canary-family-evidence.py` | Main immutable roster/cache preflight and ready summary retained in the still-Python source-plan/build owner; Rust `canary_receipts` owns aggregation only. |
| `scripts/run-command-with-timeout.py` | Main fail-closed descendant cleanup and infrastructure exit 125 retained. |
| `scripts/skippy-laya-parity.py` | New main file retained and registered as debt. |
| `scripts/tests/test_agentic_replay.py` | Main live replay intent coverage retained. |
| `scripts/tests/test_agentic_replay_evidence.py` | Main runtime context intent retained in existing Python owner tests. |
| `scripts/tests/test_build_release.py` | Main wallet-lexe opt-in shell regression retained. |
| `scripts/tests/test_ci_artifact_actions.py` | Retained Rust adapter checks and restored main arm64 assertions; LF projection verified by Rust smoke-input test. |
| `scripts/tests/test_ci_lane_workflows.py` | Main Windows smoke projection and Laya device/gate assertions retained. |
| `scripts/tests/test_ci_laya_smoke.py` | New test retained and registered debt. |
| `scripts/tests/test_ci_product_layout_catalog.py` | Main catalog successor/deferred-gap census retained. |
| `scripts/tests/test_ci_sdk_producers.py` | Main four-arm64-target and cache/host-builder constraints retained. |
| `scripts/tests/test_ci_windows_composition.py` | Main catalog-triggered row coverage retained; added Rust real-catalog test for every platform-windows-cfg crate, including host-runtime, and absence of unintended Windows product builds. |
| `scripts/tests/test_cleanup_self_hosted.py` | Main still-Python cleanup tests retained. |
| `scripts/tests/test_llama_canary_agent_repair_contract.py` | Main infrastructure-cache gate outside repair retained. |
| `scripts/tests/test_llama_canary_family_evidence.py` | Main Python preflight tests retained with source-plan owner. |
| `scripts/tests/test_llama_upstream_canary_contract.py` | Main linear preflight/candidate/verification graph tests retained. |
| `scripts/tests/test_model_artifact_registry.py` | Main Laya manifest assertion retained; Rust smoke pin test now checks Laya repo/revision/digest and tolerates only that explicit addition to frozen product fixture. |
| `scripts/tests/test_plan_ci.py` | Main shared-host Windows selection retained; new Rust changed-crate test exercises current catalogs rather than stale frozen membership. |
| `scripts/tests/test_release_workflow_artifacts.py` | Main unsupported Intel Node-target exclusion retained; new Rust release compilation/artifact guard tests cover composition contract. |
| `scripts/tests/test_run_command_with_timeout.py` | Main supervision tests retained. |
| `scripts/tests/test_sccache_evidence.py` | Main metadata/publish job-local cache assertions retained; Rust initialization ordering/conditions/failure-suppression tests added. |
| `scripts/tests/test_skippy_laya_parity.py` | New main tests retained and registered debt. |
| `scripts/tests/test_skippy_system_one_smoke.py` | Main qualified Metal/System One and Laya CPU wiring retained. |
| `scripts/tests/test_swift_xcframework_env.py` | Main ARM64-only shell environment constraints retained. |
| `scripts/tests/test_verify_swift_xcframework.py` | Deleted replacement; intent ported to Rust full-mode tests. |
| `scripts/verify-swift-xcframework.py` | Deleted replacement; intent ported to Rust architecture contract. |
| `sdk/python/hatch_build.py` | New SDK file retained, registered component isolation debt. |
| `sdk/python/src/meshllm/__init__.py` | New SDK file retained, registered debt. |
| `sdk/python/src/meshllm/_binding.py` | New SDK file retained, registered debt and generated-binding import edge. |
| `sdk/python/src/meshllm/_generated/__init__.py` | New SDK generated module retained, registered debt. |
| `sdk/python/src/meshllm/_generated/mesh_ffi.py` | New generated bindings retained, registered debt. |
| `sdk/python/src/meshllm/client.py` | New SDK client retained, registered debt. |
| `sdk/python/src/meshllm/types.py` | New SDK types retained, registered debt. |
| `sdk/python/tests/test_client.py` | New SDK client tests retained, registered debt. |

### Validation

| Gate | Final result |
| --- | --- |
| `cargo fmt --all --check` | Exit 0 |
| `cargo check -p xtask` | Exit 0 |
| `cargo clippy -p xtask --all-targets -- -D warnings` | Exit 0 |
| `just no-console-print` | Exit 0 |
| `automation inventory --check` | Exit 0 |
| Examples prebuild | Exit 0 |
| Full `cargo test -p xtask -- --test-threads=1` | Exit 0; 59 targets, 2503 passed, 0 failed, 2 ignored |
| Restricted-PATH `just ci-validate` | Exit 0; seven Rust integration targets, 134 passed; 1717 legacy cases, 1703 passed, 14 skipped, 0 failures/errors; crate/release/console/publish consistency all passed |
| Restricted-PATH actionlint | Exit 0 |
| `git diff --check` | Exit 0 |
| Changed Rust LSP diagnostics | No diagnostics |

Cargo was serial with `CARGO_NET_OFFLINE=true`; full tests used absolute
`CARGO_TARGET_DIR`, `SPV_ADAPTER_FIXTURE=target/debug/examples/swift_privacy_lint_fixture`
and `MIGRATION_TEST_GIT=/usr/bin/git`. Logs are in the approved temporary directory
as `rebase-xtask-green.log` and `rebase-ci-green.log`. Earlier complete attempts
found a stale workspace/ABI version fixture and stale Laya frozen comparison,
both repaired; one unchanged raw-capture test hit its five-second deadline and
passed on retry. Restricted Homebrew Rust emits existing deployment-target linker
warnings; warning-denying Clippy passes on the configured Rust toolchain.

### Remaining debt and limits

- Twelve new main Python files are explicitly retained: four Laya smoke/test
  files and eight Python SDK files. Source-backed caller and boundary records
  register them without approving SDK isolation or claiming Python-free CI.
- Replay execution/evidence and canary source-plan/preflight remain Python-owned;
  aggregation/argument validation being Rust does not migrate those operations.
- No remote Windows release, live model smoke, release publication, Anthropic
  runtime or Claude CI execution was attempted. Main feature files were retained.

## Historical paused receipt

## Recovery and current state

- Branch: `task/xtask-automation-migration`.
- Pre-rebase HEAD: `597a2982269625dfc8a5ecda101d31297b7bd574`.
- Old merge base: `b3ac945f2`.
- Requested target: `origin/main`, `bd5514dd8`.
- Initial worktree: only `.scratch/` untracked.
- Started with `GIT_MASTER=1 git rebase origin/main`.
- Paused at commit 7 of 73, `ff2017a95`, `refactor(automation): separate workflow and bootstrap ownership`.
- Unresolved modify/delete conflict: `tools/xtask/src/workflow_checks.rs`.
- No conflicts have been staged as resolved. No follow-up commits, pushes,
  external CI mutations, or validation runs have occurred.
- The original branch remains recoverable through the recorded pre-rebase
  HEAD. `GIT_MASTER=1 git rebase --abort` restores the pre-rebase branch.

## Decision blocking this commit

Main adds release sccache-initialization checks to `workflow_checks.rs` and
prohibits Cargo execution in eight composition-only release jobs. The branch
deletes that module in favor of `ci_validation` modules and later switches
Windows composition to `.github/actions/prepare-automation`. That action runs
`just automation-bootstrap` to compile xtask. Its description explicitly
states that it builds the automation tool.

Preserving main's no-compilation composition contract and the branch's current
caller switch requires changing how the automation executable reaches these
jobs. Porting a direct-step text scan without checking the nested action would
miss the conflict and falsely certify the composition contract.

Options requiring a settled product/artifact decision:

1. Produce an immutable Windows automation executable upstream, declare its
   artifact dependency, and restore it in composition jobs without compiling.
2. Explicitly permit automation-only compilation in composition jobs and
   revise the normative contract and its tests. This changes main's contract.

No option has been selected. The rebase is deliberately left paused at the
conflicting commit for review.

## Main-side Python change census

Source: `GIT_MASTER=1 git diff --name-status b3ac945f2 origin/main -- '*.py'`.
The related shell/action/workflow diff was also inspected. This is a pending
disposition ledger, not a claim that ports or inventory registration passed.

| Main-side path | Required disposition and current status |
| --- | --- |
| `evals/agentic-replay.py` | Pending Rust replay-owner audit/port where callers converted; retain still-Python execution intent. Main adds progress, pinned local-model verification, and ready runtime-model context projection. |
| `evals/agentic_replay_evidence.py` | Pending Rust replay evidence port: read one `models` entry's `context_length`, not stage `ctx_size`. |
| `scripts/ci-laya-smoke.py` | New main Python; leave as-is, inventory debt registration pending. |
| `scripts/cleanup-self-hosted.py` | Still-Python L2-blocked owner; keep main's upload-gated `canary-preflight` cleanup. |
| `scripts/llama-canary-family-evidence.py` | Pending `canary_receipts` port of pinned roster/cache preflight and ready summary. |
| `scripts/run-command-with-timeout.py` | Still-Python L2-blocked owner; keep main's fail-closed descendant cleanup and infrastructure exit 125. |
| `scripts/skippy-laya-parity.py` | New main Python; leave as-is, inventory debt registration pending. |
| `scripts/tests/test_agentic_replay.py` | Pending replay-owner test intent audit/port. |
| `scripts/tests/test_agentic_replay_evidence.py` | Pending Rust replay context regression coverage. |
| `scripts/tests/test_build_release.py` | Pending release-owner test intent audit/port if converted. |
| `scripts/tests/test_ci_artifact_actions.py` | Pending artifact/action test intent audit/port if converted. |
| `scripts/tests/test_ci_lane_workflows.py` | Pending lane workflow intent audit/port if converted. |
| `scripts/tests/test_ci_laya_smoke.py` | New main Python test; leave as-is, inventory debt registration pending. |
| `scripts/tests/test_ci_product_layout_catalog.py` | Pending product layout intent audit/port if converted. |
| `scripts/tests/test_ci_sdk_producers.py` | Pending SDK producer intent audit/port if converted. |
| `scripts/tests/test_ci_windows_composition.py` | Pending Windows planner/catalog coverage port; row membership must not substitute for change-triggered selection. |
| `scripts/tests/test_cleanup_self_hosted.py` | Keep main's still-Python cleanup coverage. |
| `scripts/tests/test_llama_canary_agent_repair_contract.py` | Pending canary repair contract intent audit/port if converted. |
| `scripts/tests/test_llama_canary_family_evidence.py` | Pending Rust canary preflight tests. |
| `scripts/tests/test_llama_upstream_canary_contract.py` | Pending canary workflow intent audit/port if converted. |
| `scripts/tests/test_model_artifact_registry.py` | Pending model registry owner test intent audit/port. |
| `scripts/tests/test_plan_ci.py` | Pending Rust planner test intent audit/port. |
| `scripts/tests/test_release_workflow_artifacts.py` | Pending release workflow guard port, blocked by composition automation delivery decision. |
| `scripts/tests/test_run_command_with_timeout.py` | Keep main's still-Python process cleanup coverage. |
| `scripts/tests/test_sccache_evidence.py` | Pending Rust sccache evidence test intent audit/port. |
| `scripts/tests/test_skippy_laya_parity.py` | New main Python test; leave as-is, inventory debt registration pending. |
| `scripts/tests/test_skippy_system_one_smoke.py` | Pending owner intent audit; retain Python if still-Python owner. |
| `scripts/tests/test_swift_xcframework_env.py` | Pending Swift adapter intent audit/port if converted. |
| `scripts/tests/test_verify_swift_xcframework.py` | Keep branch deletion when reached; port ARM64-only full-platform coverage to Rust release Swift XCFramework tests. |
| `scripts/verify-swift-xcframework.py` | Keep branch deletion when reached; port full-mode ARM64-only architecture requirements to Rust release Swift XCFramework owner. |
| `sdk/python/hatch_build.py` | New main Python SDK; leave as-is, inventory registration pending. |
| `sdk/python/src/meshllm/__init__.py` | New main Python SDK; leave as-is, inventory registration pending. |
| `sdk/python/src/meshllm/_binding.py` | New main Python SDK; leave as-is, inventory registration pending. |
| `sdk/python/src/meshllm/_generated/__init__.py` | New main Python SDK; leave as-is, inventory registration pending. |
| `sdk/python/src/meshllm/_generated/mesh_ffi.py` | New main generated Python SDK bindings; leave as-is, inventory registration pending. |
| `sdk/python/src/meshllm/client.py` | New main Python SDK; leave as-is, inventory registration pending. |
| `sdk/python/src/meshllm/types.py` | New main Python SDK; leave as-is, inventory registration pending. |
| `sdk/python/tests/test_client.py` | New main Python SDK tests; leave as-is, inventory registration pending. |

## Validation

Not run. Rebase incomplete; there are no pass/fail test counts to report.
The L12 checkpoint was not skipped or dropped and has not yet been replayed.
