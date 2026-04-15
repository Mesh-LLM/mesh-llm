# Shape Up: Recover PR 235's intent without regressing CI tuning from PRs 209 + 211

## Goal

Re-introduce the **useful intent** from PR #235 while treating PRs **#209** and **#211** as the non-negotiable baseline for CI behavior.

The result should preserve the current carefully tuned fast path while adding the workflow structure improvements PR #235 was aiming for.

## Baseline to preserve

These are the invariants from the tuned CI shape. Do **not** regress them.

### 1. Fast PR feedback beats release-like fidelity in ordinary CI

PR CI should stay optimized for fast validation, not for producing release-grade binaries everywhere.

- Keep the **path filter gate** in `ci.yml` so docs-only and UI-only changes do not trigger the full backend matrix.
- Keep ordinary PR validation scoped around the smallest work that still proves correctness.
- Do not turn CPU/macOS PR lanes back into broad release-style rebuilds unless there is a specific measurement-backed reason.

### 2. Restore-only GPU caches in PR CI

The design in `warm-caches.yml` is intentional.

- PR jobs in `ci.yml` should **restore** warmed GPU caches.
- `warm-caches.yml` should remain the **single writer** for main-scoped GPU caches.
- Do not let PR CI save large GPU cache artifacts back into PR-scoped cache storage.
- Preserve cache pruning and explicit key management in `warm-caches.yml` / `gpu-warm-cache-job.yml`.

### 3. Keep the slim CI GPU shape

The tuned GPU lanes are intentionally not release-shaped.

- Preserve the **slim** CUDA cache/input shape in CI: `arch89`, `fa-off`, pinned llama SHA.
- Preserve the **slim** ROCm CI shape: representative `gfx1100` warm/restore path.
- Preserve the distinction between **slim PR-validation artifacts** and **fat release artifacts**.
- Do not move release-grade GPU arch matrices into normal PR CI.

### 4. Keep release-specific behavior in release workflows

Release artifact production belongs in `release.yml`, not in ordinary PR CI.

- Release builds should remain the place where we build the full shipping artifact shape.
- Release-only settings such as full GPU arch matrices and CUDA `FA_ALL_QUANTS=ON` must stay release-only.
- Do not blur the boundary between “fast CI proof” and “ship-ready release artifact.”

### 5. Preserve the reasoning encoded in the scripts

Do not undo the script-level tuning that was added to support CI performance and determinism.

- Keep `scripts/build-linux.sh` support for `MESH_LLM_LLAMA_PIN_SHA`.
- Keep `MESH_LLM_CUDA_FA_ALL_QUANTS=off` as a **CI-only** opt-out, never the release default.
- Keep the build-script distinction between CI-friendly shape and release-friendly shape.

## What PR 235 was trying to achieve

Do preserve these ideas from PR #235:

1. **Separate build/package work from heavier inference smoke work.**
2. **Reuse packaged artifacts** instead of rebuilding the same Linux inference binaries again for smoke validation.
3. **Gate release publication on successful inference smokes.**
4. **Upload platform artifacts** from non-CPU lanes where useful.
5. **Normalize AMD/ROCm naming** through aliases if that helps readability and future maintenance.

That is the spirit worth keeping.

## What PR 235 got wrong and must not be copied forward

Do **not** port these regressions from PR #235:

1. Reverting tuned CI lanes back to broad **release builds/tests** just to support artifact packaging.
2. Losing the practical distinction between **CI validation shape** and **release shape**.
3. Smuggling release-like behavior into PR CI merely because a follow-up smoke job wants a tarball.
4. Weakening or sidelining the explicit cache strategy established in `warm-caches.yml`.
5. Treating artifact packaging as permission to rebuild expensive things twice in different forms.

## Desired end state

Implement the following end state.

### A. `ci.yml` should keep its current tuned execution model

Keep:

- `changes` path-filter gating.
- Restore-only GPU cache consumers.
- Slim GPU build inputs for PR validation.
- No PR-side cache writes for warmed GPU artifacts.

If current CPU/macOS lanes have drifted away from the original 209 intent, use 209 as the reference point for what “fast ordinary CI” should mean.

### B. Add artifact handoff without forcing release-shaped builds

Recover PR 235's artifact-reuse idea, but do it in a way that does not force ordinary CI lanes into release-grade rebuilds.

Concretely:

- Build/package the **same artifact shape already justified for that CI lane**.
- If a smoke job only needs to consume a previously built binary, upload that binary and the required llama.cpp executables as CI artifacts.
- Do **not** change the producer lane from debug/slim to release/fat just because artifact upload is convenient.

For CPU/Linux inference-smoke reuse in PR CI:

- The producing job should emit exactly the binary shape the lane is already meant to validate.
- The downstream smoke job should download and stage those binaries, then run the smoke scripts.

### C. Keep inference smokes as a separate follow-up job where it helps

This is the strongest idea in PR 235 and should be preserved.

- A separate `inference_smoke_tests` job in `ci.yml` is good **if** it consumes artifacts from the producer build job instead of triggering another meaningful rebuild.
- The smoke job should own:
  - model cache restore/download
  - real inference smoke
  - Python/OpenAI compatibility smoke
  - split-mode smoke
  - MoE split + mesh smoke
- The producer build job should own only the work needed to create the binary payload.

### D. Release gating should stay in `release.yml`

PR 235 was right to gate publish on release-smoke success.

Implement this pattern in `release.yml`:

- Linux release build produces both:
  - release bundles under `dist/`
  - a Linux inference binary artifact for downstream smoke testing
- A release-only `inference_smoke_tests` job downloads the Linux inference binaries and runs the real smoke suite.
- `publish` depends on:
  - all release artifact build jobs
  - release inference smoke success

This keeps release validation strong without forcing PR CI to become release-shaped.

### E. AMD naming aliases are acceptable, but only as aliases

The `release-build-amd` / `release-bundle-amd` aliases are fine if they improve wording consistency.

Rules:

- Keep them as thin aliases over the ROCm recipes.
- Do not rename underlying artifact semantics in ways that break existing expectations unless you intend a broader migration.
- Keep artifact names stable unless there is a clear user-facing reason to change them.

## File-by-file instructions

### `.github/workflows/ci.yml`

1. Keep the existing `changes` job and its path-filter behavior.
2. Keep the current warmed GPU cache restore flow and key discipline.
3. Preserve the slim CI CUDA/ROCm inputs and do not widen them to release defaults.
4. Introduce or preserve a separate `inference_smoke_tests` job that:
   - depends on the Linux producer job,
   - downloads prebuilt Linux binaries,
   - stages them into the expected paths,
   - runs the smoke suite.
5. If the Linux producer job currently builds a release binary only because of PR 235, change it back to the tuned CI shape from 209 before packaging artifacts.
6. Package/upload only what the downstream smoke stage needs:
   - `mesh-llm`
   - `rpc-server`
   - `llama-server`
   - `llama-moe-split` when present
7. Keep CLI smoke / cheap boot smoke in the producer lane if they provide fast early failure before the heavier downstream inference stage.

### `.github/workflows/warm-caches.yml`

1. Leave this workflow as the single writer for main-scoped warmed GPU caches.
2. Keep both slim and fat warming where currently present.
3. Keep pruning and explicit cache-input hashing.
4. Do not move cache writes back into PR CI.

### `.github/workflows/gpu-warm-cache-job.yml`

1. Preserve the restore-short-circuit-build-save pattern.
2. Preserve verification of restored/saved binaries.
3. Keep this as reusable cache-warm plumbing, not as a PR CI producer.

### `.github/workflows/release.yml`

1. Keep release artifact builds for CPU/macOS/CUDA/AMD(Via ROCm)/Vulkan here.
2. Keep a separate `inference_smoke_tests` job in release that consumes Linux inference binaries from the release build.
3. Keep `publish` gated on release smoke success.
4. Do not move release publish logic into `ci.yml`.
5. Using `gh release` shell logic is acceptable if it is working and clearer than the old action, but that is secondary to preserving the workflow shape.

### `Justfile`

1. Keep any AMD aliases as wrappers around ROCm recipes.
2. Do not collapse slim-vs-fat behavior into Justfile aliases alone; the workflow files must continue to express which artifact shape they want.
3. Preserve release recipes as release-oriented, not PR-CI-oriented.

### `scripts/build-linux.sh` and `scripts/build-linux-rocm.sh`

1. Keep pinned llama SHA support for deterministic cache correctness.
2. Keep CI-only opt-outs and assertions that document why slim CI builds are safe.
3. Do not remove the warnings that release builds must keep the safer/full settings.

## Recommended implementation order

1. **Reset the mental baseline**
   - Treat 209/211 as the target shape.
   - Diff current `ci.yml` against that baseline and identify where 235-style changes reintroduced release-like work into PR CI.

2. **Recover the fast PR producer shape**
   - Revert CPU/macOS PR build behavior to the tuned CI shape if it has drifted.
   - Preserve slim GPU restore/build behavior.

3. **Layer in artifact handoff**
   - Add artifact packaging/upload to the producer lanes without changing their build profile or widening their backend shape.

4. **Keep heavy inference checks downstream**
   - Wire `inference_smoke_tests` to consume producer artifacts.
   - Ensure it performs no unnecessary rebuild of `mesh-llm` or llama.cpp.

5. **Finalize release gating**
   - Keep release binary production in `release.yml`.
   - Keep publish blocked on release smoke success.

6. **Preserve GPU cache boundaries**
   - Confirm `ci.yml` only restores warmed caches.
   - Confirm `warm-caches.yml` remains the sole cache writer.

## Validation checklist

An implementation is only correct if all of the following are true:

- PR CI still skips expensive backend work on docs-only changes.
- UI-only changes still avoid the full backend/GPU matrix.
- PR CI does not write warmed GPU caches.
- GPU PR lanes still consume slim warmed caches.
- CPU/Linux inference smokes consume uploaded binaries instead of rebuilding the same payload.
- Release workflow still builds shipping artifacts separately from PR CI.
- Release publish is gated on successful release inference smokes.
- No step reintroduces a duplicate build that 209/211 intentionally removed.
- No step widens slim CI GPU inputs into release defaults.
- No release-only safety setting is silently disabled for shipping artifacts.

## Short version

The safe merge of these ideas is:

- **Keep 209/211's fast CI mechanics exactly in spirit.**
- **Adopt 235's artifact handoff and downstream smoke-job structure.**
- **Keep release-grade artifact production and publish gating in `release.yml`.**
- **Never pay for release-shaped builds in ordinary PR CI unless there is a measured reason.**
