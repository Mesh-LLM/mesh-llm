# Runner selection acceptance evidence

This is the source-fixture inventory for #2785, not a provider qualification
or a record of a hosted run. The `artifact_workflows::placement` tests feed
the real planner's placement result into the checked-in selector and, for
selected cases, into the compiler-cache action. Fixture dates are synthetic:
they exercise the recorded exception boundary and do not renew the exception.
The table records what the local tests assert; actual runner identity and
effective hosted cache behavior remain unobserved.

| Fixture condition | Local selection and cache assertion | Scope |
| --- | --- | --- |
| Ordinary main, selector gate on | Depot labels; direct Depot cache and native cache denied after expiry. | Generic selector fixture; individual callers may hold their gate off. |
| Ordinary main, gate off | GitHub-hosted labels. | Exhaustive coverage remains selected. |
| Main control change, missing base, or forced-hosted override | GitHub-hosted labels despite the main gate. | Planner placement and selector composed. |
| Manual-full on main with Depot requested | Depot labels when no hosted override applies; otherwise hosted. | Generic selector fixture; caller-specific input forwarding still applies. |
| Same-repository PR before expiry, gate on | Depot labels; the selected compiler-cache fixture remains disk-only for PR execution. | Historical boundary case, not a current approval. |
| PR at expiry, fork, target event, PR-origin dispatch, or gate off | GitHub-hosted labels. | Trust and event boundary fixtures. |
| Expired exact-ref canary | Matching same-repository merge ref may select Depot but both direct Depot cache and native cache are denied. | Selector behavior only; no canary activation is asserted. |
| Wrong canary ref, fork, or PR head ref | GitHub-hosted labels. | Selector refusal fixtures. |
| Unsupported event or malformed boolean | Selector falls back to hosted for an unsupported event; malformed booleans are refused. The planner rejects unsupported event names. | No fallback to a privileged provider. |
| Main host roles, repository gate on | Linux host projects Depot; macOS and Windows host callers keep their main gates off and project hosted labels. A control-change plan projects hosted labels for all three. | Planned host rows, checked-in caller gates and real selector; no hosted execution. |
| Main Linux runtime, gate on | CPU's explicit override projects hosted placement and native-cache permission while the general runtime selector projects a Depot label and denies that cache. | Planned CPU/CUDA rows and checked-in selector projection; no observed runner or cache. |
| Main macOS and Windows runtimes | Both callers keep their main gates off and project hosted labels while retaining runtime artifact names. | Planned runtime rows and checked-in callers; no observed job or artifact. |
| Main platform checks | macOS and Windows check callers keep the main gate off; the actual mapping step maps selector labels to the planned platforms. | Local planner, selector and shell mapping fixture; no observed job. |
| Main Rust, console, website and Swift callers | These callers keep their main gates off and project hosted labels while retaining selected batches, jobs and artifact names. | Local planner and checked-in caller fixtures; no observed job or artifact. |
| Main product composition | Linux, macOS and Windows product callers keep their main gates off and project hosted labels; host, runtime and product artifact names remain bound. | Planned runtime rows and checked-in callers; no observed job or artifact. |
| Quality role and sentinel | Planner inputs and checked-in Quality callers project the expected 4/8-worker labels for main, control, expired PR, fork and PR dispatch; the separate sentinel fixture requires its exact ref. | Synthetic selector and wiring fixtures; no sentinel activation or observed job. |

## Direct selector caller census

This census covers direct workflow steps using `select-ci-runners`, not all
transitive callers, third-party actions, or live execution. `partial` means at
least one composed case above covers the caller; it does not mean its full
event/role matrix is complete. `open` means no composed caller case is recorded
here. All rows still need complete coverage and review.
The census test verifies caller identities and declared core inputs; it does
not independently prove that a row's coverage label is complete.

Current census: 20 direct selector steps; 19 partial and 1 open.

<!-- selector-census:start -->
| Workflow | Job | Step | Coverage |
| --- | --- | --- | --- |
| `ci-linux-host-slice.yml` | `runner_policy` | `policy` | partial |
| `ci-linux-product-slice.yml` | `runner_policy` | `policy` | partial |
| `ci-linux-runtime-slice.yml` | `runner_policy` | `policy` | partial |
| `ci-linux-runtime-slice.yml` | `runner_policy` | `cpu_policy` | partial |
| `ci-macos-host-slice.yml` | `runner_policy` | `policy` | partial |
| `ci-macos-product-slice.yml` | `runner_policy` | `policy` | partial |
| `ci-macos-runtime-slice.yml` | `runner_policy` | `policy` | partial |
| `ci-platform-checks-slice.yml` | `runner_policy` | `policy` | partial |
| `ci-quality-slice.yml` | `runner_policy` | `policy` | partial |
| `ci-quality-slice.yml` | `runner_policy` | `sentinel_policy` | partial |
| `ci-rust-tests-slice.yml` | `runner_policy` | `policy` | partial |
| `ci-ui-artifact-slice.yml` | `runner_policy` | `policy` | partial |
| `ci-web-slice.yml` | `runner_policy` | `policy` | partial |
| `ci-windows-host-slice.yml` | `runner_policy` | `policy` | partial |
| `ci-windows-product-slice.yml` | `runner_policy` | `policy` | partial |
| `ci-windows-runtime-slice.yml` | `runner_policy` | `policy` | partial |
| `native-sdk-artifact.yml` | `runner_policy` | `policy` | partial |
| `release.yml` | `metadata` | `runners` | open |
| `static-abi-artifact.yml` | `runner_policy` | `policy` | partial |
| `swift-sdk-artifact.yml` | `runner_policy` | `policy` | partial |
<!-- selector-census:end -->

The composed caller checks cover Linux static ABI, the native-SDK nested
static-ABI path, the three platform host, runtime and product paths, Linux
runtime's CPU exception and Quality's normal and sentinel selectors, including
selected planner rows and forwarding. A local mutation fixture verifies that
the same wiring checks reject a dropped nested hosted override and a disabled
release SDK override.
The real selector also demonstrates that those changed inputs can select Depot
in a synthetic trusted manual-main scenario. This is not a live release or
proof of other callers.
Other role/caller paths and the complete event-by-role acceptance matrix still
need the same composed proof. The source-only execution-record command can
retain caller assertions alongside a selector trace, but does not authenticate
the selector-to-job binding, observe the runner or cache, or retain an
artifact. Protected `@main` execution, live
runner identity, provider isolation and same-source timing evidence remain
separate requirements.
