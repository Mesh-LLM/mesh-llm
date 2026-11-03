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
| Quality role and sentinel | Planner inputs and checked-in Quality callers project the expected 4/8-worker labels for main, control, expired PR, fork and PR dispatch; the separate sentinel fixture requires its exact ref. | Synthetic selector and wiring fixtures; no sentinel activation or observed job. |

The composed caller checks cover Linux static ABI, the native-SDK nested
static-ABI path, the three platform host paths, Linux runtime's CPU exception
and Quality's normal and sentinel selectors, including selected planner rows
and forwarding. A local mutation fixture verifies that the same wiring checks
reject a dropped nested hosted override and a disabled release SDK override.
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
