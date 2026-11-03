# Runner role readiness inventory

This is a source inventory for the #2780 draft stack at commit
`51acbae105165a23e2bc40b3ec5e622ad7a22086`. It describes candidate workflow
wiring, not observed runner identities, runner-group membership, permissions,
capacity, cache isolation or an authorization to enable a role. Protected
`@main` PR jobs may execute a different workflow revision. Reconcile this table
against the final merged source and authenticated live readback before using it
for a rollout. The qualification and measurement work remains in #2789.

The selector emits runner labels and cache permissions together. A
`force_hosted` input wins over provider requests. Where a row says “selector,”
the trusted-main gate, manual request and PR trust inputs still apply; a
configured label is not evidence of a running machine. The repository-wide PR
exception expired on 2026-09-14 UTC. A separate exact-ref canary path is also
present in source and must not be treated as approved or enabled without its
own evidence. Fork, target-event and PR-origin dispatch refusals remain part
of the selector contract. Every selector-based role has a GitHub-hosted
fallback in its projected labels.

| Role | Source placement and event boundary | Preparation and required inputs | Readiness / fallback |
| --- | --- | --- | --- |
| Planners, lane summaries and control | Hosted Linux; PR planning uses protected policy and candidate source identity. | `prepare-ci-controller`; protected catalogs and bounded plan/summary inputs. | Hosted path exists in source; current live execution and capacity not verified here. |
| Protected authority automation | Closed hosted Linux x64/ARM64, macOS ARM64 and Windows x64 map. | `protected-automation-artifact.yml` freezes an admitted protected source; `protected-clean` producer exports source, artifact, digest and platform identity. | Remains hosted; artifact transport and reviewed audit cutover are separate #2786 gates. |
| Product automation producer | Closed hosted Linux x64, macOS ARM64 and Windows x64 map. | `product-automation-artifact.yml` produces `product-clean` automation at the declared product source; consumers restore and verify its identity. | Hosted producer exists; a Depot producer profile or cross-provider transport is not qualified. |
| Semantic Quality builds | Central selector admits trusted-main/manual paths through the existing gate and eligible PR paths through the protected PR rules. | Quality slice and pinned Linux CPU image; protected and product inputs remain distinct. Independent controls and the sentinel have their own placement. | Prepared in source; actual labels, runner-group admission and matched performance evidence remain unverified. Hosted fallback is retained. |
| Rust test shards | Trusted-main Depot gate is explicitly false; PR selection remains subject to the central protected PR rules. | Pinned Linux CPU image, verified static ABI and product automation; bounded isolated package test batches. | Hosted is the current trusted-main source decision; the libclang image and main-repair proposals must be reconciled before claiming freshness coverage. |
| Linux host and non-CPU native runtime | Existing central selector supports gated trusted-main/manual placement and eligible protected PR paths. | Host uses a pinned CPU image, verified UI and product automation. Runtime rows bind backend-specific pinned images, toolchain epochs and product automation. | Prepared source path only; verify exact provider/OS, image, runner group, artifact delivery and resource capacity for each role. |
| Linux CPU native runtime / shared cache producer | The separate CPU policy explicitly sets `force_hosted: true`. | Pinned CPU image and source-bound product automation; CPU runtime cache is a separate consumer/producer boundary. | Retained hosted; do not infer Depot readiness from non-CPU rows. |
| Static ABI and Linux native SDK | Central selector receives trusted-main gate, protected PR inputs, manual intent and hosted override; SDK forwards these and the frozen source to nested ABI. | Pinned CPU/toolchain profile, protected authority and source-bound artifact identities. | Candidate source preserves hosted fallback; qualify ABI/SDK artifact transport and each architecture independently. |
| UI, website and Linux product slices | Trusted-main Depot gate is explicitly false in these slice selectors; protected PR selection is still governed by the central selector. | UI and web use their declared Node/image setup; product jobs restore immutable host/runtime/UI inputs and verified automation. | Hosted is the current trusted-main source decision. No broader role readiness is inferred. |
| macOS and Windows lanes / platform checks | Trusted-main Depot gate is explicitly false in the platform selectors; protected PR selection remains independently bounded. | Native OS/toolchain bootstrap, closed matrix rows and platform-specific protected/product automation. Swift/native SDK preserve their explicit target resolution. | Hosted trusted-main paths retained. Native macOS cache startup correction is separately proposed in #2825; actual platform execution and cache policy need qualification. |
| Hardware GPU smokes | Dedicated `gpu-nvidia` path rather than a generic Depot label. | Hardware/device checks and the selected verified runtime/model artifacts. | Hardware eligibility and capacity are separate; no expansion follows from CPU or compile-only runtime results. |
| Full release and publication | Candidate #2783 source forces the full-release selector and nested SDK calls hosted; the separate rolling-preview CUDA override is unchanged. | Release metadata freezes source; hosted publication and secret-bearing jobs retain their existing dependencies and native artifact checks. | Full-release Depot is not enabled by the reserved manual input. Release execution, signing and publication remain separate authorized operations. |

## Cache and evidence boundaries

`allow_depot_remote_cache` remains false in the checked-in selector. Native
GitHub cache permission depends on the selected provider and trust context;
compiler-cache configuration further reduces PR and PR-origin-dispatch jobs to
job-local disk. Other cache consumers have their own exact-key and event
policies. Neither a new namespace nor workflow-only fixtures establish a server
ACL or provenance for historical shared Depot entries.

The source profiles above name preparation paths, not measured machine
resources. Memory, disk, quota, aggregate concurrency, group workflow
restrictions, current variables and cache backend identity require authenticated
readback and matched observations. #2787 owns capacity and admission; #2784
owns cache trust; #2786 owns transport; #2789 owns role-specific qualification
and rollout. Do not label any row active or qualified solely from this table.

The planner validates that the three configured platform limits sum to no
more than `total_max_workers`, but those values are forwarded independently to
multiple matrices. For example, macOS host, native-runtime, platform-check and
Swift SDK calls each receive `macos_max_parallel` and can have overlapping
dependencies. The planner's arithmetic therefore does not establish an
aggregate per-run or organization-wide worker ceiling. #2787 must model and
enforce aggregate admission, including overlapping PRs, main revisions and
releases, before such a ceiling is claimed.
