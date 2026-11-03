# Runner role readiness inventory

This is a source inventory for the #2780 draft stack. It describes the
candidate workflow wiring checked in alongside this document, not observed
runner identities, runner-group membership, permissions,
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
| macOS and Windows lanes / platform checks | Trusted-main Depot gate is explicitly false in the platform selectors; protected PR selection remains independently bounded. | Native OS/toolchain bootstrap, closed matrix rows and platform-specific protected/product automation. Swift/native SDK preserve their explicit target resolution. | Hosted trusted-main paths retained. macOS cache startup applies the selected permissions; actual platform execution and cache policy need qualification. |
| Hardware GPU smokes | Dedicated `gpu-nvidia` path rather than a generic Depot label. | Hardware/device checks and the selected verified runtime/model artifacts. | Hardware eligibility and capacity are separate; no expansion follows from CPU or compile-only runtime results. |
| Full release and publication | Candidate #2783 source forces the full-release selector and nested SDK calls hosted; the separate rolling-preview CUDA override is unchanged. | Release metadata freezes source; hosted publication and secret-bearing jobs retain their existing dependencies and native artifact checks. | Full-release Depot is not enabled by the reserved manual input. Release execution, signing and publication remain separate authorized operations. |

## Default and nested automation preparation

The checked-in `prepare-automation` default is `image`. On Linux it verifies
the pinned public CPU image before building automation. Direct default callers
are classified as follows; these are source paths, not machine attestations.

| Direct path | Declared preparation boundary | Remaining evidence |
| --- | --- | --- |
| UI stamp, nightly stability and static ABI fallback | Linux job containers name pinned images; the static ABI fallback runs only without a product automation input. | Actual container identity, tools and provider admission. |
| macOS smoke, Swift target and native SDK target | macOS label or target-specific macOS runner selection; the SDK target is restricted to Apple Darwin. | Native tools, cache policy and actual runner identity. |
| Windows unit platform checks, release host and two cache warmers | Windows-specific unit guard or Windows runner label. | Native tools, source/artifact identity and actual runner identity. |

Five local composite actions can lead to this preparation action. Product
composition and native runtime preparation invoke its default only on non-Linux
when no automation executable is already exported. Automation upload forwards
its closed profile and has a Unix image-profile bootstrap fallback. Smoke input
restoration uses the default for non-Linux or a container, and `hosted-bare`
only for bare GitHub-hosted Linux. A bare self-hosted Linux smoke may instead
fall back to `cargo xtool`; the runner's toolchain and cache state must be
independently verified. The Laya CUDA, ROCm and Vulkan jobs prepare a
source-bound cleanup executable before their nested smoke action, but their
hardware, installed tools and group admission are still unverified here.

The source contract inventories these local composite callers and the local
reusable-workflow edges that reach them. Six PR entrypoints call protected
`@main` workflow references, so their actual protected source may differ from
the candidate revision. Runtime shell dispatch, third-party action internals,
live runner state and external admission remain outside this source census.

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
