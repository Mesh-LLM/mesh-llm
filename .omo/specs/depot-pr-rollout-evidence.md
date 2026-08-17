# Depot PR bounded-exception rollout evidence

Status: candidate and hosted rollback complete

This record captures the 2026-08-15 validation of the time-bounded,
same-repository PR exception defined in `ci/DEPOT_PR_RISK_EXCEPTION.md`. It is
evidence that the protected routing and unchanged build graph work; it is not
evidence of cache isolation. The previously recorded sentinel still proves
that Depot's Actions-cache authority crosses the PR/main trust boundary.

## Fixed test identity

- Pull request: `#1335`
- Head SHA: `631df713d86cd12e89e7e4e7d75b5360ca1de81e`
- Merge ref: `refs/pull/1335/merge`
- Attempt 1: GitHub-hosted baseline
- Attempt 2: exact ref/SHA-approved Depot candidate
- Attempt 3: gate-absent GitHub-hosted rollback

The source SHA is identical in every row below. `E` is one representative
ordinary executor job and `S` is the stable lane-summary job.

| Attempt | Lane / workflow run | Relevant job IDs (`E` / `S`) | Protected base SHA |
| ---: | --- | --- | --- |
| 1 | Quality `31856900751` | `94943336698` / `94944531916` | `19dfd731a00454161a486117eca28e9c1bce3dfb` |
| 1 | Website `31856900758` | `94943463706` / `94944272901` | `19dfd731a00454161a486117eca28e9c1bce3dfb` |
| 1 | Linux `31856900875` | `94943343184` / `94948475940` | `19dfd731a00454161a486117eca28e9c1bce3dfb` |
| 1 | macOS `31856900858` | `94943361090` / `94946328625` | `19dfd731a00454161a486117eca28e9c1bce3dfb` |
| 1 | Windows `31856900759` | `94943335837` / `94949094847` | `19dfd731a00454161a486117eca28e9c1bce3dfb` |
| 2 | Quality `31856900751` | `94949254565` / `94949627063` | `19dfd731a00454161a486117eca28e9c1bce3dfb` |
| 2 | Website `31856900758` | `94949254364` / `94949620720` | `19dfd731a00454161a486117eca28e9c1bce3dfb` |
| 2 | Linux `31856900875` | `94949255563` / `94951376609` | `19dfd731a00454161a486117eca28e9c1bce3dfb` |
| 2 | macOS `31856900858` | `94949261037` / `94952506197` | `19dfd731a00454161a486117eca28e9c1bce3dfb` |
| 2 | Windows `31856900759` | `94949257612` / `94960217656` | `19dfd731a00454161a486117eca28e9c1bce3dfb` |
| 3 | Quality `31856900751` | `94965049631` / `94966569930` | `1854b527c417770ed4c651c7afdd398a7bea5fa0` |
| 3 | Website `31856900758` | `94965041540` / `94966339212` | `1854b527c417770ed4c651c7afdd398a7bea5fa0` |
| 3 | Linux `31856900875` | `94965140517` / `94969537370` | `1854b527c417770ed4c651c7afdd398a7bea5fa0` |
| 3 | macOS `31856900858` | `94965516732` / `94968237314` | `1854b527c417770ed4c651c7afdd398a7bea5fa0` |
| 3 | Windows `31856900759` | `94965087121` / `94968895592` | `1854b527c417770ed4c651c7afdd398a7bea5fa0` |

The protected default branch advanced before attempt 3. A path-limited diff
between the two protected revisions changed only `.github/workflows/release.yml`;
none of the PR-referenced lane/slice/action/planner files changed. The complete
job-name sets were also exactly equal across attempts. The candidate was armed
only with `DEPOT_PR_RUNNERS_ENABLED=true` plus the exact approved ref and SHA.
All three bounded-exception variables were deleted immediately after attempt 2
completed and before attempt 3 was requested.

## Depot candidate result

All five attempt-2 workflows and their stable `PR / <lane>` summaries completed
successfully. Forty-one executed eligible ordinary rows succeeded on Depot;
two additional Depot-selected matrix rows were expected skips. Hosted control,
summary, credential-bearing smoke, and hardware-exception jobs retained their
documented providers. The approved `gpu-nvidia` CUDA smoke also succeeded.

Every inspected Depot executor ran the protected isolation audit before
checkout, selected `allow_native_github_cache=true`, and kept
`allow_depot_remote_cache=false`. Cache hits were observed on Linux, Website,
and Windows rows. Those hits are the intentional cross-branch acceleration
accepted by the exception, not a provenance or correctness claim.

Depot macOS jobs do not expose GitHub-hosted `ImageOS`/`ImageVersion` values.
The Metal, portable, unit, host, product, and Swift-input rows nevertheless
passed using the reviewed tool-version fingerprint fallback. The resolved
Metal/unit epoch was
`runner-macOS-ARM64-native-1d61d7c9cf228ea257cd32496f60045c23a10cc70feaef902473d8b786873a57`.

## Provider comparison

Schema-v3 reports were generated from the attempt-1 and attempt-2 raw job data
under `/tmp`; raw reports are intentionally not checked in. Queue values below
are job runner-queue p95 for common ordinary executor families.

| Lane | Hosted / Depot jobs | Queue p95 hosted → Depot | Queue signal |
| --- | ---: | ---: | --- |
| Quality | 5 / 5 | 58.0s → 1.0s | favorable, but full comparison unclassified |
| Website | 2 / 2 | 170.6s → 1.0s | insufficient sample |
| Linux | 16 / 16 | 163.25s → 1.0s | favorable, but full comparison unclassified |
| macOS | 7 / 7 | 407.5s → 590.0s | `rollback` (queue contamination) |
| Windows | 11 / 11 | 2369.5s → 3419.5s | `rollback` (queue contamination) |

Provider sets were disjoint and job families, OS, architecture, source, plan,
and toolchain policy matched. Cache authority and actual hit/miss state did
not: the Depot candidate intentionally used repository-wide cross-branch cache
entries, while hosted PR cache access remained ref-scoped. Execution timings
are therefore cache-confounded and are not used for a provider-speed or
eligibility claim. Runner queue occurs before cache restore and is retained as
a capacity observation. Attempt 2 was a rerun, so workflow wall/queue timing is
also intentionally excluded.

The Windows workflow preserves `max-parallel: 1` for both native-runtime and
product matrices. GitHub therefore leaves sibling matrix rows queued even when
Depot Windows runners are visibly idle. The collector lacks dependency-ready
timestamps and counts that intentional serial wait as runner queue. The formal
capacity classification remains `rollback`; the idle-runner view must not be
misreported as provider starvation. macOS also remains formally contaminated
and requires more capacity evidence before any performance claim.

## Hosted rollback evidence

Attempt 3 ran after all three approval variables were deleted. Quality,
Website, Linux, macOS, and Windows all completed successfully, including their
stable `PR / <lane>` summaries. The complete job-name sets were exactly equal
between attempts 2 and 3 for each workflow. Provider choice was the only
routing difference: attempt 3 used `ubuntu-24.04`, `macos-15`, and
`windows-2022`, plus the unchanged approved `gpu-nvidia` smoke exception, and
contained no `depot-*` labels.

The rollback conclusions matched the candidate: Quality 9 success/3 expected
skips, Website 6 success, Linux 32 success/5 expected skips, macOS 19
success/1 expected skip, and Windows 19 success. No source, plan, matrix,
dependency, artifact, command, test, or stable-summary change was required.
Deleting the variables therefore proved the documented one-policy-change
rollback on the identical SHA and graph.

## Decision

The candidate proves that the exact ref/SHA approval path can run the eligible
five-lane executor denominator on Depot without changing plan membership,
commands, artifacts, tests, or stable summaries. It does not remove the shared
cache risk and does not justify permanent or fork activation. Each
same-repository revision still requires an explicit maintainer decision under
the expiry and controls in `ci/DEPOT_PR_RISK_EXCEPTION.md`; the variables stay
absent between approvals.

Quality and Linux provide favorable queue observations, not normalized
execution-speed evidence. Website needs a larger sample, and macOS/Windows
remain formal rollback classifications. These results do not weaken the
automatic 2026-09-14 expiry, the
no-secret/credential exceptions, or the requirement for provider-enforced
per-PR cache authority before making the policy permanent.
