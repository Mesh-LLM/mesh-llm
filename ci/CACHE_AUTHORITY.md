# Cache authority by runner decision

This source-level decision record for #2784 describes the checked-in selector
and cache action. It is not a live configuration readback,
a provider ACL, a renewed exception, or approval to enable a runner. The
repository-wide PR exception expired on 2026-09-14 UTC. The exact-ref canary
path is a separate control and remains subject to independent approval and
qualification before use. Protected `@main` PR workflows may differ from a
candidate branch.

The selector returns three distinct permissions: Depot build-tool remote cache
(`D`), the GitHub Actions cache API (`G`), and restoration of the trusted
hosted compiler seed (`S`). These are caller permissions, not evidence that a
backend was reached or that a cache entry is trustworthy. Rows assume valid
boolean inputs, the canonical repository and an eligible role unless stated.
The selectors' generic labels do not override a role's separate hosted,
hardware, authority or publication restriction.

| Event and trust context | Selected provider under the stated condition | D | G | S |
| --- | --- | --- | --- | --- |
| Same-repository direct PR, expired gate, no exact canary | GitHub-hosted | false | true | true |
| Same-repository direct PR, matching configured canary ref after expiry | Depot | false | false | false |
| Fork PR or non-merge PR ref, with a valid merge-ref canary configured | GitHub-hosted | false | true | true |
| `pull_request_target` or dispatch declaring PR origin | GitHub-hosted | false | true | true |
| Trusted-main push or manual dispatch, eligible role and Depot enabled after expiry | Depot | false | false | false |
| Trusted-main push with the Depot gate disabled, or manual dispatch with both opt-ins disabled | GitHub-hosted | false | true | true |
| Any selector caller with `force_hosted=true`, including full release | GitHub-hosted | false | true | true |
| Historical same-repository PR or trusted-main Depot selection while the bounded exception and its gate were active | Depot | false | true | false |
| Malformed boolean or canary ref | No decision; selector fails | n/a | n/a | n/a |

The historical row describes source behavior before expiry, not permission to
repeat it. The selector does not verify a human approval for the canary ref;
that remains an independent prerequisite. The Actions-cache API on Depot was
observed to use a shared
cross-branch namespace; it is not an isolation or provenance boundary. Do not
promote historical shared entries into trusted inputs based on key prefixes or
checksums stored with those entries. A separate approved namespace/access
boundary and provenance evidence are needed before trusted consumption.

The `configure-sccache-gha` action can reduce a granted `G` permission: direct
PRs, `pull_request_target`, and dispatches whose event payload identifies a PR
use job-local disk for sccache; explicit job opt-out, denied native cache, or
a present-but-denied Depot endpoint also select disk. If native access is
permitted, endpoint/token availability and server startup still determine
whether a trusted job obtains a remote tier. Other `actions/cache` consumers
have their own event, key and scope rules; the sccache decision does not
silently govern them. A `true` seed flag only permits the exact hosted seed
restore where that action is actually called; it does not prove a cache hit.

Control and protected-authority jobs remain hosted. Product automation remains
hosted pending #2786 qualification; hardware GPU smokes use their explicit
hardware path, and credential-bearing release/publication roles retain their
own hosted restrictions. Their cache permissions and any service-side access
must be reviewed per caller, not inferred from a neighboring selector row.
macOS startup and ARM64 release-smoke callers apply the selected cache
permissions. Local fixture success does not verify provider isolation,
credentials, live runner identity, actual cache mode, artifact provenance, or
rollback. Record those separately in the approved #2789 qualification evidence.
