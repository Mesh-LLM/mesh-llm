# Local Node Reputation

Local node reputation is a process-local routing safety mechanism. It helps a
node avoid peers or local serving targets that recently timed out or returned
unavailable, while preserving mesh availability when there is no healthy
alternative.

This is not a mesh trust system. Reputation is not gossiped, not persisted as a
network-wide score, and not used to prove peer identity, model honesty, owner
attestation, or release provenance.

## Goals

- Prefer targets that have recently completed requests successfully.
- Temporarily cool down targets that returned retryable transport or
  availability failures.
- Keep request routing available when every candidate is cooling.
- Expose small operator counters in `/api/status` without leaking request
  content or peer-private state.
- Keep all policy local until the project has a reviewed design for
  cross-node reputation.

## Non-goals

- No gossip field or protocol change.
- No distributed trust or peer scoring.
- No punishment for client-side disconnects, request rejection, or known
  context-fit failures.
- No persistent ban list.
- No replacement for owner identity, release attestation, or future output
  verification.

## Signals

The local proxy records an outcome after a target attempt:

| Outcome | Reputation effect |
|---|---|
| Success | Clears active cooldown and gradually recovers any local penalty |
| Timeout | Adds a retryable failure cooldown and local penalty |
| Unavailable | Adds a retryable failure cooldown and local penalty |
| Context overflow | No reputation penalty |
| Rejected | No reputation penalty |
| Client disconnected | No reputation penalty |

Timeouts and unavailable results are treated as retryable health signals because
the next target may still serve the same request. Context overflow and explicit
rejections are admission or request-fit outcomes, so penalizing the peer would
mix routing health with request semantics.

## Routing Behavior

For a model-specific candidate list, routing first removes cooling targets when
there is at least one routable alternative. Remaining candidates are ordered by
local penalty, so a peer that has recovered from cooldown can still sit behind
clean peers until it proves itself again.

If every candidate is cooling, explicit model routing preserves availability by
returning the original candidates instead of making the model unreachable. Auto
routing can use the stricter eligible set so it may pick another model/target
instead of immediately retrying a known cooling target.

Successful attempts rebuild reputation gradually. One success clears the active
cooldown; repeated successes remove the residual penalty.

## Status Surface

`GET /api/status` exposes local counters under:

```text
routing_affinity.target_reputation
```

The current counters are:

| Field | Meaning |
|---|---|
| `penalized_targets` | Number of local model+target entries that still carry a penalty |
| `routes_penalized` | Number of route orderings that moved penalized candidates behind cleaner alternatives |

These counters are local to the node that served `/api/status`. They should be
read as operator diagnostics, not as mesh-wide health truth.

## Testing Model

Unit tests cover the target-health state machine directly and the routing
integration through `AffinityRouter`. The integration tests include a simulated
three-peer candidate list: a peer starts first in route order, receives a local
unavailable outcome, is avoided on the next request, and returns to normal
routing after a successful attempt clears the active cooldown. Lower-level
target-health tests cover residual penalty ordering after cooldown.

This simulation is intentionally local. It proves how the routing layer consumes
local reputation without requiring a live multi-node mesh or adding protocol
fixtures for a behavior that is not gossiped.

## Future Work

Any cross-node reputation design needs a separate proposal. Before gossiping or
sharing any reputation signal, the design should answer:

- What exact evidence is being shared.
- How peers prevent spoofing, Sybil amplification, or retaliation loops.
- How private meshes can opt in or out.
- How old nodes ignore the signal safely.
- How status and API surfaces distinguish local observation from remote claims.
