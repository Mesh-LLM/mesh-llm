# Plan: fast-fail for new requests routed to a dead split stage

## Problem (measured 2026-07-18)

2-node WAN split, remote stage (vast) killed mid-serving:

- **In-flight request:** fails fast, 502 in ~1.8s. Good — bounded by #1011
  (`open_return_sink_once` / `connect_lane_once` read timeouts).
- **New request after peer death:** coordinator still shows `peers:1, serving`
  (stale peer state), routes to the dead stage, and hangs **~30s** before erroring.

## Root cause (confirmed in code)

Three timers interact; the split lane path uses the slow ones:

1. `mesh/heartbeat.rs`: heartbeat runs **every 60s**, `failure_threshold = 2`
   (direct) / `5` (relay-only). A dead direct peer is not declared down / evicted
   for **~2 min**; relay peers ~5 min. Thresholds are deliberately lenient to
   survive relay hiccups — do **not** tighten blindly.
2. New split requests trust that stale peer state and open lanes to the dead
   stage.
3. `skippy-server/frontend.rs`: `connect_lane_once` bounds the ready handshake at
   `LANE_READY_READ_TIMEOUT = 20s`, and `connect_binary_downstream` retries the
   TCP connect `timeout_secs * 2` times × 500ms sleep. Combined worst case ≈ the
   observed ~30s stall.

## Fix (two layers; leave mesh timing alone)

### Layer 1 — shorter first-attempt lane-open deadline (contained, unit-testable)

`crates/skippy-server/src/frontend.rs`

- Distinguish **cold-start / pool warmup** (where a longer wait is legitimate:
  peer still loading) from **steady-state re-dial of a live split** (where a
  healthy peer at 20ms RTT answers in ms).
- Add a separate, short deadline for steady-state `checkout()` reconnects, e.g.
  `LANE_STEADY_CONNECT_TIMEOUT ≈ 3s`, distinct from the existing
  `LANE_READY_READ_TIMEOUT = 20s` used during pool `new()`.
- Plumb it so `connect_lane_once` called from `checkout()` (mid-life) uses the
  short deadline; pool construction keeps the long one.
- Effect: dead stage on a new request fails in ~3s, not ~30s.

Risk: low. Only changes how long a *reconnect* waits. Cold-start unaffected.

Test: unit test with a `TcpListener` that accepts but never sends ready (as in
`persistent_lane_ready_handshake_times_out_for_silent_downstream`) asserting the
steady-state path errors within the short bound.

### Layer 2 — feed lane failures into `network/target_health.rs` (right layer, reuses existing machinery)

`crates/mesh-llm-host-runtime/src/network/*` (routing side)

- `target_health.rs` already exists: `record_outcome(... TargetHealthOutcome::Timeout | Unavailable ...)`
  with a 30s base cooldown (`cooldown_for_failure`), and `eligible_candidates()`
  / `strict_eligible_candidates()` for routing.
- When a split lane connect/handshake to a downstream stage fails, record a
  `Timeout`/`Unavailable` outcome for that **stage target** so the next request
  sees the cooldown and rejects immediately with a clear "stage unavailable"
  instead of re-dialing the corpse.
- Requires wiring: the split coordinator's downstream-stage selection must
  consult `target_health` before opening lanes, and report lane outcomes back
  into it. Today `target_health` is used by the proxy/routing layer, not the
  split lane pool — this is the missing connection.

Risk: medium (touches routing/selection). Needs the live 2-node kill test to
confirm the cooldown actually engages on the split path and that a recovered peer
clears (via `record_reputation_success`).

## Explicitly NOT doing (needs live multi-node validation first)

- Tightening `heartbeat.rs` intervals / `failure_threshold`. Comments warn this
  causes false-positive peer death on relay hiccups (Sydney↔Sydney relay spike to
  10s+ RTT). Out of scope for a fast-fail fix.
- Automatic re-planning onto a replacement worker. Bigger design conversation.

## Validation gate (before committing either layer)

Re-run the 2-node WAN kill test:

1. Start split, confirm serving.
2. Fire a new request → baseline hang time.
3. Kill remote stage.
4. New request must fail in ≤ ~3s (Layer 1) with a clear stage-unavailable error
   (Layer 2), not hang ~30s.
5. Restart remote stage → confirm the target cools back in and serving resumes.

## Sequencing

Bundle with the next dense-model / stage-count proof run (cached builds, ~\$0.30)
so both layers are validated live before landing. Do not commit blind.
