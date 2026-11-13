# Generation wire characterization

`generation-wire-main.json` was captured before the generation-boundary migration
on main `3947de0cbc73fe5cbcaae548b297f03a401532d7`. Its 38 cases exercise the real
HTTP router with a deterministic recording backend: Chat Completions, Anthropic
Messages and token counting, Responses, and the separate SystemOne operation.

The fixture records backend inputs/context, HTTP status/headers, JSON bodies and
the ordered SSE event sequence, including tool argument fragments, usage after
finish, and errors. Client nonce and request ID are fixed. Frontend-created
IDs, wall-clock timestamps, and validated Responses profile padding are
normalized; scripted IDs, tool IDs and semantic payloads remain exact.
Content-Length is omitted because it describes the unnormalized serialization. Existing lifecycle and wrapper tests cover cancellation and
terminal-event ownership separately.

Run the contract with:

```sh
cargo test -p skippy-inference-api --test generation_wire_characterization
```

The test's `UPDATE_GENERATION_WIRE_FIXTURES` switch exists to capture an explicitly
reviewed behavior baseline. Regenerating the golden file from a refactored backend
is not evidence of compatibility. Keep this fixture unchanged for the migration;
an upstream behavior fix requires a separately recorded baseline adjustment.

## Upstream baseline adjustments

Main commit `b8b0da3439ddaa27d6e42639286c119fa61c0429` (#2614) folds inline
Anthropic system turns into the leading system message and revises the invalid
role diagnostic. Accordingly, `messages-inline-system` now records the leading
`first\ninline` system message followed by the user turn, and `messages-bad-role`
uses that commit's diagnostic. These two adjustments were checked against the
upstream translator and its passing unit tests in an unrefactored API checkout;
the other 36 cases are unchanged. The fixture was not regenerated from the
migrated backend.

The Responses tool-policy checkpoint intentionally changes only the
`responses-tools` case: its request declares `alpha`, while the scripted worker
also calls undeclared `beta`. The original ingress now rejects that entire
terminal output with `tool_choice_violation` instead of exposing an executable
undeclared call. The backend request, usage, headers, and other 37 cases are
unchanged. This is an enforcement fix, not a compatibility-neutral migration.

The Responses no-truncation correction deliberately changes only the backend
request in `responses`, `responses-stream`, and `responses-tools`: each now
contains `mesh_compact: false`. That frontend-owned marker prevents the shared
backend compactor from silently shortening accepted Responses input. No other
fixture field or case changed for this correction.

The Responses stream-obfuscation checkpoint retains the default-enabled request
in `responses-stream` and leaves the historical golden unchanged. Before
normalizing IDs or timestamps, the characterization checks that each eligible
Responses delta carries 1–256 URL-safe ASCII padding characters and that its raw
JSON payload length is a multiple of 256 bytes. Only that validated profile
`obfuscation` field is then removed for semantic comparison. Non-delta events
must not contain the field. This is the provider's padding profile, not a
specification-mandated algorithm or a privacy guarantee; dedicated stream tests
also cover disabling padding, byte-budget admission, and randomness failures.
