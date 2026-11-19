# Generation wire characterization

`generation-wire-main.json` was captured before the generation-boundary migration
on main `3947de0cbc73fe5cbcaae548b297f03a401532d7`. Its 38 cases exercise the real
HTTP router with a deterministic recording backend: Chat Completions, Anthropic
Messages and token counting, Responses, and the separate SystemOne operation.

The fixture records backend inputs/context, HTTP status/headers, JSON bodies and
the ordered SSE event sequence, including tool argument fragments, usage after
finish, and errors. Client nonce and request ID are fixed. Only frontend-created
IDs and wall-clock timestamps are normalized; scripted IDs, tool IDs and payloads
remain exact. Content-Length is omitted because it describes the unnormalized
serialization. Existing lifecycle and wrapper tests cover cancellation and
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
