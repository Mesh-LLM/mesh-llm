# mesh-llm-events

`mesh-llm-events` owns the typed event contract shared by the mesh runtime,
CLI, SDK-facing embedded runtime, and terminal UI.

The crate intentionally does not render anything. It defines the structured
values that runtime code can emit and that presentation layers such as
`mesh-llm-tui` can render as pretty terminal output, TUI dashboard state, or
JSONL records.

## API Shape

- `LogFormat` selects pretty terminal output or JSONL.
- `OutputEvent` is the structured runtime event taxonomy.
- `RuntimeStatus`, `DashboardSnapshot`, and related dashboard row types are the
  shared status model consumed by the TUI.
- `DashboardSnapshotProvider` lets runtime code provide periodic dashboard
  snapshots without depending on a renderer.

Rendering, progress bars, alternate-screen handling, and terminal control stay
in `mesh-llm-tui`.

## Logging presentation boundary

Canonical request lifecycle events use the production `OutputEvent` emitter.
The projection is intentionally operational: JSONL and TUI records carry a
stable event name and level plus bounded local request/event IDs, replay
channel/sequence, terminal outcome, status, duration, and numeric token counts
when present. Optional payload artifacts, prompts, completions, credentials,
URLs, local paths, and free-form error detail are excluded. These local IDs are
not network or telemetry fields; the trusted-local ledger remains the source
for details, artifacts, replay, retention, and audit operations.

The event contract does not expose the trusted-local request ledger or replace
its detail, artifact, replay, retention, or audit APIs. For the operator
workflow, see [the repository logging guide](../../docs/LOGGING.md); for the
stdout/stderr contract, see
[the TUI event reference](../mesh-llm-tui/src/output/EVENTS.md).
