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
