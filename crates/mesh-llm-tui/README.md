# mesh-llm-tui

`mesh-llm-tui` owns mesh-llm's terminal presentation layer: structured runtime
events, JSON/pretty log formatting, terminal progress lines, and the interactive
dashboard renderer.

The crate is intentionally separate from host runtime orchestration. Host-side
code emits `OutputEvent` values and lets this crate decide whether to render
plain text, JSON, progress indicators, or the ratatui dashboard.
