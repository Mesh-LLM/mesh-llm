# skippy-events

Skippy-owned request identifiers, token usage and inference lifecycle contracts,
shared by standalone Skippy and Mesh.

## Diagnostic output

An embedding product installs a `diagnostics::DiagnosticSink` before starting
serving. Without an installed sink, diagnostics are intentionally discarded;
the library does not install a terminal renderer as a loading side effect.

The current registry is process-wide and last-writer-wins, with no chaining.
It supports one embedding product's output policy per process. Callbacks run
outside the registry lock, must return promptly, and may return output errors.
Diagnostic delivery is independent of tracing subscribers and their filters.

Per-instance routing must be decided during the Skippy lifecycle API extraction,
before this observer API stabilizes. Independent runtimes cannot currently
select separate renderers within one process.
