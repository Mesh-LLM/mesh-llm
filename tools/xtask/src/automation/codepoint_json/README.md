# Lossless automation JSON

Replay export and workload evidence share this codepoint-based JSON owner.
The parser and string emission come from the verified replay implementation.
Strings and object keys retain lone escaped surrogates; duplicate reduction
uses decoded codepoints and never converts through Rust `String` or the
String-based `ci_metrics_value::Value`. Numeric tokens alone reuse the existing
exact-number parser, and finite float spelling still uses `float_repr`.

Keep each complete tree lifetime inside its consumer's 64 MiB worker. Parsing,
duplicate replacement, cloning, validation, rendering, error unwinding and final
drop are recursive. Returning a parsed tree to the caller would lose this
execution guarantee. The parser's existing 9,998-container limit is unchanged.

Only JSON representation, parsing and scalar emission are shared. Replay owns
its policy, positive-integer wrapper, diagnostics, single-line layout, ordered
export effects and CLI. Workload evidence owns its PCM policy, identity checks,
two-space layout and file effects. The existing String decoder and planner
writer keep their contracts for all other consumers.
