# Order-preserving automation JSON

Replay export and workload evidence share this codepoint-based JSON owner.
The parser and string emission come from the verified replay implementation.
Parsing goes through the order-preserving `ci_plan::document::Json` decoder,
which is backed by `serde_json`, so a lone escaped surrogate (for example
`"\ud800"`) in a string or object key is rejected as invalid input rather than
preserved. Strings therefore hold Unicode scalar values only; duplicate
reduction compares decoded codepoints and never converts through the
String-based `ci_metrics_value::Value`. Integers that fit i64 or u64 are kept exactly;
any other number is converted to f64, so values beyond f64 precision (for example `0.10000000000000001`) are
rounded. Finite float spelling still uses `float_repr`.

Keep each complete tree lifetime inside its consumer's 64 MiB worker. Parsing,
duplicate replacement, cloning, validation, rendering, error unwinding and final
drop are recursive. Returning a parsed tree to the caller would lose this
execution guarantee. The parser's existing 9,998-container limit is unchanged.

Only JSON representation, parsing and scalar emission are shared. Replay owns
its policy, positive-integer wrapper, diagnostics, single-line layout, ordered
export effects and CLI. Workload evidence owns its PCM policy, identity checks,
two-space layout and file effects. The existing String decoder and planner
writer keep their contracts for all other consumers.
