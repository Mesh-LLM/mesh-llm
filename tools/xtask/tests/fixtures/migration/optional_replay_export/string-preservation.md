# String preservation regression expectations

The historical `legacy.json`, `README.md`, and `SurrogateGap` case labels are
unchanged records of the earlier qualification. G01 and G02 now require exact
status, both streams, and complete file snapshots against the frozen Python
observations. Their old Rust mismatches remain in the independent verification
evidence and the fix's failing-first captures; no historical capture is replaced.

`replay_strings.rs` adds exact CLI expectations for lone surrogates, distinct
surrogate and U+FFFD keys, canonical scalar pairs, decoded duplicate keys,
Python code-point ordering, and literal marker/escape collision controls.
`replay_input.rs` now requires the preserved lone-surrogate policy diagnostic.

Replay uses its own code-point strings before reducing duplicate object keys.
Only adjacent escaped high/low surrogate pairs combine. Numeric tokens still
use the unchanged exact-number decoder. The shared JSON consumers, parser
hooks, serde features, and historical numeric receipts are unchanged.
