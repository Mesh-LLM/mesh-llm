# CLI and OS recorder observations

`cli-os-20260929.json` freezes the fixed I01-I40/O01-O12 campaign observed by
the unchanged Python wrapper through the reviewed Rust recorder. The original
candidate-local process, input, environment and raw argv receipts remain under
`.omo/evidence/oracle-*-20260929-a`.

These observations are not Python pure-return captures. I27 records a pre-exec
integer-argument failure, not the original return value's kind. I21, I22 and O08
have sanitized, incomplete stderr. I39 is compared with explicitly normalized
dataset/output paths. UTF-8/surrogateescape observations are macOS-specific.

The Rust CLI has no run-family adapter. Sequencing observations retain legacy
behavior; export-stage comparisons do not invent candidate run sequencing.
Container-valued pin Unicode repr remains an explicit known gap outside this
fixed roster. No caller cutover or full parity is approved by these fixtures.
