# mesh-llm-cli

`mesh-llm-cli` owns reusable command-line support code for the shipped
`mesh-llm` binary: terminal progress indicators, pager behavior, shell quoting,
and shared CLI-facing output format types.

The current host runtime still owns command dispatch while its handlers are
being untangled from runtime internals. New CLI-only helpers should live here
instead of in `mesh-llm-host-runtime`.
