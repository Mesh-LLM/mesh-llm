# mesh-llm-cli

`mesh-llm-cli` owns the command-line surface for the shipped `mesh-llm` binary:
Clap parser types, runtime surface normalization, terminal progress indicators,
pager behavior, shell quoting, and shared CLI-facing output format types.

The current host runtime still owns command dispatch while its handlers are
being untangled from runtime internals. New parser types and CLI-only helpers
should live here instead of in `mesh-llm-host-runtime`.
