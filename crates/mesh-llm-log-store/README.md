# mesh-llm-log-store

`mesh-llm-log-store` provides the durable SQLite persistence layer for
mesh-llm's canonical logging pipeline.

It owns bounded request and lifecycle history, privacy-safe metadata and
artifact storage, and the audited maintenance operations used by the host
runtime's trusted local logging APIs. The crate keeps persistence policy and
storage details below the runtime and transport layers so callers can query,
retain, and clean up logs without coupling those APIs to SQLite internals.
