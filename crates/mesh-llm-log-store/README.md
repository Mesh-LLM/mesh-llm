# mesh-llm-log-store

`mesh-llm-log-store` provides the durable SQLite persistence layer for
mesh-llm's canonical logging pipeline.

It owns bounded request and lifecycle history, privacy-safe metadata and
artifact storage, and the audited maintenance operations used by the host
runtime's trusted local logging APIs. The crate keeps persistence policy and
storage details below the runtime and transport layers so callers can query,
retain, and clean up logs without coupling those APIs to SQLite internals.

## Schema lifecycle

New empty databases are initialized atomically to the complete current schema.
Reopening a current database performs no schema work and preserves its data. A
database with unrecognized objects at version 0 is rejected without being reset
or migrated.

The only public-schema compatibility exception is the exact released physical
schema with source `user_version` 3 or 11. Its tables, columns, indexes, foreign
keys, checks, partial-index predicates, and `AUTOINCREMENT` semantics must match
the released structure exactly, its `application_id` must be zero, and it must
not contain private lineage objects. The store then adds the three nullable
caller identity columns in place and atomically adopts the database into the
private lineage at epoch 1 without rebuilding tables or copying rows.

Lookalikes, partial schemas, and the same physical schema under any source marker
other than 3 or 11 fail closed without mutation. Future private schema changes
must be registered as contiguous, forward-only steps; each step commits its
schema, data, and `user_version` together so a later failure can resume from the
last committed version.
