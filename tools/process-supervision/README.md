# process-supervision

`process-supervision` runs CI automation child processes under bounded
supervision. It is a library only. Its normal dependencies are thiserror, libc
on Unix, and windows-sys on Windows, so a narrow CI binary can link it without
pulling in full xtask's dependency closure.

It owns:

- Supervision of cooperative child trees (`supervise`, `supervise_raw`,
  `supervise_with_probe`, `supervise_projected`, `supervise_inherited`):
  deadlines, readiness probes, bounded output capture, secret redaction
  (`Value::Secret`), and cleanup. Unix children keep their inherited process
  group. Windows children start suspended and join a kill-on-close job.
- `retained`: supervision of several long-lived members under one coordinator,
  with recovery and shutdown reporting.
- `interrupt`: the process-wide SIGINT/SIGTERM (or console control) scope whose
  `Cancellation` stops supervised children.

It doesn't own HTTPS transfers through curl (`curl_https`) or any command
dispatch. Those stay in [xtask](../xtask/README.md).

## Consumers

- [xtask](../xtask/README.md), which imports this crate as `crate::process` and
  `crate::command_interrupt`. Its integration targets and examples use the crate
  directly. The interrupt scope's behavior tests live in xtask
  (`tests/migration_lifecycle/interrupt_scope.rs` and `blocked_scope.rs`),
  because they exercise xtask's client readiness and retained sessions.

`just ci-automation-contracts` runs this crate's tests; the main Rust test
matrix skips it.
