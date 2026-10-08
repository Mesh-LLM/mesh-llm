//! Behavioural parity tests for `cargo xtool release ...` ports.
//!
//! `notes-base` replaces `scripts/select-release-notes-base.py`;
//! `notes-link` replaces `scripts/release-notes-link.py`;
//! `notes-classify` replaces `scripts/release-notes-classify.py`. Each case runs in a
//! scratch directory with stub `git` and `gh` first on `PATH` (no network),
//! and compares exit status, stdout, stderr, written files and the recorded
//! `git`/`gh` argv with a golden under `fixtures/release/`, captured from the
//! legacy script under Python 3.13 (`{root}` stands for the scratch path).

#[path = "migration_release/failure_diagnostics.rs"]
mod failure_diagnostics;
#[path = "migration_release/notes_base.rs"]
mod notes_base;
#[path = "migration_release/notes_classify.rs"]
mod notes_classify;
#[path = "migration_release/notes_generate.rs"]
mod notes_generate;
#[path = "migration_release/notes_link.rs"]
mod notes_link;
#[path = "migration_release/notes_regroup.rs"]
mod notes_regroup;
#[path = "migration_release/support.rs"]
mod support;
