//! `artifact {extract-tar,extract-zip,verify-checksum}` parity with
//! `scripts/safe-extract-tar.py`, `scripts/safe-extract-zip.py` and
//! `scripts/verify-checksum-sidecar.py`.
//!
//! Every archive is assembled byte by byte at runtime in a scratch directory,
//! so malformed members (traversal, device nodes, duplicates) need no
//! checked-in binaries. Expected diagnostics are the legacy scripts' observed
//! output. Set `MIGRATION_ARCHIVES_LEGACY_PYTHON=<python3>` to also run each
//! legacy script on an identical fixture and require matching status, streams
//! and resulting tree (modes, link targets and hard-link counts included).
#![cfg(unix)]

#[path = "migration_archives/support.rs"]
mod support;

#[path = "migration_archives/checksum.rs"]
mod checksum;

#[path = "migration_archives/writers.rs"]
mod writers;

#[path = "migration_archives/tar_extract.rs"]
mod tar_extract;

#[path = "migration_archives/tar_reject.rs"]
mod tar_reject;

#[path = "migration_archives/zip_writers.rs"]
mod zip_writers;

#[path = "migration_archives/zip_extract.rs"]
mod zip_extract;
