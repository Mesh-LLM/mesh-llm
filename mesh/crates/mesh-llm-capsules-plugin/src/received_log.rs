//! "Asked of you": this node's own log of the requests other nodes made of
//! it, and what it answered. The provider's half of the double entry; the
//! Evidence page's drill reads it (`evidence_panes::read_received_log`).
//!
//! The same file and line shape the Python reference wrote: `received_log.jsonl`, one
//! `{ts, path, requester_id, subject_kind, status, reason}` object per line,
//! plus who proved they asked, for an evidence request: `{request_digest,
//! principal_key_id, principal_node_id, principal_binding, audience, replay}`
//! (null where nothing was proven; `principal_binding` is `unbound` for a key
//! that verified but no host binds, which is never a principal; `replay` is
//! `unchecked` when such a key's nonce could not be held for the replay
//! check),
//! in the directory `evidence_routes::EvidenceSource::received_log_dir`
//! resolves. **Opt-in, as before:** nothing is written unless that directory
//! existed when the plugin started and this process holds a lock covering it
//! (`data_dir::lock`); otherwise the log is off for the run. Never the ledger
//! directory: this log is not evidence.
//!
//! **Best-effort.** An answer is decided before it is logged, and a logging
//! failure never changes it.
//!
//! **Bounded.** The requester's id is its own word, so it is cut to
//! [`MAX_REQUESTER_ID_CHARS`] characters, and a line is never longer than
//! the few fields it names. The file stops growing at
//! [`MAX_RECEIVED_LOG_BYTES`]: past it, a line is not written, and the
//! dropped lines are counted ([`dropped`]) and reported in the node's own
//! log (the first drop, then every 1000th). The Evidence page shows only the
//! newest lines anyway; an operator who wants more moves the file aside.

use std::io::Write;
use std::path::Path;

use serde_json::{json, Value};

pub const RECEIVED_LOG_FILE: &str = "received_log.jsonl";

/// The largest the log grows: 4 MiB, some 20,000 lines.
pub const MAX_RECEIVED_LOG_BYTES: u64 = 4 * 1024 * 1024;

static DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Lines not written since this process started, because the log was full.
/// The warning a full log raises carries the same count; only the tests read
/// it directly.
#[cfg(test)]
pub fn dropped() -> u64 {
    DROPPED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The longest self-declared requester id kept, in characters.
pub const MAX_REQUESTER_ID_CHARS: usize = 128;

/// One line: what was asked, by whom (as they said), and the outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry<'a> {
    pub ts: &'a str,
    /// The route asked: `evidence-request`, `evidence/record-push`.
    pub path: &'a str,
    /// The id the request states: its word, not who asked.
    pub requester_id: Option<&'a str>,
    pub subject_kind: Option<&'a str>,
    /// `answered`, `refused`, `received` or `issued`.
    pub status: &'a str,
    pub reason: Option<&'a str>,
    /// The digest of the request bytes as received.
    pub request_digest: Option<&'a str>,
    /// The key the request's signature verified under, if it did.
    pub principal_key_id: Option<&'a str>,
    /// The node that key is bound to, if the binding verified.
    pub principal_node_id: Option<&'a str>,
    /// `bound`, or `unbound` for a key that verified but no host binds.
    pub principal_binding: Option<&'a str>,
    /// The node a signed request named as its audience, as checked.
    pub audience: Option<&'a str>,
    /// `checked`, or `unchecked` for an unbound key's nonce the full pool
    /// could not hold.
    pub replay: Option<&'a str>,
}

impl Entry<'_> {
    pub fn to_json(&self) -> Value {
        json!({
            "ts": self.ts,
            "path": self.path,
            "requester_id": self.requester_id.map(|id| id.chars().take(MAX_REQUESTER_ID_CHARS).collect::<String>()),
            "subject_kind": self.subject_kind,
            "status": self.status,
            "reason": self.reason,
            "request_digest": self.request_digest,
            "principal_key_id": self.principal_key_id,
            "principal_node_id": self.principal_node_id,
            "principal_binding": self.principal_binding,
            "audience": self.audience,
            "replay": self.replay,
        })
    }
}

/// The one directory this run may write the log to, decided when the data
/// directory is locked (`data_dir::lock`): `None` when the log is off for this
/// run (no directory at start, or one this process could not lock).
static WRITABLE: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();

/// Set once, at start, by `data_dir::lock`. Not in unit tests, which take
/// many locks in one process; they check [`writable_under`] directly.
pub(crate) fn set_writable(dir: Option<std::path::PathBuf>) {
    if !cfg!(test) {
        let _ = WRITABLE.set(dir);
    }
}

fn writable(dir: &Path) -> bool {
    writable_under(WRITABLE.get(), dir)
}

/// Whether `dir` is the directory this run locked for the log (`decided`).
/// A directory made after start, or any other, is never written. Undecided
/// (no lock taken in this process: a unit test), any existing directory is.
fn writable_under(decided: Option<&Option<std::path::PathBuf>>, dir: &Path) -> bool {
    match decided {
        Some(Some(locked)) => std::fs::canonicalize(dir).is_ok_and(|real| &real == locked),
        Some(None) => false,
        None => cfg!(test),
    }
}

/// Append `entry` to `<dir>/received_log.jsonl` when `dir` is given, is the
/// directory locked for the log at start, and the log is under
/// [`MAX_RECEIVED_LOG_BYTES`]. Failures are logged and swallowed.
pub fn append(dir: Option<&Path>, entry: &Entry<'_>) {
    append_capped(dir.filter(|d| writable(d)), entry, MAX_RECEIVED_LOG_BYTES);
}

fn append_capped(dir: Option<&Path>, entry: &Entry<'_>, max_bytes: u64) {
    let Some(dir) = dir.filter(|d| d.is_dir()) else {
        return;
    };
    let line = format!("{}\n", entry.to_json());
    let result = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(RECEIVED_LOG_FILE))
        .and_then(|mut file| {
            if file.metadata()?.len() + line.len() as u64 > max_bytes {
                let dropped = DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if dropped == 1 || dropped.is_multiple_of(1000) {
                    tracing::warn!(
                        dropped,
                        max_bytes,
                        "the received log is full; requests are no longer logged"
                    );
                }
                return Ok(());
            }
            file.write_all(line.as_bytes())
        });
    if let Err(error) = result {
        tracing::warn!(%error, "could not append to the received log");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_directory_locked_at_start_is_written() {
        let base = std::env::temp_dir().join(format!("received-writable-{}", std::process::id()));
        let (locked, other) = (base.join("locked"), base.join("made-later"));
        std::fs::create_dir_all(&locked).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        let decided = Some(std::fs::canonicalize(&locked).unwrap());
        assert!(writable_under(Some(&decided), &locked));
        assert!(
            !writable_under(Some(&decided), &other),
            "a directory made after start"
        );
        assert!(!writable_under(Some(&None), &locked), "off for this run");
        let _ = std::fs::remove_dir_all(&base);
    }

    fn entry() -> Entry<'static> {
        Entry {
            ts: "2026-09-29T00:00:00Z",
            path: "evidence-request",
            requester_id: Some("node-a"),
            subject_kind: Some("record"),
            status: "refused",
            reason: Some("no_such_subject"),
            request_digest: Some("ab"),
            principal_key_id: Some("cd"),
            principal_node_id: Some("node-a"),
            principal_binding: Some("bound"),
            audience: Some("node-r"),
            replay: Some("checked"),
        }
    }

    #[test]
    fn a_line_is_what_the_evidence_page_reads() {
        let dir = tempfile::tempdir().unwrap();
        append(Some(dir.path()), &entry());
        let read = crate::evidence_panes::read_received_log(dir.path());
        assert_eq!(read, vec![entry().to_json()]);
    }

    #[test]
    fn nothing_is_written_without_the_directory() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("received-log");
        append(Some(&missing), &entry());
        append(None, &entry());
        assert!(!missing.exists());
    }

    #[test]
    fn the_log_stops_growing_at_its_cap_and_counts_what_it_drops() {
        let dir = tempfile::tempdir().unwrap();
        let line_len = entry().to_json().to_string().len() as u64 + 1;
        let before = dropped();
        for _ in 0..10 {
            append_capped(Some(dir.path()), &entry(), 3 * line_len);
        }
        let len = std::fs::metadata(dir.path().join(RECEIVED_LOG_FILE))
            .unwrap()
            .len();
        assert_eq!(len, 3 * line_len, "exactly the lines that fit");
        assert!(dropped() - before >= 7, "the other seven are counted");
    }

    #[test]
    fn a_requester_id_is_cut_to_the_limit() {
        let long = "x".repeat(10_000);
        let e = Entry {
            requester_id: Some(&long),
            ..entry()
        };
        assert_eq!(
            e.to_json()["requester_id"].as_str().unwrap().len(),
            MAX_REQUESTER_ID_CHARS
        );
    }
}
