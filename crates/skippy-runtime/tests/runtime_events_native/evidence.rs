//! Buffered evidence for the gated native run.
//!
//! Step markers are collected in memory and flushed only once the run's
//! outcome is known, so a file that starts with `executed` still means every
//! gated step passed, and a file that starts with `failed` names the step
//! that did not, followed by whatever the earlier steps established.

use std::cell::RefCell;
use std::path::PathBuf;

pub struct Evidence {
    path: PathBuf,
    lines: RefCell<Vec<String>>,
}

impl Evidence {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            lines: RefCell::new(Vec::new()),
        }
    }

    pub fn record(&self, line: impl Into<String>) {
        self.lines.borrow_mut().push(line.into());
    }

    /// Writes `executed` followed by every recorded step marker.
    pub fn flush_executed(&self) {
        self.flush_with_header("executed");
    }

    /// Writes a `failed:` header followed by every recorded step marker.
    pub fn flush_failed(&self, reason: &str) {
        self.flush_with_header(&format!("failed: {reason}"));
    }

    fn flush_with_header(&self, header: &str) {
        skippy_runtime::write_evidence_marker(Some(&self.path), header);
        for line in self.lines.borrow().iter() {
            skippy_runtime::write_evidence_marker(Some(&self.path), line);
        }
    }
}
