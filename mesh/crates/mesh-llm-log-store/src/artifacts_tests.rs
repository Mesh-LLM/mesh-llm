//! Artifact storage tests — real tempdir filesystem, no in-memory shortcuts.

use crate::artifact_privacy::ArtifactPrivacy;
use crate::artifacts::{ArtifactFileStore, ArtifactStatus};
use crate::error::LogStoreError;
use crate::store::{Clock as ClockTrait, LogStore};
use sha2::{Digest, Sha256};
use std::fs;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

#[derive(Debug)]
struct TestClock {
    instant: AtomicU64,
}

impl Default for TestClock {
    fn default() -> Self {
        Self {
            instant: AtomicU64::new(0),
        }
    }
}

impl ClockTrait for TestClock {
    fn now(&self) -> String {
        let n = self.instant.fetch_add(1, Ordering::Relaxed);
        format!("2025-01-01T00:00:{:02}Z", n % 60)
    }
}

fn expected_checksum(content: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content);
    hex::encode(hasher.finalize())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrivacyPathKind {
    Directory,
    File,
}

#[derive(Default)]
struct RecordingPrivacy {
    paths: Mutex<Vec<(std::path::PathBuf, PrivacyPathKind)>>,
    reject_files: bool,
}

impl RecordingPrivacy {
    fn rejecting_files() -> Self {
        Self {
            paths: Mutex::new(Vec::new()),
            reject_files: true,
        }
    }

    fn paths(&self) -> Vec<(std::path::PathBuf, PrivacyPathKind)> {
        self.paths.lock().expect("privacy calls lock").clone()
    }

    fn record(&self, path: &std::path::Path, kind: PrivacyPathKind) {
        self.paths
            .lock()
            .expect("privacy calls lock")
            .push((path.to_path_buf(), kind));
    }
}

impl ArtifactPrivacy for RecordingPrivacy {
    fn prepare_directory(&self, path: &std::path::Path) -> Result<(), LogStoreError> {
        self.record(path, PrivacyPathKind::Directory);
        Ok(())
    }

    fn prepare_file(&self, path: &std::path::Path) -> Result<(), LogStoreError> {
        self.record(path, PrivacyPathKind::File);
        if self.reject_files {
            return Err(LogStoreError::PrivacyNotGuaranteed);
        }
        Ok(())
    }
}

include!("artifacts_tests/repository.rs");
include!("artifacts_tests/lifecycle.rs");
include!("artifacts_tests/privacy.rs");
