//! Observer contracts for a package download and its shared batch lifetime.
use hf_hub::progress::{ProgressEvent, ProgressHandler};
use std::{path::Path, sync::Arc};

/// One observer per acquisition, or per group of missing package files.
pub trait PackageProgress: Send + Sync {
    fn batch(&self, label: &str, total_files: usize) -> Arc<dyn PackageProgress>;
    fn file(
        &self,
        label: &str,
        file: &str,
        total_bytes: Option<u64>,
        completed_before: usize,
    ) -> Arc<dyn PackageFileProgress>;
}

/// Lives through a transfer, including its preparing and ready notifications.
/// Drop releases any presentation resources on both success and error.
pub trait PackageFileProgress: ProgressHandler {
    fn ensuring(&self);
    fn ready(&self, path: &Path);
}

#[derive(Default)]
pub struct NoPackageProgress;
impl PackageProgress for NoPackageProgress {
    fn batch(&self, _: &str, _: usize) -> Arc<dyn PackageProgress> {
        Arc::new(Self)
    }
    fn file(&self, _: &str, _: &str, _: Option<u64>, _: usize) -> Arc<dyn PackageFileProgress> {
        Arc::new(Self)
    }
}
impl ProgressHandler for NoPackageProgress {
    fn on_progress(&self, _: &ProgressEvent) {}
}
impl PackageFileProgress for NoPackageProgress {
    fn ensuring(&self) {}
    fn ready(&self, _: &Path) {}
}

pub(super) struct ForwardProgress(pub Arc<dyn PackageFileProgress>);
impl ProgressHandler for ForwardProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        self.0.on_progress(event);
    }
}
