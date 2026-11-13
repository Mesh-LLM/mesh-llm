//! The download callback captures its node before crossing downloader task boundaries.
use crate::runtime_data::{RuntimeDataCollector, RuntimeDataDirty};
use crate::sdk::status::DownloadProgress;
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use std::sync::{Arc, Mutex};

pub(super) struct HfDownloadProgress {
    pub(super) native: DownloadObserver,
    pub(super) visible: Option<Arc<super::catalog::MeshDownloadProgress>>,
    pub(super) transfer: Arc<Mutex<super::download_transfer::DownloadTransferTracker>>,
}

impl ProgressHandler for HfDownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        self.native.on_progress(event);
        if let Some(visible) = &self.visible {
            visible.on_progress(event);
        }
        if let ProgressEvent::Download(event) = event
            && let Ok(mut transfer) = self.transfer.lock()
        {
            transfer.apply_download_event(event);
        }
    }
}

tokio::task_local! {
    pub(crate) static COLLECTOR: RuntimeDataCollector;
}

#[derive(Default)]
pub(super) struct DownloadObserver(Option<RuntimeDataCollector>);

impl DownloadObserver {
    pub(super) fn current() -> Self {
        Self(COLLECTOR.try_with(Clone::clone).ok())
    }

    pub(super) fn on_progress(&self, event: &ProgressEvent) {
        let Some(collector) = &self.0 else { return };
        let progress = match event {
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                files.first().map(|file| DownloadProgress {
                    file: file.filename.clone(),
                    downloaded_bytes: file.bytes_completed,
                    total_bytes: (file.total_bytes > 0).then_some(file.total_bytes),
                })
            }
            ProgressEvent::Download(DownloadEvent::Complete) => None,
            _ => return,
        };
        collector.update_runtime_status(RuntimeDataDirty::STATUS, |runtime| {
            if runtime.download == progress {
                return false;
            }
            runtime.download = progress;
            true
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::progress::{FileProgress, FileStatus};

    #[tokio::test]
    async fn captured_download_callbacks_publish_only_to_their_node() {
        let collector = RuntimeDataCollector::new();
        let other = RuntimeDataCollector::new();
        let mut updates = collector.subscribe();
        let observer = COLLECTOR
            .scope(collector.clone(), async { DownloadObserver::current() })
            .await;
        observer.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "weights.gguf".into(),
                bytes_completed: 10,
                total_bytes: 100,
                status: FileStatus::InProgress,
            }],
        }));
        updates.changed().await.unwrap();
        assert_eq!(
            collector
                .runtime_status_snapshot()
                .download
                .unwrap()
                .downloaded_bytes,
            10
        );
        assert!(other.runtime_status_snapshot().download.is_none());
        observer.on_progress(&ProgressEvent::Download(DownloadEvent::Complete));
        assert!(collector.runtime_status_snapshot().download.is_none());
    }
}
