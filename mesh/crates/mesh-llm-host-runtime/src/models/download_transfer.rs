use hf_hub::progress::{DownloadEvent, FileStatus};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DownloadTransferStats {
    pub bytes: u64,
    pub elapsed: Duration,
    pub bytes_per_sec: Option<f64>,
}

impl DownloadTransferStats {
    pub(crate) fn combine(stats: Vec<Self>) -> Option<Self> {
        if stats.is_empty() {
            return None;
        }
        let bytes = stats.iter().map(|stat| stat.bytes).sum();
        let elapsed = stats.iter().map(|stat| stat.elapsed).sum();
        let bytes_per_sec = combined_bytes_per_sec(&stats);
        Some(Self {
            bytes,
            elapsed,
            bytes_per_sec,
        })
    }
}

fn combined_bytes_per_sec(stats: &[DownloadTransferStats]) -> Option<f64> {
    let mut rate_bytes = 0.0;
    let mut rate_seconds = 0.0;
    for stat in stats {
        let Some(bytes_per_sec) = stat.bytes_per_sec.filter(|value| *value > 0.0) else {
            continue;
        };
        rate_bytes += stat.bytes as f64;
        rate_seconds += stat.bytes as f64 / bytes_per_sec;
    }
    (rate_seconds > 0.0).then_some(rate_bytes / rate_seconds)
}

#[derive(Debug, Default)]
pub(crate) struct DownloadTransferTracker {
    file_bytes: HashMap<String, u64>,
    aggregate_bytes: u64,
    bytes_per_sec: Option<f64>,
    started_at: Option<Instant>,
    last_progress_at: Option<Instant>,
}

impl DownloadTransferTracker {
    pub(crate) fn apply_download_event(&mut self, event: &DownloadEvent) {
        match event {
            DownloadEvent::Start { .. } => self.record_start(),
            DownloadEvent::Complete => {}
            DownloadEvent::Progress { files } => {
                for file in files {
                    let previous = self.file_bytes.get(&file.filename).copied().unwrap_or(0);
                    let counts_as_transfer = match file.status {
                        FileStatus::Started | FileStatus::InProgress => {
                            file.bytes_completed > previous
                        }
                        FileStatus::Complete => previous > 0 && file.bytes_completed > previous,
                    };
                    if counts_as_transfer {
                        self.record_progress();
                        self.file_bytes
                            .insert(file.filename.clone(), file.bytes_completed);
                    }
                }
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                bytes_per_sec,
                ..
            } => {
                if *bytes_completed > self.aggregate_bytes {
                    self.record_progress();
                    self.aggregate_bytes = *bytes_completed;
                }
                if bytes_per_sec.is_some_and(|value| value > 0.0) {
                    self.bytes_per_sec = *bytes_per_sec;
                }
            }
        }
    }

    pub(crate) fn finish(self) -> Option<DownloadTransferStats> {
        let per_file_bytes = self.file_bytes.values().sum();
        let bytes = self.aggregate_bytes.max(per_file_bytes);
        if bytes == 0 {
            return None;
        }
        let elapsed = match (self.started_at, self.last_progress_at) {
            (Some(start), Some(end)) => end.saturating_duration_since(start),
            _ => Duration::ZERO,
        };
        Some(DownloadTransferStats {
            bytes,
            elapsed,
            bytes_per_sec: self.bytes_per_sec,
        })
    }

    pub(crate) fn finish_with_file_fallback(
        self,
        cached_before: bool,
        path: &Path,
    ) -> Option<DownloadTransferStats> {
        let elapsed = self
            .started_at
            .map(|started_at| started_at.elapsed())
            .unwrap_or(Duration::ZERO);
        if let Some(stats) = self.finish() {
            return Some(stats);
        }
        if cached_before {
            return None;
        }
        let bytes = std::fs::metadata(path).ok()?.len();
        if bytes == 0 {
            return None;
        }
        Some(DownloadTransferStats {
            bytes,
            elapsed,
            bytes_per_sec: None,
        })
    }

    fn record_start(&mut self) {
        if self.started_at.is_none() {
            self.started_at = Some(Instant::now());
        }
    }

    fn record_progress(&mut self) {
        self.record_start();
        let now = Instant::now();
        self.last_progress_at = Some(now);
    }
}

#[cfg(test)]
mod tests {
    use super::DownloadTransferTracker;
    use hf_hub::progress::{DownloadEvent, FileProgress, FileStatus};

    #[test]
    fn download_transfer_stats_ignore_cache_hit_events() {
        let mut tracker = DownloadTransferTracker::default();

        tracker.apply_download_event(&DownloadEvent::Start {
            total_files: 1,
            total_bytes: 1_000,
        });
        tracker.apply_download_event(&DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "model.gguf".to_string(),
                bytes_completed: 1_000,
                total_bytes: 1_000,
                status: FileStatus::Complete,
            }],
        });
        tracker.apply_download_event(&DownloadEvent::Complete);

        assert_eq!(tracker.finish(), None);
    }

    #[test]
    fn download_transfer_stats_accumulate_multipart_progress() {
        let mut tracker = DownloadTransferTracker::default();

        tracker.apply_download_event(&DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "model-00001-of-00002.gguf".to_string(),
                bytes_completed: 400,
                total_bytes: 1_000,
                status: FileStatus::InProgress,
            }],
        });
        tracker.apply_download_event(&DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "model-00001-of-00002.gguf".to_string(),
                bytes_completed: 900,
                total_bytes: 1_000,
                status: FileStatus::InProgress,
            }],
        });
        tracker.apply_download_event(&DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "model-00002-of-00002.gguf".to_string(),
                bytes_completed: 300,
                total_bytes: 1_000,
                status: FileStatus::InProgress,
            }],
        });

        let stats = tracker.finish().expect("multipart transfer stats");
        assert_eq!(stats.bytes, 1_200);
    }

    #[test]
    fn download_transfer_stats_preserve_xet_speed() {
        let mut tracker = DownloadTransferTracker::default();

        tracker.apply_download_event(&DownloadEvent::AggregateProgress {
            bytes_completed: 32_000_000,
            total_bytes: 64_000_000,
            bytes_per_sec: Some(128_000_000.0),
        });

        let stats = tracker.finish().expect("xet transfer stats");
        assert_eq!(stats.bytes, 32_000_000);
        assert_eq!(stats.bytes_per_sec, Some(128_000_000.0));
    }

    #[test]
    fn download_transfer_stats_count_uncached_complete_only_asset() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("model-00001-of-00003.gguf");
        std::fs::write(&path, vec![0; 4_096]).expect("write shard");
        let mut tracker = DownloadTransferTracker::default();

        tracker.apply_download_event(&DownloadEvent::Start {
            total_files: 1,
            total_bytes: 4_096,
        });
        tracker.apply_download_event(&DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "model-00001-of-00003.gguf".to_string(),
                bytes_completed: 4_096,
                total_bytes: 4_096,
                status: FileStatus::Complete,
            }],
        });

        let stats = tracker
            .finish_with_file_fallback(false, &path)
            .expect("uncached complete-only transfer stats");
        assert_eq!(stats.bytes, 4_096);
    }

    #[test]
    fn download_transfer_stats_ignore_cached_complete_only_asset() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("model-00001-of-00003.gguf");
        std::fs::write(&path, vec![0; 4_096]).expect("write shard");
        let mut tracker = DownloadTransferTracker::default();

        tracker.apply_download_event(&DownloadEvent::Start {
            total_files: 1,
            total_bytes: 4_096,
        });
        tracker.apply_download_event(&DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "model-00001-of-00003.gguf".to_string(),
                bytes_completed: 4_096,
                total_bytes: 4_096,
                status: FileStatus::Complete,
            }],
        });

        assert_eq!(tracker.finish_with_file_fallback(true, &path), None);
    }
}
