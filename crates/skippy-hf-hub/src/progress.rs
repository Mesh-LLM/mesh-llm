//! Progress reporting for upload and download operations.
//!
//! Implement [`ProgressHandler`] and pass the handler to the `.progress(...)` setter on any
//! method builder that supports progress reporting (upload, download, snapshot download,
//! `create_commit`, bucket sync, etc.). Each `.progress(...)` argument is converted to [`Progress`]
//! via [`Into`]: an owned handler, an `Arc<H>`, [`Progress::new`], or a [`Progress`] value. When no handler is set,
//! the library emits nothing — there is no runtime cost.
//!
//! # Event model
//!
//! Every operation that supports progress emits a stream of [`ProgressEvent`]s
//! framed by a `Start` event and a `Complete` event. If the operation returns
//! an error, `Complete` is **not** emitted — consumers should rely on the
//! returned `Result` for operation success, not on observing `Complete`.
//!
//! Upload and download events are distinct enums ([`UploadEvent`] and
//! [`DownloadEvent`]) wrapped in a [`ProgressEvent`] discriminator. An
//! individual operation only emits one variant family (e.g., `create_commit`
//! only emits `Upload(*)`, `snapshot_download` only emits `Download(*)`).
//!
//! ## Upload event sequence
//!
//! ```text
//!   Start ──┐
//!           │ (silent preflight: preupload API, LFS classification)
//!   Progress ── Progress ── … ── Progress
//!           │ (active upload — poll loop fires ~every 100ms)
//!   Committing
//!           │ (silent: commit API round-trip)
//!   Complete
//! ```
//!
//! ## Download event sequence
//!
//! ```text
//!   Start ──┐
//!           │ (HEAD fan-out may precede this for snapshot downloads)
//!   Progress ── Progress ── … ── AggregateProgress ── Progress ── …
//!           │ (Progress = per-file deltas; AggregateProgress = xet batch
//!           │  totals. Either or both, interleaved.)
//!   Complete
//! ```
//!
//! # Implementing a handler
//!
//! ```
//! use std::sync::Arc;
//!
//! use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler, UploadEvent};
//!
//! struct PrintHandler;
//!
//! impl ProgressHandler for PrintHandler {
//!     fn on_progress(&self, event: &ProgressEvent) {
//!         match event {
//!             ProgressEvent::Upload(UploadEvent::Start {
//!                 total_files,
//!                 total_bytes,
//!             }) => {
//!                 println!("Uploading {total_files} file(s), {total_bytes} bytes");
//!             },
//!             ProgressEvent::Upload(UploadEvent::Progress {
//!                 bytes_completed,
//!                 total_bytes,
//!                 ..
//!             }) => {
//!                 println!("  {bytes_completed}/{total_bytes}");
//!             },
//!             ProgressEvent::Upload(UploadEvent::Committing) => {
//!                 println!("Committing...");
//!             },
//!             ProgressEvent::Upload(UploadEvent::Complete) => {
//!                 println!("Done.");
//!             },
//!             _ => {},
//!         }
//!     }
//! }
//! ```
//!
//! # Thread safety and performance contract
//!
//! [`ProgressHandler`] requires `Send + Sync` because the library may invoke
//! `on_progress` from arbitrary tokio tasks, including background poll loops
//! running on ~100ms tick intervals during active transfers. Implementations
//! should:
//!
//! - **Never block.** Blocking `on_progress` blocks the emitting task, which for upload/download poll loops means
//!   delaying subsequent progress observations. For network streams the library calls `on_progress` on the stream-read
//!   path itself — slow handlers directly slow the transfer.
//! - **Not panic.** Panics propagate through the tokio runtime and can abort the operation.
//! - **Be idempotent / tolerant of redundant state.** The library guarantees event *ordering* but not *deduplication*;
//!   e.g., a file may receive multiple `FileStatus::Complete` events across `Progress` and a final cleanup emit in edge
//!   cases. Consumers that track completion should use a set keyed by filename to ignore repeats.

use std::sync::Arc;

/// Receives progress updates from long-running upload and download operations.
///
/// Register a handler by wrapping it in [`Progress`] and passing it to the
/// `.progress(...)` setter of any method builder that supports progress reporting.
/// See the [module-level docs](self) for the event model, ordering guarantees, and
/// the must-not-block / `Send + Sync` contract.
///
/// # Example
///
/// ```
/// use std::sync::atomic::{AtomicU64, Ordering};
///
/// use hf_hub::progress::{ProgressEvent, ProgressHandler, UploadEvent};
///
/// struct ByteCounter {
///     bytes: AtomicU64,
/// }
///
/// impl ProgressHandler for ByteCounter {
///     fn on_progress(&self, event: &ProgressEvent) {
///         if let ProgressEvent::Upload(UploadEvent::Progress {
///             bytes_completed, ..
///         }) = event
///         {
///             self.bytes.store(*bytes_completed, Ordering::Relaxed);
///         }
///     }
/// }
/// ```
pub trait ProgressHandler: Send + Sync {
    /// Invoked by the library for each progress event. The `event` reference is
    /// only valid for the duration of the call.
    fn on_progress(&self, event: &ProgressEvent);
}

/// Shared-ownership wrapper around a [`ProgressHandler`] trait object.
///
/// Internally an `Arc<dyn ProgressHandler>`, so cloning is cheap. `progress` setters
/// take `impl Into<Progress>`, so an owned handler, an `Arc<H>`, or an
/// `Arc<dyn ProgressHandler>` can be passed directly.
///
/// ```
/// use std::sync::Arc;
///
/// use hf_hub::progress::{Progress, ProgressEvent, ProgressHandler};
///
/// struct Noop;
/// impl ProgressHandler for Noop {
///     fn on_progress(&self, _event: &ProgressEvent) {}
/// }
///
/// let handler: Progress = Noop.into();
/// let shared: Progress = Arc::new(Noop).into();
/// let direct = Progress::new(Noop);
/// ```
pub struct Progress(Arc<dyn ProgressHandler>);

impl Progress {
    /// Wrap a handler value in a new `Progress`.
    pub fn new<H: ProgressHandler + 'static>(handler: H) -> Self {
        Self(Arc::new(handler))
    }
}

impl Clone for Progress {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl std::fmt::Debug for Progress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Progress").finish_non_exhaustive()
    }
}

impl std::ops::Deref for Progress {
    type Target = dyn ProgressHandler;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl<H: ProgressHandler + 'static> From<H> for Progress {
    fn from(handler: H) -> Self {
        Self(Arc::new(handler))
    }
}

impl<H: ProgressHandler + 'static> From<Arc<H>> for Progress {
    fn from(handler: Arc<H>) -> Self {
        Self(handler)
    }
}

impl From<Arc<dyn ProgressHandler>> for Progress {
    fn from(handler: Arc<dyn ProgressHandler>) -> Self {
        Self(handler)
    }
}

/// Top-level progress event dispatched to [`ProgressHandler::on_progress`].
///
/// A single operation emits only one variant family — uploads never produce
/// `Download(*)` and vice versa.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// Emitted by upload operations (`upload_file`, `upload_folder`, `create_commit`,
    /// `HFBucket::upload_files`, bucket sync in the upload direction).
    Upload(UploadEvent),
    /// Emitted by download operations (`download_file`, `snapshot_download`,
    /// `HFBucket::download_files`, bucket sync in the download direction).
    Download(DownloadEvent),
}

/// Lifecycle events for a single upload operation. See the [module-level
/// docs](self) for the `Start` → `Progress` → `Committing` → `Complete` ordering and
/// the silent-gap caveats.
#[derive(Debug, Clone)]
pub enum UploadEvent {
    /// Upload has begun; totals are known.
    Start {
        /// Number of files the operation will upload (excludes deletes and other
        /// non-add operations in a commit).
        total_files: usize,
        /// Sum of source-content sizes in bytes, before xet deduplication.
        total_bytes: u64,
    },

    /// Byte-level progress during the active upload phase, emitted at ~10Hz by the
    /// xet upload poll loop.
    ///
    /// Two byte-count dimensions are reported because xet performs content-defined
    /// deduplication. The `bytes_completed` / `total_bytes` pair tracks logical
    /// content bytes (use for a "% processed" bar); the `transfer_bytes_*` triplet
    /// tracks post-dedup network bytes actually sent (use for a "network activity"
    /// bar). For deduplicated data, `transfer_bytes` ≪ `total_bytes`.
    ///
    /// `files` is a snapshot of every xet-tracked file's state at this event. May
    /// be empty for operations that don't go through xet (small inline files skip
    /// `Progress` entirely).
    Progress {
        /// Logical content bytes processed so far across all files.
        bytes_completed: u64,
        /// Total logical content bytes for the operation (matches `Start.total_bytes`).
        total_bytes: u64,
        /// Rate of logical content processing in bytes/sec. `None` during warm-up.
        bytes_per_sec: Option<f64>,
        /// Post-dedup network bytes actually sent so far.
        transfer_bytes_completed: u64,
        /// Total post-dedup network bytes the operation is expected to send.
        transfer_bytes: u64,
        /// Rate of network transfer in bytes/sec. `None` during warm-up.
        transfer_bytes_per_sec: Option<f64>,
        /// Per-file snapshot of every xet-tracked file in the upload.
        files: Vec<FileProgress>,
    },

    /// Emitted once, immediately before the commit API call. Signals that all byte
    /// transfer is done; the call itself is silent until `Complete`.
    Committing,

    /// Terminal event on success. Not emitted on failure — check the returned `Result`.
    Complete,
}

/// Lifecycle events for a single download operation. See the [module-level
/// docs](self) for ordering, the two-channel `Progress` vs `AggregateProgress`
/// model, and cache-hit fast paths.
#[derive(Debug, Clone)]
pub enum DownloadEvent {
    /// Download operation has begun; totals are known. Fires after the HEAD round-trip
    /// (or HEAD fan-out for `snapshot_download`).
    Start {
        /// Number of files to download.
        total_files: usize,
        /// Sum of remote file sizes in bytes, as reported by HEAD responses.
        total_bytes: u64,
    },

    /// Per-file progress **delta** — `files` contains only files whose status or
    /// byte count changed since the previous `Progress` event. Consumers wanting a
    /// running view of every file must accumulate state by filename.
    Progress {
        /// Files whose state changed since the previous `Progress` event.
        files: Vec<FileProgress>,
    },

    /// Aggregate byte-level progress for the in-flight xet batch (~10Hz). Reports
    /// cumulative bytes for the entire batch with no per-file breakdown — xet
    /// reports aggregate stats only.
    AggregateProgress {
        /// Bytes downloaded so far across the in-flight xet batch.
        bytes_completed: u64,
        /// Total bytes for the in-flight xet batch.
        total_bytes: u64,
        /// Download rate in bytes/sec. `None` until enough samples accumulate.
        bytes_per_sec: Option<f64>,
    },

    /// Terminal event on success. Not emitted on failure — check the returned `Result`.
    Complete,
}

/// Progress for a single file, carried in `Progress` events. See the parent
/// variant's docs for whether `files` is a snapshot ([`UploadEvent::Progress`]) or
/// a delta ([`DownloadEvent::Progress`]).
#[derive(Debug, Clone)]
pub struct FileProgress {
    /// Path as known by the repository or bucket (the `path_in_repo` used when
    /// uploading, or the remote path as returned from tree listing).
    pub filename: String,
    /// Bytes transferred so far for this file.
    pub bytes_completed: u64,
    /// Total bytes expected for this file. Zero when the size is unknown (e.g.,
    /// fast-path cached files emitted purely to signal completion).
    pub total_bytes: u64,
    /// Current lifecycle stage.
    pub status: FileStatus,
}

/// Lifecycle stage of an individual file within a transfer: `Started` →
/// `InProgress` → `Complete`. Not every stage is observed for every file (fast
/// transfers may skip from `Started` to `Complete`, cache hits emit only `Complete`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileStatus {
    /// File has been queued for transfer but no bytes have moved yet.
    Started,
    /// Bytes are actively being transferred.
    InProgress,
    /// All bytes for this file have been transferred. Terminal state.
    Complete,
}

impl From<UploadEvent> for ProgressEvent {
    fn from(event: UploadEvent) -> Self {
        ProgressEvent::Upload(event)
    }
}

impl From<DownloadEvent> for ProgressEvent {
    fn from(event: DownloadEvent) -> Self {
        ProgressEvent::Download(event)
    }
}

pub(crate) trait EmitEvent {
    fn emit(&self, event: impl Into<ProgressEvent>);
}

impl<T: ProgressHandler + ?Sized> EmitEvent for T {
    fn emit(&self, event: impl Into<ProgressEvent>) {
        self.on_progress(&event.into());
    }
}

impl EmitEvent for Option<Progress> {
    fn emit(&self, event: impl Into<ProgressEvent>) {
        if let Some(h) = self {
            h.on_progress(&event.into());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    struct RecordingHandler {
        events: Mutex<Vec<ProgressEvent>>,
    }

    impl RecordingHandler {
        fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
            }
        }

        fn events(&self) -> Vec<ProgressEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl ProgressHandler for RecordingHandler {
        fn on_progress(&self, event: &ProgressEvent) {
            self.events.lock().unwrap().push(event.clone());
        }
    }

    #[test]
    fn handler_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<RecordingHandler>>();
    }

    #[test]
    fn emit_with_none_is_noop() {
        let progress: Option<Progress> = None;
        progress.emit(DownloadEvent::Complete);
    }

    #[test]
    fn emit_records_events() {
        let handler = Arc::new(RecordingHandler::new());
        let progress: Option<Progress> = Some(handler.clone().into());

        progress.emit(UploadEvent::Start {
            total_files: 2,
            total_bytes: 1024,
        });
        progress.emit(UploadEvent::Progress {
            bytes_completed: 512,
            total_bytes: 1024,
            bytes_per_sec: Some(100.0),
            transfer_bytes_completed: 0,
            transfer_bytes: 0,
            transfer_bytes_per_sec: None,
            files: vec![],
        });
        progress.emit(UploadEvent::Complete);

        let events = handler.events();
        assert_eq!(events.len(), 3);
        assert!(matches!(
            events[0],
            ProgressEvent::Upload(UploadEvent::Start { .. })
        ));
        assert!(matches!(
            events[1],
            ProgressEvent::Upload(UploadEvent::Progress { .. })
        ));
        assert!(matches!(
            events[2],
            ProgressEvent::Upload(UploadEvent::Complete)
        ));
    }

    #[test]
    fn download_file_lifecycle() {
        let handler = Arc::new(RecordingHandler::new());
        let progress: Option<Progress> = Some(handler.clone().into());

        progress.emit(DownloadEvent::Start {
            total_files: 1,
            total_bytes: 1000,
        });
        progress.emit(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "file.bin".to_string(),
                bytes_completed: 0,
                total_bytes: 1000,
                status: FileStatus::Started,
            }],
        });
        progress.emit(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "file.bin".to_string(),
                bytes_completed: 500,
                total_bytes: 1000,
                status: FileStatus::InProgress,
            }],
        });
        progress.emit(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "file.bin".to_string(),
                bytes_completed: 1000,
                total_bytes: 1000,
                status: FileStatus::Complete,
            }],
        });
        progress.emit(DownloadEvent::Complete);

        let events = handler.events();
        assert_eq!(events.len(), 5);
    }

    #[test]
    fn upload_event_ordering() {
        let handler = Arc::new(RecordingHandler::new());
        let progress: Option<Progress> = Some(handler.clone().into());

        progress.emit(UploadEvent::Start {
            total_files: 1,
            total_bytes: 100,
        });
        progress.emit(UploadEvent::Progress {
            bytes_completed: 50,
            total_bytes: 100,
            bytes_per_sec: None,
            transfer_bytes_completed: 0,
            transfer_bytes: 0,
            transfer_bytes_per_sec: None,
            files: vec![],
        });
        progress.emit(UploadEvent::Committing);
        progress.emit(UploadEvent::Complete);

        let events = handler.events();
        assert_eq!(events.len(), 4);
        assert!(matches!(
            events[0],
            ProgressEvent::Upload(UploadEvent::Start { .. })
        ));
        assert!(matches!(
            events[1],
            ProgressEvent::Upload(UploadEvent::Progress { .. })
        ));
        assert!(matches!(
            events[2],
            ProgressEvent::Upload(UploadEvent::Committing)
        ));
        assert!(matches!(
            events[3],
            ProgressEvent::Upload(UploadEvent::Complete)
        ));
    }

    #[test]
    fn upload_progress_with_per_file_data() {
        let handler = Arc::new(RecordingHandler::new());
        let progress: Option<Progress> = Some(handler.clone().into());

        progress.emit(UploadEvent::Progress {
            bytes_completed: 500,
            total_bytes: 1000,
            bytes_per_sec: Some(100.0),
            transfer_bytes_completed: 250,
            transfer_bytes: 800,
            transfer_bytes_per_sec: Some(50.0),
            files: vec![
                FileProgress {
                    filename: "model/weights.bin".to_string(),
                    bytes_completed: 300,
                    total_bytes: 600,
                    status: FileStatus::InProgress,
                },
                FileProgress {
                    filename: "config.json".to_string(),
                    bytes_completed: 200,
                    total_bytes: 400,
                    status: FileStatus::InProgress,
                },
            ],
        });

        let events = handler.events();
        assert_eq!(events.len(), 1);
        if let ProgressEvent::Upload(UploadEvent::Progress {
            files,
            transfer_bytes_completed,
            transfer_bytes,
            transfer_bytes_per_sec,
            ..
        }) = &events[0]
        {
            assert_eq!(files.len(), 2);
            assert_eq!(files[0].filename, "model/weights.bin");
            assert_eq!(files[1].filename, "config.json");
            assert_eq!(*transfer_bytes_completed, 250);
            assert_eq!(*transfer_bytes, 800);
            assert_eq!(*transfer_bytes_per_sec, Some(50.0));
        } else {
            panic!("expected Upload(Progress)");
        }
    }
}
