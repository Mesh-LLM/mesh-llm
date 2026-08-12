//! Fail-open artifact-content capture for platforms where privacy cannot be guaranteed.
//!
//! Metadata persistence is deliberately not coupled to this circuit breaker. Callers retain
//! their `Arc<LogStore>` and can continue recording summaries, lifecycle events, and audit data
//! even when artifact content capture has been disabled.

use crate::artifact_privacy::{ArtifactPrivacy, PlatformArtifactPrivacy};
use crate::artifacts::{
    ArtifactContent, ArtifactFileStore, ArtifactRedactor, ArtifactWriteReceipt,
};
use crate::error::LogStoreError;
use crate::repositories::CascadeArtifactPointer;
use crate::store::{Clock, LogStore};
use crate::{
    DeleteOneRequest, MaintenanceExecutionControl, MaintenanceOperationId, MaintenanceReason,
    MaintenanceReceipt,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Stable, sanitised code for the privacy circuit breaker.
pub const ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE: &str =
    "artifact_capture_disabled_privacy_unavailable";

/// Path-free reason attached to the one-shot health/audit marker and disabled outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactCaptureDisabledReason;

impl ArtifactCaptureDisabledReason {
    pub const fn code(self) -> &'static str {
        ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE
    }
}

/// One-time signal for the host health/audit layer.
///
/// The store does not emit this marker itself; the host owns health and audit emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactCaptureHealthMarker {
    reason: ArtifactCaptureDisabledReason,
}

impl ArtifactCaptureHealthMarker {
    pub const fn reason(self) -> ArtifactCaptureDisabledReason {
        self.reason
    }
}

/// Result of attempting to persist artifact content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactCaptureOutcome {
    Written(ArtifactWriteReceipt),
    Disabled(ArtifactCaptureDisabledReason),
}

enum CaptureState {
    Available(ArtifactFileStore),
    Disabled {
        marker: Option<ArtifactCaptureHealthMarker>,
    },
}

/// Serialised, fail-open facade around strict artifact storage.
///
/// A privacy failure permanently disables only content capture. Strict `ArtifactFileStore`
/// callers retain their existing error-returning API.
pub struct FailOpenArtifactCapture {
    state: Mutex<CaptureState>,
}

impl FailOpenArtifactCapture {
    /// Open content capture, disabling it rather than failing if privacy cannot be guaranteed.
    pub fn open(
        artifact_root: PathBuf,
        clock: Arc<dyn Clock>,
        store: Arc<LogStore>,
        redactor: ArtifactRedactor,
    ) -> Result<Self, LogStoreError> {
        Self::open_with_privacy(
            artifact_root,
            clock,
            store,
            redactor,
            Arc::new(PlatformArtifactPrivacy),
        )
    }

    /// Open capture with an explicit privacy enforcer.
    ///
    /// This is primarily an integration-test seam. Production callers should
    /// use [`Self::open`], which selects the platform enforcer.
    #[doc(hidden)]
    pub fn open_with_privacy(
        artifact_root: PathBuf,
        clock: Arc<dyn Clock>,
        store: Arc<LogStore>,
        redactor: ArtifactRedactor,
        privacy: Arc<dyn ArtifactPrivacy>,
    ) -> Result<Self, LogStoreError> {
        match ArtifactFileStore::open_with_privacy_and_shared_store(
            artifact_root,
            clock,
            store,
            redactor,
            privacy,
        ) {
            Ok(store) => Ok(Self {
                state: Mutex::new(CaptureState::Available(store)),
            }),
            // Artifact setup is always privacy-sensitive. A failed root,
            // unsafe symlink/path, or platform ACL failure must disable only
            // content capture and leave the already-open metadata store alive.
            Err(_) => Ok(Self::disabled()),
        }
    }

    fn disabled() -> Self {
        Self {
            state: Mutex::new(CaptureState::Disabled {
                marker: Some(ArtifactCaptureHealthMarker {
                    reason: ArtifactCaptureDisabledReason,
                }),
            }),
        }
    }

    /// Persist one artifact, or return a stable disabled outcome after a privacy failure.
    #[allow(clippy::too_many_arguments)]
    pub fn write_artifact(
        &self,
        artifact_id: &str,
        request_id: &str,
        kind: &str,
        occurred_at: &str,
        content: &[u8],
        media_kind: Option<&str>,
        version: u32,
        redacted_flag: bool,
        truncated_flag: bool,
        byte_limit: usize,
        aggregate_limit: usize,
    ) -> Result<ArtifactCaptureOutcome, LogStoreError> {
        let mut state = self.state.lock().expect("capture state mutex poisoned");
        let CaptureState::Available(store) = &mut *state else {
            return Ok(ArtifactCaptureOutcome::Disabled(
                ArtifactCaptureDisabledReason,
            ));
        };

        match store.write_artifact(
            artifact_id,
            request_id,
            kind,
            occurred_at,
            content,
            media_kind,
            version,
            redacted_flag,
            truncated_flag,
            byte_limit,
            aggregate_limit,
        ) {
            Ok(receipt) => Ok(ArtifactCaptureOutcome::Written(receipt)),
            Err(LogStoreError::PrivacyNotGuaranteed) => {
                *state = CaptureState::Disabled {
                    marker: Some(ArtifactCaptureHealthMarker {
                        reason: ArtifactCaptureDisabledReason,
                    }),
                };
                Ok(ArtifactCaptureOutcome::Disabled(
                    ArtifactCaptureDisabledReason,
                ))
            }
            Err(error) => Err(error),
        }
    }

    /// Returns the sanitised health marker exactly once over this object's lifetime.
    pub fn take_health_marker(&self) -> Option<ArtifactCaptureHealthMarker> {
        let mut state = self.state.lock().expect("capture state mutex poisoned");
        match &mut *state {
            CaptureState::Available(_) => None,
            CaptureState::Disabled { marker } => marker.take(),
        }
    }

    pub fn is_disabled(&self) -> bool {
        let state = self.state.lock().expect("capture state mutex poisoned");
        matches!(&*state, CaptureState::Disabled { .. })
    }

    /// Read an artifact through the same confined capture owner that wrote it.
    ///
    /// This remains a host-internal primitive: callers must first authorize
    /// the query and verify the pointer's redaction metadata before returning
    /// any bytes. A disabled capture facade deliberately exposes no file
    /// content even when metadata remains available in SQLite.
    pub fn read_artifact(&self, artifact_id: &str) -> Result<ArtifactContent, LogStoreError> {
        let state = self.state.lock().expect("capture state mutex poisoned");
        let CaptureState::Available(store) = &*state else {
            return Err(LogStoreError::ArtifactMissing {
                artifact_id: artifact_id.to_string(),
            });
        };
        store.read_artifact(artifact_id)
    }

    /// Remove only files whose durable pointers were already selected by a
    /// committed cascade cleanup. A disabled capture facade has no trusted
    /// artifact root to touch, so it deliberately becomes a no-op.
    ///
    /// The caller owns the database transaction that produced `pointers`.
    /// This method is intentionally best-effort because cleanup must never
    /// turn a local storage fault into request-serving failure.
    pub fn delete_cascade_artifact_files(&self, pointers: &[CascadeArtifactPointer]) {
        let state = self.state.lock().expect("capture state mutex poisoned");
        if let CaptureState::Available(store) = &*state {
            // The durable queue remains intact on an unsafe path or I/O
            // failure, so the next bounded retention pass can retry it.
            let _ = store.delete_artifact_files(pointers);
        }
    }

    /// Execute a receipt-backed cleanup through the confined artifact owner.
    /// Receipts expose only IDs and counts; they never expose artifact paths.
    pub fn execute_cleanup(
        &self,
        operation_id: MaintenanceOperationId,
        reason: &MaintenanceReason,
        control: &dyn MaintenanceExecutionControl,
    ) -> Result<MaintenanceReceipt, LogStoreError> {
        let state = self.state.lock().expect("capture state mutex poisoned");
        let CaptureState::Available(store) = &*state else {
            return Err(LogStoreError::ArtifactDeletionUnavailable);
        };
        store.execute_cleanup(operation_id, reason, control)
    }

    /// Delete one terminal request through the same confined artifact owner
    /// that persisted its content. A disabled capture facade deliberately
    /// declines the operation because it has no trusted artifact root.
    pub fn delete_request_cascade(
        &self,
        request: &DeleteOneRequest,
        control: &dyn MaintenanceExecutionControl,
    ) -> Result<MaintenanceReceipt, LogStoreError> {
        let state = self.state.lock().expect("capture state mutex poisoned");
        let CaptureState::Available(store) = &*state else {
            return Err(LogStoreError::ArtifactDeletionUnavailable);
        };
        store.delete_request_cascade(request, control)
    }

    /// Persist a delete-one receipt before starting its artifact cascade.
    pub fn prepare_delete_request(
        &self,
        request: &DeleteOneRequest,
        control: &dyn MaintenanceExecutionControl,
    ) -> Result<MaintenanceReceipt, LogStoreError> {
        let state = self.state.lock().expect("capture state mutex poisoned");
        let CaptureState::Available(store) = &*state else {
            return Err(LogStoreError::ArtifactDeletionUnavailable);
        };
        store.prepare_delete_request(request, control)
    }

    /// Execute a receipt that was durably prepared by
    /// [`Self::prepare_delete_request`].
    pub fn execute_prepared_delete_request(
        &self,
        request: &DeleteOneRequest,
        control: &dyn MaintenanceExecutionControl,
    ) -> Result<MaintenanceReceipt, LogStoreError> {
        let state = self.state.lock().expect("capture state mutex poisoned");
        let CaptureState::Available(store) = &*state else {
            return Err(LogStoreError::ArtifactDeletionUnavailable);
        };
        store.execute_prepared_delete_request(request, control)
    }
}
