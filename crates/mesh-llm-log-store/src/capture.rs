//! Fail-open artifact-content capture for platforms where privacy cannot be guaranteed.
//!
//! Metadata persistence is deliberately not coupled to this circuit breaker. Callers retain
//! their `Arc<LogStore>` and can continue recording summaries, lifecycle events, and audit data
//! even when artifact content capture has been disabled.

use crate::artifact_privacy::{ArtifactPrivacy, PlatformArtifactPrivacy};
use crate::artifacts::{ArtifactFileStore, ArtifactRedactor, ArtifactWriteReceipt};
use crate::error::LogStoreError;
use crate::store::{Clock, LogStore};
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
}
