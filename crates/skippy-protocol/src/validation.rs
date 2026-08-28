//! Stage protocol constants and wire-frame validation.

use crate::proto;
pub const SCHEMA_VERSION: u32 = 1;
pub const STAGE_ALPN_V2: &[u8] = b"skippy-stage/2";
pub const STAGE_SUBPROTOCOL_NAME: &str = "skippy-stage";
pub const STAGE_SUBPROTOCOL_MAJOR: u32 = 2;
pub const STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL: &str = "stage-control";
pub const STAGE_PROTOCOL_GENERATION: u32 = 5;
/// Generation-scoped stage capability. A peer can advertise `stage-control`
/// while still rejecting current-generation frames, so split planning gates on
/// this exact token before sending current-generation control requests.
pub const STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V5: &str = "stage-generation-5";
pub const STAGE_SUBPROTOCOL_FEATURE_STAGE_GENERATION: &str =
    STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V5;
pub const STAGE_SUBPROTOCOL_FEATURE_ARTIFACT_TRANSFER: &str = "artifact-transfer";
pub const STAGE_SUBPROTOCOL_FEATURE_STATUS_LIST: &str = "status-list";
pub const STAGE_STREAM_CONTROL: u8 = 0x01;
pub const STAGE_STREAM_TRANSPORT: u8 = 0x02;
pub const STAGE_STREAM_ARTIFACT_TRANSFER: u8 = 0x03;
pub const MAX_STAGE_FRAME_BYTES: usize = 8 * 1024 * 1024;
/// Maximum number of unresolved verify windows covered by native checkpoints.
pub const MAX_VERIFY_WINDOW_PIPELINE_DEPTH: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageFrameError {
    BadGeneration { got: u32 },
    InvalidEndpointId { got: usize },
    InvalidArtifactDigestLength { got: usize },
    InvalidArtifactPath,
    InvalidArtifactOffset,
    MissingStageControlCommand,
    MissingStageControlResponse,
    MissingStageTransportTarget,
    MissingStageArtifactTarget,
}

impl std::fmt::Display for StageFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageFrameError::BadGeneration { got } => write!(
                f,
                "bad skippy stage generation: expected {}, got {}",
                STAGE_PROTOCOL_GENERATION, got
            ),
            StageFrameError::InvalidEndpointId { got } => {
                write!(f, "invalid endpoint_id length: expected 32, got {got}")
            }
            StageFrameError::InvalidArtifactDigestLength { got } => write!(
                f,
                "invalid artifact sha256 length: expected 64 hex chars, got {got}"
            ),
            StageFrameError::InvalidArtifactPath => {
                write!(f, "artifact relative_path must be a safe relative path")
            }
            StageFrameError::InvalidArtifactOffset => {
                write!(f, "artifact offset exceeds expected artifact size")
            }
            StageFrameError::MissingStageControlCommand => {
                write!(f, "stage control command is required but missing")
            }
            StageFrameError::MissingStageControlResponse => {
                write!(f, "stage control response is required but missing")
            }
            StageFrameError::MissingStageTransportTarget => {
                write!(f, "stage transport target is required but missing")
            }
            StageFrameError::MissingStageArtifactTarget => {
                write!(f, "stage artifact transfer target is required but missing")
            }
        }
    }
}

impl std::error::Error for StageFrameError {}

pub fn validate_stage_control_request(
    frame: &proto::stage::StageControlRequest,
) -> Result<(), StageFrameError> {
    validate_generation(frame.r#gen)?;
    validate_endpoint_id(frame.requester_id.len())?;
    if frame.command.is_none() {
        return Err(StageFrameError::MissingStageControlCommand);
    }
    Ok(())
}

pub fn validate_stage_control_response(
    frame: &proto::stage::StageControlResponse,
) -> Result<(), StageFrameError> {
    validate_generation(frame.r#gen)?;
    if frame.response.is_none() {
        return Err(StageFrameError::MissingStageControlResponse);
    }
    Ok(())
}

pub fn validate_stage_transport_open(
    frame: &proto::stage::StageTransportOpen,
) -> Result<(), StageFrameError> {
    validate_generation(frame.r#gen)?;
    validate_endpoint_id(frame.requester_id.len())?;
    if frame.topology_id.is_empty() || frame.run_id.is_empty() || frame.stage_id.is_empty() {
        return Err(StageFrameError::MissingStageTransportTarget);
    }
    Ok(())
}

pub fn validate_stage_artifact_transfer_request(
    frame: &proto::stage::StageArtifactTransferRequest,
) -> Result<(), StageFrameError> {
    validate_generation(frame.r#gen)?;
    validate_endpoint_id(frame.requester_id.len())?;
    if frame.topology_id.is_empty()
        || frame.run_id.is_empty()
        || frame.stage_id.is_empty()
        || !frame.package_ref.starts_with("hf://")
    {
        return Err(StageFrameError::MissingStageArtifactTarget);
    }
    validate_artifact_digest(&frame.manifest_sha256)?;
    if let Some(expected_sha) = frame.expected_sha256.as_deref() {
        validate_artifact_digest(expected_sha)?;
    }
    if frame.expected_size.is_some_and(|size| frame.offset > size) {
        return Err(StageFrameError::InvalidArtifactOffset);
    }
    validate_safe_relative_artifact_path(&frame.relative_path)?;
    Ok(())
}

pub fn validate_stage_artifact_transfer_response(
    frame: &proto::stage::StageArtifactTransferResponse,
) -> Result<(), StageFrameError> {
    validate_generation(frame.r#gen)?;
    if let Some(sha256) = frame.sha256.as_deref() {
        validate_artifact_digest(sha256)?;
    }
    Ok(())
}

fn validate_generation(r#gen: u32) -> Result<(), StageFrameError> {
    if r#gen != STAGE_PROTOCOL_GENERATION {
        return Err(StageFrameError::BadGeneration { got: r#gen });
    }
    Ok(())
}

fn validate_artifact_digest(value: &str) -> Result<(), StageFrameError> {
    if value.len() != 64 || !value.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(StageFrameError::InvalidArtifactDigestLength { got: value.len() });
    }
    Ok(())
}

fn validate_safe_relative_artifact_path(path: &str) -> Result<(), StageFrameError> {
    use std::path::{Component, Path};

    if path.trim().is_empty() {
        return Err(StageFrameError::InvalidArtifactPath);
    }
    let path = Path::new(path);
    let mut components = path.components();
    let Some(first) = components.next() else {
        return Err(StageFrameError::InvalidArtifactPath);
    };
    if !matches!(first, Component::Normal(_))
        || !components.all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(StageFrameError::InvalidArtifactPath);
    }
    Ok(())
}

fn validate_endpoint_id(len: usize) -> Result<(), StageFrameError> {
    if len != 32 {
        return Err(StageFrameError::InvalidEndpointId { got: len });
    }
    Ok(())
}
