use super::{options::Options, transport::millis};
use crate::attestation::{InspectArgs, inspect_release_attestation_summary};
use serde::Serialize;
use std::{path::PathBuf, time::Instant};

#[derive(Serialize)]
pub(super) struct ResultRow {
    pub status: String,
    pub ok: bool,
    pub binary: Option<PathBuf>,
    pub expected_status: Option<String>,
    pub version: Option<u32>,
    pub signer_key_id: Option<String>,
    pub artifact_digest: Option<String>,
    pub error: Option<String>,
    pub elapsed_ms: u64,
}

pub(super) fn inspect(options: &Options) -> ResultRow {
    let started = Instant::now();
    let mut result = ResultRow {
        status: "not_configured".into(),
        ok: true,
        binary: options.binary.clone(),
        expected_status: None,
        version: None,
        signer_key_id: None,
        artifact_digest: None,
        error: None,
        elapsed_ms: 0,
    };
    let Some(binary) = &options.binary else {
        return result;
    };
    result.expected_status = Some(options.expected_attestation.clone());
    let arguments = InspectArgs {
        binary: Some(binary.clone()),
        public_key_file: options.public_key.clone(),
        json: true,
    };
    match inspect_release_attestation_summary(&arguments) {
        Ok(summary) => {
            result.ok = summary.status == options.expected_attestation;
            result.status = summary.status;
            result.version = summary.version;
            result.signer_key_id = summary.signer_key_id;
            result.artifact_digest = summary.artifact_digest;
            result.error = summary.error;
        }
        Err(error) => {
            result.status = "inspection_failed".into();
            result.ok = false;
            result.error = Some(error.to_string());
        }
    }
    result.elapsed_ms = millis(started);
    result
}
