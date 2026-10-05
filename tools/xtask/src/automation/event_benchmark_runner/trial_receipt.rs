//! Bounded receipts after retained cleanup; never infer an end-of-trial health snapshot.
use super::{health_log, measurement_worker, stream_metrics, worker_frontends};
use crate::command::DynResult;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

const MAX_LOG: u64 = 64 * 1024 * 1024;
const MAX_HEALTH_LINE: usize = 16 * 1024;
#[cfg(test)]
const MAX_RECEIPT: u64 = 64 * 1024;

fn regular(path: &Path) -> DynResult<File> {
    if !std::fs::symlink_metadata(path)?.file_type().is_file() {
        return Err("trial evidence must be a regular file".into());
    }
    Ok(File::open(path)?)
}

fn final_stream(path: &Path) -> DynResult<Option<health_log::Observation>> {
    let mut reader = BufReader::new(regular(path)?);
    let mut total = 0_u64;
    let mut pending = Vec::new();
    let mut oversized = false;
    let mut observed = None;
    loop {
        let bytes = reader.fill_buf()?;
        if bytes.is_empty() {
            if !oversized && let Some(value) = health_log::line(&pending) {
                observed = Some(value);
            }
            return Ok(observed);
        }
        let count = bytes
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(bytes.len(), |n| n + 1);
        total += count as u64;
        if total > MAX_LOG {
            return Err("trial capture exceeds bounded health scan".into());
        }
        let ended = bytes[count - 1] == b'\n';
        let content = &bytes[..count - usize::from(ended)];
        if pending.len() + content.len() > MAX_HEALTH_LINE {
            pending.clear();
            oversized = true;
        } else if !oversized {
            pending.extend_from_slice(content);
        }
        if ended {
            if !oversized && let Some(value) = health_log::line(&pending) {
                observed = Some(value);
            }
            pending.clear();
            oversized = false;
        }
        reader.consume(count);
    }
}

pub(super) fn final_health(directory: &Path) -> DynResult<health_log::Observation> {
    let stdout = final_stream(&directory.join("server.stdout.log"))?;
    let stderr = final_stream(&directory.join("server.stderr.log"))?;
    match (stdout, stderr) {
        (Some(_), Some(_)) => {
            Err("final health chronology is ambiguous across stdout and stderr".into())
        }
        (Some(value), None) | (None, Some(value)) => Ok(value),
        (None, None) => Ok(health_log::Observation::default()),
    }
}

fn timing(value: Option<f64>) -> bool {
    value.is_none_or(|n| n.is_finite() && n >= 0.0)
}

fn measurement(value: &stream_metrics::Measurement) -> bool {
    timing(Some(value.elapsed_ms))
        && timing(value.ttft_ms)
        && timing(value.decode_tok_s)
        && timing(value.decode_only_tok_s)
        && value.ttft_ms.is_none_or(|n| n <= value.elapsed_ms)
        && equal_rate(
            value.decode_tok_s,
            stream_metrics::decode_rate(value.completion_tokens, Some(value.elapsed_ms)),
        )
        && equal_rate(
            value.decode_only_tok_s,
            stream_metrics::decode_only(
                value.completion_tokens,
                Some(value.elapsed_ms),
                value.ttft_ms,
            ),
        )
}

fn equal_rate(observed: Option<f64>, expected: Option<f64>) -> bool {
    match (observed, expected) {
        (None, None) => true,
        (Some(observed), Some(expected)) => {
            observed.is_finite()
                && expected.is_finite()
                && (observed - expected).abs() <= 1e-12 * expected.abs().max(1.0)
        }
        _ => false,
    }
}

pub(super) fn worker(
    path: &Path,
    request_sha256: &str,
    prompt_sha256: &str,
    status: Option<i32>,
) -> DynResult<measurement_worker::Evidence> {
    let evidence = worker_frontends::measurement_receipt_hash(path, request_sha256, prompt_sha256)?;
    if evidence.schema_version != 1
        || evidence.prompt_sha256 != prompt_sha256
        || !timing(evidence.readiness_ms)
        || !timing(evidence.warmup_ms)
        || evidence
            .measurement
            .as_ref()
            .is_some_and(|m| !measurement(m))
        || evidence
            .model
            .as_ref()
            .is_some_and(|model| model.trim().is_empty() || model.len() > 4096)
    {
        return Err("measurement receipt identity or timing is invalid".into());
    }
    match status {
        Some(0)
            if evidence.error.is_none()
                && evidence.model.is_some()
                && evidence.readiness_ms.is_some()
                && evidence.warmup_ms.is_some()
                && evidence
                    .measurement
                    .as_ref()
                    .is_some_and(|m| !m.malformed && m.completion_tokens.is_some()) =>
        {
            Ok(evidence)
        }
        Some(1) if evidence.error.is_some() => Ok(evidence),
        _ => Err("measurement receipt contradicts observed worker status".into()),
    }
}

#[cfg(test)]
#[path = "trial_receipt_tests.rs"]
mod tests;
