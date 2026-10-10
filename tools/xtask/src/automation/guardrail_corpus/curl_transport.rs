//! Synchronous owned curl runs retain cleanup inside the caller's absolute deadline.
use super::transport::Response;
use crate::{
    command::DynResult,
    process::{
        self, Cancellation,
        curl_https::{Curl, Files, Request},
    },
};
use serde_json::Value;
use std::{
    io::{Read, Write},
    path::Path,
    time::Instant,
};
const BODY_LIMIT: usize = 1024 * 1024;

fn remaining(deadline: Instant, cancellation: &Cancellation) -> DynResult<std::time::Duration> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() || cancellation.is_cancelled() {
        return Err("guardrail curl deadline/cancellation".into());
    }
    Ok(remaining)
}
fn read(path: &Path, maximum: usize) -> DynResult<Vec<u8>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take((maximum + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > maximum {
        return Err("guardrail curl bounded response exceeded".into());
    }
    Ok(bytes)
}
fn status(bytes: &[u8]) -> DynResult<u16> {
    if bytes.len() != 3 || !bytes.iter().all(u8::is_ascii_digit) {
        return Err("guardrail invalid bounded curl status".into());
    }
    let code = std::str::from_utf8(bytes)?.parse::<u16>()?;
    if !(200..600).contains(&code) {
        return Err("guardrail curl final status absent".into());
    }
    Ok(code)
}
fn clean(report: &process::ProcessReport) -> bool {
    report.success()
        && report.failure.is_none()
        && report.cleanup.complete
        && !report.cleanup.forced
        && !report.cleanup.graceful_signal_failed
        && report.cleanup.failure.is_none()
        && report.stdout.line_capture_complete
        && report.stderr.line_capture_complete
}
pub(super) fn exchange(
    endpoint: &str,
    body: Option<&Value>,
    deadline: Instant,
    cancellation: &Cancellation,
    authorization: Option<&str>,
) -> DynResult<Response> {
    exchange_mode(endpoint, body, deadline, cancellation, authorization, false)
}
pub(super) fn exchange_private(
    endpoint: &str,
    body: Option<&Value>,
    deadline: Instant,
    cancellation: &Cancellation,
    authorization: Option<&str>,
) -> DynResult<Response> {
    exchange_mode(endpoint, body, deadline, cancellation, authorization, true)
}
fn exchange_mode(
    endpoint: &str,
    body: Option<&Value>,
    deadline: Instant,
    cancellation: &Cancellation,
    authorization: Option<&str>,
    private: bool,
) -> DynResult<Response> {
    let curl = Curl::discover_for(remaining(deadline, cancellation)?, cancellation)?;
    let directory = tempfile::Builder::new()
        .prefix("guardrail-curl-")
        .tempdir()?;
    let root = directory.path().canonicalize()?;
    let response = root.join("body");
    let headers = root.join("headers");
    let payload = root.join("payload");
    if let Some(body) = body {
        let bytes = serde_json::to_vec(body)?;
        if bytes.len() > BODY_LIMIT {
            return Err("guardrail JSON request exceeds 1 MiB".into());
        }
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&payload)?;
        file.write_all(&bytes)?;
        file.flush()?;
    }
    let execution = process::curl_https::execution_budget(remaining(deadline, cancellation)?)?;
    let method = if private {
        Curl::json_private_specification
    } else {
        Curl::json_specification
    };
    let spec = method(
        &curl,
        &Request {
            endpoint: endpoint.into(),
            method: if body.is_some() {
                hyper::Method::POST
            } else {
                hyper::Method::GET
            },
            token: authorization.unwrap_or(""),
        },
        Files {
            directory: &root,
            body: &response,
            headers: &headers,
            payload: body.map(|_| payload.as_path()),
        },
        execution,
        BODY_LIMIT as u64,
    )?;
    let raw = process::supervise_raw(
        &spec,
        &process::curl_https::limits(execution),
        cancellation,
        process::RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(16),
            stderr: std::num::NonZeroUsize::new(4096),
        },
    )?;
    let report = &raw.process;
    let code = raw
        .stdout
        .as_ref()
        .ok_or("guardrail curl status unavailable")?
        .as_bytes();
    if code.len() as u64 != report.stdout.bytes_seen
        || report.stdout.truncated
        || report.stdout.suppressed_lines != 0
    {
        return Err("guardrail curl status capture incomplete".into());
    }
    if cancellation.is_cancelled() || Instant::now() >= deadline || !clean(report) {
        return Err("guardrail curl transfer/cleanup incomplete".into());
    }
    let reply = Response {
        status: status(code)?,
        body: read(&response, BODY_LIMIT)?,
    };
    directory.close()?;
    Ok(reply)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_status_preserves_http_errors_and_refuses_incomplete_or_ambiguous_codes() {
        assert_eq!(status(b"503").unwrap(), 503);
        for bytes in [
            b"".as_slice(),
            b"20",
            b"200200",
            b"200\n",
            b"000",
            b"100",
            b"bad",
        ] {
            assert!(status(bytes).is_err());
        }
    }
}
