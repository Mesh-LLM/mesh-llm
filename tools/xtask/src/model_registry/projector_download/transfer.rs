use crate::{command::DynResult, process};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};
use url::Url;

pub(super) fn clean(report: &process::ProcessReport) -> bool {
    report.outcome == process::Outcome::Exited
        && report.status.is_some_and(|s| s.success())
        && report.failure.is_none()
        && report.cleanup.complete
        && !report.cleanup.forced
        && !report.cleanup.graceful_signal_failed
        && report.cleanup.failure.is_none()
}

pub(super) fn download(
    input: &str,
    output: &Path,
    cancellation: &process::Cancellation,
) -> DynResult<String> {
    let started = Instant::now();
    let mut url = super::url_policy::trusted(input)?;
    let curl = process::curl_https::Curl::discover_for(remaining(started)?, cancellation)?;
    let parent = output.parent().ok_or("projector output parent absent")?;
    std::fs::create_dir_all(parent)?;
    let directory = tempfile::tempdir_in(parent)?;
    let result: DynResult<String> = (|| {
        for hop in 0..=10 {
            let host = url.host_str().ok_or("projector host absent")?;
            let addresses = super::resolver::resolve(host, remaining(started)?, cancellation)?;
            let pin = super::url_policy::pins(host, &addresses)?;
            match exchange(
                &curl,
                &url,
                &pin,
                directory.path(),
                remaining(started)?,
                cancellation,
                64 * 1024 * 1024 * 1024,
            )? {
                Reply::Redirect(location) => {
                    if hop == 10 {
                        return Err("projector redirect limit exceeded".into());
                    }
                    url = super::url_policy::trusted(url.join(&location)?.as_str())?;
                }
                Reply::Complete(body) => {
                    return publish(
                        &body,
                        output,
                        64 * 1024 * 1024 * 1024,
                        started,
                        cancellation,
                    );
                }
            }
        }
        Err("projector redirect limit exceeded".into())
    })();
    let close = directory.close();
    match (result, close) {
        (Ok(hash), Ok(())) => Ok(hash),
        (Err(error), Ok(())) => Err(error),
        (result, Err(error)) => Err(format!(
            "projector result {}; private-state cleanup failed: {error}",
            result
                .as_ref()
                .err()
                .map_or("successful".into(), ToString::to_string)
        )
        .into()),
    }
}
fn remaining(started: Instant) -> DynResult<Duration> {
    let duration = Duration::from_secs(60).saturating_sub(started.elapsed());
    if duration.is_zero() {
        return Err("projector total deadline exceeded".into());
    }
    Ok(duration)
}

pub(super) enum Reply {
    Redirect(String),
    Complete(std::path::PathBuf),
}
pub(super) fn exchange(
    curl: &process::curl_https::Curl,
    url: &Url,
    pin: &str,
    directory: &Path,
    budget: Duration,
    cancellation: &process::Cancellation,
    maximum: u64,
) -> DynResult<Reply> {
    let execution = process::curl_https::execution_budget(budget)?;
    let body = directory.join("body");
    let spec = curl.pinned_get(
        url.as_str(),
        pin,
        process::curl_https::Files {
            directory,
            body: &body,
            headers: Path::new("-"),
            payload: None,
        },
        execution,
        maximum,
    );
    let raw = process::supervise_raw(
        &spec,
        &process::curl_https::limits(execution),
        cancellation,
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(4096),
        },
    )?;
    if !clean(&raw.process) {
        return Err(format!(
            "projector HTTPS transport failed: {:?}; cleanup {:?}",
            raw.process.outcome, raw.process.cleanup
        )
        .into());
    }
    let headers = raw.stdout.ok_or("projector headers absent")?;
    if headers.as_bytes().len() as u64 != raw.process.stdout.bytes_seen
        || std::fs::metadata(&body)?.len() > maximum
    {
        return Err("projector header or body limit exceeded".into());
    }
    let (status, location, length) = headers_final(headers.as_bytes())?;
    if length.is_some_and(|length| length > maximum) {
        return Err("projector declared body exceeds limit".into());
    }
    if [301, 302, 303, 307, 308].contains(&status) {
        return Ok(Reply::Redirect(location.ok_or("redirect Location absent")?));
    }
    if status != 200 {
        return Err("projector HTTP status refused".into());
    }
    if length.is_some_and(|length| std::fs::metadata(&body).map_or(true, |m| m.len() != length)) {
        return Err("projector body incomplete".into());
    }
    Ok(Reply::Complete(body))
}
fn headers_final(bytes: &[u8]) -> DynResult<(u16, Option<String>, Option<u64>)> {
    if !bytes.ends_with(b"\r\n\r\n") {
        return Err("incomplete response headers".into());
    }
    let text = std::str::from_utf8(bytes)?;
    let mut final_headers = None;
    for block in text.split("\r\n\r\n").filter(|s| !s.is_empty()) {
        let mut lines = block.split("\r\n");
        let first = lines.next().ok_or("HTTP status absent")?;
        let mut tokens = first.split_whitespace();
        let protocol = tokens.next().ok_or("HTTP protocol absent")?;
        if !matches!(protocol, "HTTP/1.1" | "HTTP/1.0" | "HTTP/2" | "HTTP/3") {
            return Err("invalid HTTP protocol".into());
        }
        let status: u16 = tokens.next().ok_or("HTTP status absent")?.parse()?;
        if !(100..600).contains(&status) {
            return Err("invalid HTTP status".into());
        }
        let mut location = None;
        let mut length = None;
        for line in lines {
            let (name, value) = line.split_once(':').ok_or("malformed HTTP header")?;
            let value = value.trim();
            if name.eq_ignore_ascii_case("location")
                && location.replace(value.to_string()).is_some()
            {
                return Err("ambiguous redirect Location".into());
            }
            if name.eq_ignore_ascii_case("content-length")
                && length.replace(value.parse::<u64>()?).is_some()
            {
                return Err("ambiguous Content-Length".into());
            }
        }
        if status >= 200 && final_headers.replace((status, location, length)).is_some() {
            return Err("multiple terminal response headers".into());
        }
    }
    final_headers.ok_or("terminal response headers absent".into())
}
fn publish(
    body: &Path,
    output: &Path,
    maximum: u64,
    started: Instant,
    cancellation: &process::Cancellation,
) -> DynResult<String> {
    remaining(started)?;
    if cancellation.is_cancelled() {
        return Err("projector publication cancelled".into());
    }
    let mut input = std::fs::File::open(body)?;
    let mut magic = [0; 4];
    input.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err("invalid projector GGUF magic".into());
    }
    let mut temporary =
        tempfile::NamedTempFile::new_in(output.parent().ok_or("output parent absent")?)?;
    let mut hash = Sha256::new();
    hash.update(magic);
    temporary.write_all(&magic)?;
    let mut count = 4u64;
    let mut buffer = vec![0; 8 * 1024 * 1024];
    loop {
        remaining(started)?;
        if cancellation.is_cancelled() {
            return Err("projector publication cancelled".into());
        }
        let size = input.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        count = count
            .checked_add(size as u64)
            .ok_or("projector byte count overflow")?;
        if count > maximum {
            return Err("projector body exceeds limit".into());
        }
        hash.update(&buffer[..size]);
        temporary.write_all(&buffer[..size])?;
    }
    temporary.flush()?;
    remaining(started)?;
    if cancellation.is_cancelled() {
        return Err("projector publication cancelled".into());
    }
    temporary.persist(output)?;
    Ok(hex::encode(hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hf_projector_bounded_publication_preserves_existing_target_on_magic_limit_or_cancel_refusal()
    {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("output");
        let body = directory.path().join("body");
        for (bytes, maximum, cancel) in [
            (b"noGGUF".as_slice(), 64, false),
            (b"GGUFoverflow".as_slice(), 4, false),
            (b"GGUFvalid".as_slice(), 64, true),
        ] {
            std::fs::write(&output, b"keep").unwrap();
            std::fs::write(&body, bytes).unwrap();
            let cancellation = process::Cancellation::default();
            if cancel {
                cancellation.cancel();
            }
            assert!(publish(&body, &output, maximum, Instant::now(), &cancellation).is_err());
            assert_eq!(std::fs::read(&output).unwrap(), b"keep");
        }
        std::fs::write(&body, b"GGUFvalid").unwrap();
        let hash = publish(
            &body,
            &output,
            64,
            Instant::now(),
            &process::Cancellation::default(),
        )
        .unwrap();
        assert_eq!(hash, hex::encode(Sha256::digest(b"GGUFvalid")));
        assert_eq!(std::fs::read(&output).unwrap(), b"GGUFvalid");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 2);
        directory.close().unwrap();
    }
}
