//! Decode exactly the bounded HTTP entity that would be released to a client.
use super::{
    ImageResponseSafetyPlan, ImageSafetyResponseError, Node, SafetyResult, artifacts, screen_image,
};

pub(crate) async fn screen_response(
    node: &Node,
    plan: &ImageResponseSafetyPlan,
    wire: &[u8],
) -> SafetyResult<()> {
    let (content_type, body) = entity(wire).map_err(|_| {
        tracing::debug!(stage = "http_entity_decode", "image safety response failed");
        ImageSafetyResponseError::Unavailable
    })?;
    static DECODERS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(2);
    let permit = tokio::time::timeout(std::time::Duration::from_secs(2), DECODERS.acquire())
        .await
        .map_err(|_| {
            tracing::debug!(
                stage = "artifact_decode",
                reason = "semaphore_timeout",
                "image safety response failed"
            );
            ImageSafetyResponseError::Unavailable
        })?
        .map_err(|_| {
            tracing::debug!(
                stage = "artifact_decode",
                reason = "semaphore_closed",
                "image safety response failed"
            );
            ImageSafetyResponseError::Unavailable
        })?;
    let decoder = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        artifacts::images_from_response(&content_type, &body)
    });
    let images = tokio::time::timeout(std::time::Duration::from_secs(30), decoder)
        .await
        .map_err(|_| {
            tracing::debug!(
                stage = "artifact_decode",
                reason = "worker_timeout",
                "image safety response failed"
            );
            ImageSafetyResponseError::Unavailable
        })?
        .map_err(|_| {
            tracing::debug!(
                stage = "artifact_decode",
                reason = "worker_join_failed",
                "image safety response failed"
            );
            ImageSafetyResponseError::Unavailable
        })?
        .map_err(|_| {
            tracing::debug!(
                stage = "artifact_decode",
                reason = "invalid_artifact",
                "image safety response failed"
            );
            ImageSafetyResponseError::Unavailable
        })?;
    for image in images {
        for name in &plan.guard_plugins {
            screen_image(node, name, image.data_uri.clone())
                .await
                .inspect_err(|error| {
                    tracing::debug!(stage = "image_guard", error = ?error, "image safety response failed");
                })?;
        }
    }
    Ok(())
}

fn entity(wire: &[u8]) -> anyhow::Result<(String, Vec<u8>)> {
    use anyhow::{Context, ensure};
    ensure!(
        wire.len() <= artifacts::MAX_RESPONSE_BYTES,
        "image response exceeds limit"
    );
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut response = httparse::Response::new(&mut headers);
    let start = match response.parse(wire)? {
        httparse::Status::Complete(start) => start,
        httparse::Status::Partial => anyhow::bail!("incomplete response headers"),
    };
    ensure!(start <= MAX_HEADER_BYTES, "response headers exceed limit");
    // Following a redirect would fetch bytes outside this inline artifact gate.
    ensure!(
        response
            .code
            .is_some_and(|code| (200..=599).contains(&code) && !(300..=399).contains(&code)),
        "unexpected response status"
    );
    ensure!(
        wire[..start].ends_with(b"\r\n\r\n"),
        "noncanonical response headers"
    );
    ensure!(
        !wire[..start]
            .split(|&byte| byte == b'\n')
            .any(|line| !line.is_empty() && !line.ends_with(b"\r")),
        "noncanonical response header line ending"
    );
    let framing = Framing::from_headers(response.headers)?;
    let body = &wire[start..];
    let decoded = if framing.chunked {
        chunked_body(body, artifacts::MAX_RESPONSE_BYTES, &framing.trailers)?
    } else if let Some(length) = framing.length {
        ensure!(body.len() == length, "incomplete or trailing response body");
        body.to_vec()
    } else {
        ensure!(
            response.version == Some(0) || framing.close,
            "ambiguous response body framing"
        );
        body.to_vec()
    };
    Ok((
        framing
            .content_type
            .context("missing response content type")?,
        decoded,
    ))
}

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_CHUNK_LINE_BYTES: usize = 1024;
const MAX_TRAILER_LINE_BYTES: usize = 8192;
const MAX_CHUNKS: usize = 65_536;

#[derive(Default)]
struct Framing {
    content_type: Option<String>,
    length: Option<usize>,
    chunked: bool,
    close: bool,
    trailers: std::collections::BTreeSet<String>,
}

impl Framing {
    fn from_headers(headers: &[httparse::Header<'_>]) -> anyhow::Result<Self> {
        use anyhow::{Context, ensure};
        let mut result = Self::default();
        let mut seen = std::collections::BTreeSet::new();
        for header in headers {
            let name = header.name.to_ascii_lowercase();
            if !matches!(
                name.as_str(),
                "content-type"
                    | "content-length"
                    | "transfer-encoding"
                    | "content-encoding"
                    | "connection"
                    | "trailer"
            ) {
                continue;
            }
            ensure!(
                seen.insert(name.clone()),
                "duplicate response framing header"
            );
            let value = std::str::from_utf8(header.value)
                .context("invalid response header encoding")?
                .trim();
            match name.as_str() {
                "content-type" => {
                    ensure!(!value.is_empty(), "empty response content type");
                    result.content_type = Some(value.to_owned());
                }
                "content-length" => {
                    ensure!(
                        !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()),
                        "invalid content length"
                    );
                    let length = value.parse()?;
                    ensure!(
                        length <= artifacts::MAX_RESPONSE_BYTES,
                        "response content length exceeds limit"
                    );
                    result.length = Some(length);
                }
                "transfer-encoding" => {
                    ensure!(
                        value.eq_ignore_ascii_case("chunked"),
                        "unsupported transfer encoding"
                    );
                    result.chunked = true;
                }
                "content-encoding" => ensure!(
                    value.eq_ignore_ascii_case("identity"),
                    "unsupported content encoding"
                ),
                "connection" => {
                    for token in header_tokens(value)? {
                        ensure!(
                            !forbidden_trailer(&token),
                            "connection nominates a framing header"
                        );
                        result.close |= token == "close";
                    }
                }
                "trailer" => {
                    for token in header_tokens(value)? {
                        ensure!(!forbidden_trailer(&token), "forbidden trailer declaration");
                        ensure!(
                            result.trailers.insert(token),
                            "duplicate trailer declaration"
                        );
                    }
                }
                _ => unreachable!(),
            }
        }
        ensure!(
            !(result.chunked && result.length.is_some()),
            "conflicting response framing"
        );
        ensure!(
            result.trailers.is_empty() || result.chunked,
            "trailers require chunked framing"
        );
        Ok(result)
    }
}

fn header_tokens(value: &str) -> anyhow::Result<Vec<String>> {
    use anyhow::ensure;
    value
        .split(',')
        .map(|value| {
            let token = value.trim();
            ensure!(
                !token.is_empty() && token.bytes().all(is_token),
                "invalid header token"
            );
            Ok(token.to_ascii_lowercase())
        })
        .collect()
}

fn is_token(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte)
}

fn forbidden_trailer(name: &str) -> bool {
    matches!(
        name,
        "content-type"
            | "content-length"
            | "content-encoding"
            | "transfer-encoding"
            | "connection"
            | "trailer"
            | "host"
            | "authorization"
            | "proxy-authorization"
            | "content-range"
    )
}

fn chunked_body(
    wire: &[u8],
    limit: usize,
    declared: &std::collections::BTreeSet<String>,
) -> anyhow::Result<Vec<u8>> {
    use anyhow::{Context, bail, ensure};
    let mut cursor = 0usize;
    let mut decoded = Vec::new();
    for _ in 0..MAX_CHUNKS {
        let size = chunk_size(read_line(wire, &mut cursor, MAX_CHUNK_LINE_BYTES)?)?;
        if size == 0 {
            validate_trailers(wire, &mut cursor, declared)?;
            ensure!(cursor == wire.len(), "bytes after chunked response");
            return Ok(decoded);
        }
        ensure!(
            size <= limit - decoded.len(),
            "decoded image response exceeds limit"
        );
        let end = cursor.checked_add(size).context("chunk size overflow")?;
        decoded.extend_from_slice(wire.get(cursor..end).context("incomplete response chunk")?);
        ensure!(
            wire.get(end..end + 2) == Some(b"\r\n"),
            "invalid response chunk terminator"
        );
        cursor = end + 2;
    }
    bail!("too many response chunks")
}

fn read_line<'a>(wire: &'a [u8], cursor: &mut usize, limit: usize) -> anyhow::Result<&'a [u8]> {
    use anyhow::{Context, ensure};
    let tail = wire
        .get(*cursor..)
        .context("invalid response framing offset")?;
    let scanned = &tail[..tail.len().min(limit + 2)];
    let length = scanned
        .windows(2)
        .position(|pair| pair == b"\r\n")
        .context("incomplete or oversized response framing line")?;
    ensure!(length <= limit, "response framing line exceeds limit");
    *cursor += length + 2;
    Ok(&tail[..length])
}

fn chunk_size(line: &[u8]) -> anyhow::Result<usize> {
    use anyhow::{Context, ensure};
    let end = line
        .iter()
        .position(|b| !b.is_ascii_hexdigit())
        .unwrap_or(line.len());
    ensure!(end > 0 && end <= 16, "invalid response chunk size");
    validate_chunk_extensions(&line[end..])?;
    usize::from_str_radix(std::str::from_utf8(&line[..end])?, 16)
        .context("invalid response chunk size")
}

fn validate_chunk_extensions(mut rest: &[u8]) -> anyhow::Result<()> {
    use anyhow::ensure;
    while !rest.is_empty() {
        rest = trim_ows(rest);
        if rest.is_empty() {
            return Ok(());
        }
        ensure!(
            rest.first() == Some(&b';'),
            "invalid chunk extension delimiter"
        );
        rest = trim_ows(&rest[1..]);
        let end = rest
            .iter()
            .position(|&b| !is_token(b))
            .unwrap_or(rest.len());
        ensure!(end > 0, "missing chunk extension name");
        rest = trim_ows(&rest[end..]);
        if rest.first() == Some(&b'=') {
            rest = trim_ows(&rest[1..]);
            rest = if rest.first() == Some(&b'\"') {
                quoted_extension(rest)?
            } else {
                let end = rest
                    .iter()
                    .position(|&b| !is_token(b))
                    .unwrap_or(rest.len());
                ensure!(end > 0, "missing chunk extension value");
                &rest[end..]
            };
        }
    }
    Ok(())
}

fn trim_ows(value: &[u8]) -> &[u8] {
    let start = value
        .iter()
        .position(|byte| !matches!(byte, b' ' | b'\t'))
        .unwrap_or(value.len());
    &value[start..]
}

fn quoted_extension(rest: &[u8]) -> anyhow::Result<&[u8]> {
    use anyhow::{bail, ensure};
    let mut cursor = 1usize;
    while let Some(&byte) = rest.get(cursor) {
        cursor += 1;
        if byte == b'\"' {
            return Ok(&rest[cursor..]);
        }
        if byte == b'\\' {
            let escaped = *rest
                .get(cursor)
                .ok_or_else(|| anyhow::anyhow!("incomplete quoted chunk extension"))?;
            ensure!(
                escaped == b'\t' || (0x20..=0x7e).contains(&escaped),
                "invalid quoted chunk escape"
            );
            cursor += 1;
        } else {
            ensure!(
                byte == b'\t' || (0x20..=0x7e).contains(&byte),
                "invalid quoted chunk extension"
            );
        }
    }
    bail!("incomplete quoted chunk extension")
}

fn validate_trailers(
    wire: &[u8],
    cursor: &mut usize,
    declared: &std::collections::BTreeSet<String>,
) -> anyhow::Result<()> {
    use anyhow::{Context, bail, ensure};
    let start = *cursor;
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..=64 {
        let line = read_line(wire, cursor, MAX_TRAILER_LINE_BYTES)?;
        ensure!(
            *cursor - start <= MAX_HEADER_BYTES,
            "response trailers exceed limit"
        );
        if line.is_empty() {
            return Ok(());
        }
        let colon = line
            .iter()
            .position(|&b| b == b':')
            .context("invalid response trailer")?;
        ensure!(
            colon > 0 && line[..colon].iter().copied().all(is_token),
            "invalid response trailer name"
        );
        let name = std::str::from_utf8(&line[..colon])?.to_ascii_lowercase();
        ensure!(!forbidden_trailer(&name), "forbidden response trailer");
        ensure!(seen.insert(name.clone()), "duplicate response trailer");
        ensure!(
            declared.is_empty() || declared.contains(&name),
            "undeclared response trailer"
        );
        ensure!(
            line[colon + 1..]
                .iter()
                .all(|&byte| byte == b'\t' || byte >= 0x20 && byte != 0x7f),
            "invalid response trailer value"
        );
    }
    bail!("too many response trailers")
}

#[cfg(test)]
mod tests;
