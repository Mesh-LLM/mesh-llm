use anyhow::{Context, Result, bail};

use super::{CRLF, CRLF_HEADER_TERMINATOR, MAX_HEADER_BYTES};

/// Extract an HTTP form boundary only if it satisfies the bounded ASCII grammar.
pub(super) fn multipart_boundary(content_type: &str) -> Option<&str> {
    let mut parts = content_type.split(';');
    if !parts
        .next()?
        .trim()
        .eq_ignore_ascii_case("multipart/form-data")
    {
        return None;
    }
    let boundary = parts.find_map(|part| {
        let (name, value) = part.trim().split_once('=')?;
        name.trim()
            .eq_ignore_ascii_case("boundary")
            .then_some(value.trim().trim_matches('"'))
    })?;
    let valid = !boundary.is_empty()
        && boundary.len() <= 70
        && boundary
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"'()+_,-./:=?".contains(&byte));
    valid.then_some(boundary)
}

/// Find a nonempty byte marker without decoding the surrounding media payload.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    (!needle.is_empty())
        .then(|| {
            haystack
                .windows(needle.len())
                .position(|window| window == needle)
        })
        .flatten()
}

/// Ignore marker-like file bytes unless followed by a valid delimiter suffix.
fn find_multipart_boundary(body: &[u8], marker: &[u8], from: usize) -> Option<usize> {
    let mut cursor = from;
    while let Some(offset) = find_subslice(&body[cursor..], marker) {
        let start = cursor + offset;
        let suffix = body.get(start + marker.len()..start + marker.len() + 2);
        if suffix == Some(CRLF) || suffix == Some(b"--") {
            return Some(start);
        }
        cursor = start + 1;
    }
    None
}

/// Split parameters outside quoted strings and reject unterminated escaping.
fn disposition_parameters(value: &str) -> Result<Vec<&str>> {
    let mut parameters = Vec::new();
    let mut start = 0;
    let mut quoted = false;
    let mut escaped = false;
    for (index, byte) in value.bytes().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        match byte {
            b'\\' if quoted => escaped = true,
            b'"' => quoted = !quoted,
            b';' if !quoted => {
                parameters.push(&value[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if quoted || escaped {
        bail!("malformed multipart Content-Disposition");
    }
    parameters.push(&value[start..]);
    Ok(parameters)
}

/// Identify the model field without mistaking quoted filename text for its name.
pub(super) fn multipart_part_is_model(headers: &str) -> Result<bool> {
    let mut disposition = None;
    for line in headers.split("\r\n") {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("content-disposition")
            && disposition.replace(value.trim()).is_some()
        {
            bail!("duplicate multipart Content-Disposition header");
        }
    }
    let Some(disposition) = disposition else {
        return Ok(false);
    };
    let parameters = disposition_parameters(disposition)?;
    if !parameters[0].trim().eq_ignore_ascii_case("form-data") {
        return Ok(false);
    }
    let mut field_name = None;
    for parameter in parameters.iter().skip(1) {
        let Some((name, value)) = parameter.trim().split_once('=') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case("name") {
            if field_name.is_some() {
                bail!("duplicate multipart field name parameter");
            }
            let value = value.trim();
            let parsed = if let Some(inner) = value
                .strip_prefix('"')
                .and_then(|quoted| quoted.strip_suffix('"'))
            {
                inner
            } else if !value.contains('"') {
                value
            } else {
                bail!("malformed multipart field name parameter");
            };
            field_name = Some(parsed);
        }
    }
    Ok(field_name == Some("model"))
}

/// Locate the unique model field while validating framing and bounded part headers.
pub(super) fn multipart_model_value_range(
    content_type: &str,
    body: &[u8],
) -> Result<Option<std::ops::Range<usize>>> {
    let Some(boundary) = multipart_boundary(content_type) else {
        return Ok(None);
    };
    let delimiter = format!("--{boundary}").into_bytes();
    if !body.starts_with(&delimiter) {
        bail!("multipart body must start with its declared boundary");
    }
    let next_marker = [CRLF, &delimiter].concat();
    let mut cursor = 0;
    let mut model_range = None;
    loop {
        let part_start = cursor + delimiter.len();
        if body.get(part_start..part_start + 2) == Some(b"--") {
            return Ok(model_range);
        }
        if body.get(part_start..part_start + 2) != Some(CRLF) {
            bail!("malformed multipart boundary");
        }
        let content_start = part_start + CRLF.len();
        let header_search_end = content_start
            .saturating_add(MAX_HEADER_BYTES + CRLF_HEADER_TERMINATOR.len())
            .min(body.len());
        let headers_end = find_subslice(
            &body[content_start..header_search_end],
            CRLF_HEADER_TERMINATOR,
        )
        .map(|offset| content_start + offset)
        .context("multipart part is missing a header terminator")?;
        if headers_end.saturating_sub(content_start) > MAX_HEADER_BYTES {
            bail!("multipart part headers exceed the limit");
        }
        let headers = std::str::from_utf8(&body[content_start..headers_end])
            .context("multipart part headers are not UTF-8")?;
        let is_model = multipart_part_is_model(headers)?;
        let value_start = headers_end + CRLF_HEADER_TERMINATOR.len();
        let value_end = find_multipart_boundary(body, &next_marker, value_start)
            .context("multipart part is missing its closing boundary")?;
        if is_model {
            if model_range.is_some() {
                bail!("duplicate multipart model field");
            }
            model_range = Some(value_start..value_end);
        }
        cursor = value_end + CRLF.len();
    }
}

/// Read the routed model as bounded UTF-8, leaving all uploaded file bytes intact.
pub(super) fn multipart_model_field(content_type: &str, body: &[u8]) -> Result<Option<String>> {
    let Some(range) = multipart_model_value_range(content_type, body)? else {
        return Ok(None);
    };
    let value = std::str::from_utf8(&body[range])?.trim();
    Ok((!value.is_empty() && value.len() <= 256).then(|| value.to_string()))
}
