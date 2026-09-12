//! Body transformations operate on decoded payloads, never HTTP chunk framing.

use super::audio_multipart::multipart_model_value_range;
use super::{BufferedHttpRequest, MAX_BODY_BYTES, MAX_HEADERS, try_decode_chunked_body};

#[cfg(test)]
mod tests;

struct BodyHeaders<'a> {
    end: usize,
    content_type: Option<&'a str>,
    chunked: bool,
}

impl BodyHeaders<'_> {
    fn permits_json(&self) -> bool {
        self.content_type.is_none_or(|value| {
            value.split(';').next().is_some_and(|media_type| {
                media_type.trim().eq_ignore_ascii_case("application/json")
            })
        })
    }
}

fn body_headers(raw: &[u8]) -> Option<BodyHeaders<'_>> {
    let mut headers = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut parsed = httparse::Request::new(&mut headers);
    let httparse::Status::Complete(end) = parsed.parse(raw).ok()? else {
        return None;
    };
    let content_type = parsed
        .headers
        .iter()
        .find(|header| header.name.eq_ignore_ascii_case("content-type"))
        .map(|header| std::str::from_utf8(header.value))
        .transpose()
        .ok()?;
    let chunked = parsed
        .headers
        .iter()
        .any(|header| header.name.eq_ignore_ascii_case("transfer-encoding"));
    Some(BodyHeaders {
        end,
        content_type,
        chunked,
    })
}

/// Replace framing with one Content-Length, preserving all other header bytes.
fn replace_body(raw: &mut Vec<u8>, header_end: usize, body: &[u8]) {
    let mut rebuilt = Vec::with_capacity(header_end + body.len());
    for line in raw[..header_end].split_inclusive(|byte| *byte == b'\n') {
        let name = line.split(|byte| *byte == b':').next().unwrap_or_default();
        if line == b"\r\n"
            || line == b"\n"
            || name.eq_ignore_ascii_case(b"content-length")
            || name.eq_ignore_ascii_case(b"transfer-encoding")
            || name.eq_ignore_ascii_case(b"trailer")
        {
            continue;
        }
        rebuilt.extend_from_slice(line);
    }
    rebuilt.extend_from_slice(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes());
    rebuilt.extend_from_slice(body);
    *raw = rebuilt;
}

/// Set the hook flag only on an actual JSON object. Multipart and binary
/// requests remain byte-identical, even when their media contains `{`.
pub fn inject_mesh_hooks_flag(raw: &mut Vec<u8>, enabled: bool) {
    let Some(headers) = body_headers(raw) else {
        return;
    };
    if !headers.permits_json() {
        return;
    }
    let decoded;
    let body = if headers.chunked {
        let Ok(Some((_, bytes))) = try_decode_chunked_body(&raw[headers.end..], MAX_BODY_BYTES)
        else {
            return;
        };
        decoded = bytes;
        &decoded[..]
    } else {
        &raw[headers.end..]
    };
    let Ok(mut json) = serde_json::from_slice::<serde_json::Value>(body) else {
        return;
    };
    let Some(object) = json.as_object_mut() else {
        return;
    };
    object.insert("mesh_hooks".into(), enabled.into());
    let Ok(body) = serde_json::to_vec(&json) else {
        return;
    };
    let end = headers.end;
    replace_body(raw, end, &body);
}

fn rebuild_request_body(
    request: &mut BufferedHttpRequest,
    header_end: usize,
    body: Vec<u8>,
    json: Option<serde_json::Value>,
    model: &str,
) {
    replace_body(&mut request.raw, header_end, &body);
    request.body_len_bytes = body.len();
    request.body_bytes = Some(body);
    request.body_json = json;
    request.body_json_attempted = true;
    request.model_name = Some(model.to_string());
}

/// Rewrite the JSON or multipart model field using the reader's decoded body
/// for chunked uploads. Only the model part changes; binary media stays intact.
pub fn rewrite_model_field(request: &mut BufferedHttpRequest, model: &str) {
    let Some(headers) = body_headers(&request.raw) else {
        return;
    };
    let original = if headers.chunked {
        let Some(body) = request.body_bytes.as_deref() else {
            return;
        };
        body
    } else {
        &request.raw[headers.end..]
    };
    if headers.permits_json() {
        let Ok(mut json) = serde_json::from_slice::<serde_json::Value>(original) else {
            return;
        };
        let Some(object) = json.as_object_mut() else {
            return;
        };
        object.insert("model".into(), model.into());
        let Ok(body) = serde_json::to_vec(&json) else {
            return;
        };
        let end = headers.end;
        rebuild_request_body(request, end, body, Some(json), model);
        return;
    }
    let Some(content_type) = headers.content_type else {
        return;
    };
    let Ok(Some(range)) = multipart_model_value_range(content_type, original) else {
        return;
    };
    let mut body = Vec::with_capacity(original.len() - range.len() + model.len());
    body.extend_from_slice(&original[..range.start]);
    body.extend_from_slice(model.as_bytes());
    body.extend_from_slice(&original[range.end..]);
    let end = headers.end;
    rebuild_request_body(request, end, body, None, model);
}
