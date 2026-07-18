use crate::mesh;
use crate::plugin;
use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use super::request_normalize::{
    ResponseAdapter, normalize_openai_compat_request, resolve_request_object_references,
};
use super::routing_rank::descriptor_for_model;

pub(super) const MAX_HEADER_BYTES: usize = 64 * 1024;
pub(super) const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_OBJECT_UPLOAD_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_CHUNKED_WIRE_BYTES: usize = MAX_BODY_BYTES * 6 + 64 * 1024;
const MAX_OBJECT_UPLOAD_CHUNKED_WIRE_BYTES: usize = MAX_OBJECT_UPLOAD_BODY_BYTES * 6 + 64 * 1024;
pub(super) const MAX_HEADERS: usize = 64;

#[derive(Debug, Clone, Copy)]
pub(super) struct HttpReadLimits {
    pub(super) max_header_bytes: usize,
    pub(super) max_body_bytes: usize,
    pub(super) max_chunked_wire_bytes: usize,
}

const HTTP_READ_LIMITS: HttpReadLimits = HttpReadLimits {
    max_header_bytes: MAX_HEADER_BYTES,
    max_body_bytes: MAX_BODY_BYTES,
    max_chunked_wire_bytes: MAX_CHUNKED_WIRE_BYTES,
};

/// Parsed header metadata extracted via httparse.
struct ParsedHeaders {
    header_end: usize,
    method: String,
    path: String,
    content_length: Option<usize>,
    is_chunked: bool,
    expects_continue: bool,
}

#[derive(Debug)]
pub struct BufferedHttpRequest {
    pub raw: Vec<u8>,
    pub method: String,
    pub path: String,
    pub client_path: String,
    pub body_json: Option<serde_json::Value>,
    pub(super) body_json_attempted: bool,
    pub(super) body_bytes: Option<Vec<u8>>,
    pub body_len_bytes: usize,
    pub completion_tokens: Option<u32>,
    pub stream: Option<bool>,
    pub model_name: Option<String>,
    pub request_object_request_ids: Vec<String>,
    pub response_adapter: ResponseAdapter,
}

impl BufferedHttpRequest {
    pub fn ensure_body_json(&mut self) {
        if self.body_json.is_none() && !self.body_json_attempted {
            self.body_json = self
                .body_bytes
                .as_deref()
                .and_then(|body| serde_json::from_slice(body).ok())
                .or_else(|| parse_json_body_from_http_request(&self.raw));
            self.body_json_attempted = true;
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct RequestMetadata {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    n_predict: Option<u32>,
}

struct RequestRewriteOutcome {
    body_json: Option<serde_json::Value>,
    request_object_request_ids: Vec<String>,
    request_path: String,
    response_adapter: ResponseAdapter,
    rewritten_body: Option<Vec<u8>>,
}

// ── Request parsing ──

/// Read and buffer one HTTP request for routing decisions.
///
/// This reads complete headers plus the full request body when body framing is
/// known via `Content-Length` or `Transfer-Encoding: chunked`. The raw request
/// bytes are preserved so the chosen upstream sees the original payload.
pub async fn read_http_request(stream: &mut TcpStream) -> Result<BufferedHttpRequest> {
    read_http_request_with_limits(stream, HTTP_READ_LIMITS, None).await
}

pub async fn read_http_request_with_plugin_manager(
    stream: &mut TcpStream,
    plugin_manager: Option<&plugin::PluginManager>,
) -> Result<BufferedHttpRequest> {
    read_http_request_with_limits(stream, HTTP_READ_LIMITS, plugin_manager).await
}

pub(super) async fn read_http_request_with_limits(
    stream: &mut TcpStream,
    limits: HttpReadLimits,
    plugin_manager: Option<&plugin::PluginManager>,
) -> Result<BufferedHttpRequest> {
    let mut raw = Vec::with_capacity(8192);
    let parsed = read_until_headers_parsed(stream, &mut raw, limits.max_header_bytes).await?;
    let body_limits = body_limits_for_path(&parsed.path, limits);
    let header_end = parsed.header_end;
    let body =
        read_buffered_request_body(stream, &mut raw, &parsed, header_end, body_limits).await?;

    let metadata = if body.is_empty() {
        None
    } else {
        serde_json::from_slice::<RequestMetadata>(&body).ok()
    };
    let requires_json_transform =
        request_requires_json_transform(&parsed.path, &body, plugin_manager.is_some());
    let rewrite = rewrite_request_body_for_forwarding(
        &parsed.path,
        &body,
        plugin_manager,
        requires_json_transform,
    )
    .await?;
    let mut response_adapter = rewrite.response_adapter;
    if response_adapter == ResponseAdapter::None
        && parsed.path.split('?').next().unwrap_or(&parsed.path) == "/v1/chat/completions"
    {
        response_adapter = if metadata.as_ref().and_then(|value| value.stream) == Some(true) {
            ResponseAdapter::OpenAiChatCompletionsStream
        } else {
            ResponseAdapter::OpenAiChatCompletionsJson
        };
    }
    let model_name = metadata.as_ref().and_then(|value| value.model.clone());
    let completion_tokens = metadata.as_ref().and_then(|value| {
        value
            .max_completion_tokens
            .or(value.max_tokens)
            .or(value.max_output_tokens)
            .or(value.n_predict)
    });
    let raw = finalize_forwarded_request(
        raw,
        header_end,
        parsed.expects_continue,
        Some(&rewrite.request_path),
        rewrite.rewritten_body.as_deref(),
    )?;
    let body_len_bytes = body.len();
    let body_bytes = if body.is_empty() { None } else { Some(body) };

    Ok(BufferedHttpRequest {
        raw,
        method: parsed.method,
        client_path: parsed.path,
        path: rewrite.request_path,
        body_json: rewrite.body_json,
        body_json_attempted: requires_json_transform,
        body_bytes,
        body_len_bytes,
        completion_tokens,
        stream: metadata.as_ref().and_then(|value| value.stream),
        model_name,
        request_object_request_ids: rewrite.request_object_request_ids,
        response_adapter,
    })
}

async fn read_buffered_request_body(
    stream: &mut TcpStream,
    raw: &mut Vec<u8>,
    parsed: &ParsedHeaders,
    header_end: usize,
    body_limits: HttpReadLimits,
) -> Result<Vec<u8>> {
    if parsed.is_chunked {
        return read_chunked_request_body(stream, raw, parsed, header_end, body_limits).await;
    }
    if let Some(content_length) = parsed.content_length {
        return read_fixed_length_request_body(
            stream,
            raw,
            parsed,
            header_end,
            content_length,
            body_limits,
        )
        .await;
    }
    raw.truncate(header_end);
    Ok(Vec::new())
}

async fn read_chunked_request_body(
    stream: &mut TcpStream,
    raw: &mut Vec<u8>,
    parsed: &ParsedHeaders,
    header_end: usize,
    body_limits: HttpReadLimits,
) -> Result<Vec<u8>> {
    let mut sent_continue = false;
    loop {
        if let Some((consumed, decoded)) =
            try_decode_chunked_body(&raw[header_end..], body_limits.max_body_bytes)?
        {
            raw.truncate(header_end + consumed);
            return Ok(decoded);
        }
        if !sent_continue && parsed.expects_continue {
            stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").await?;
            sent_continue = true;
        }
        read_more(stream, raw).await?;
        if raw.len().saturating_sub(header_end) > body_limits.max_chunked_wire_bytes {
            bail!(
                "HTTP chunked wire body exceeds {} bytes",
                body_limits.max_chunked_wire_bytes
            );
        }
    }
}

async fn read_fixed_length_request_body(
    stream: &mut TcpStream,
    raw: &mut Vec<u8>,
    parsed: &ParsedHeaders,
    header_end: usize,
    content_length: usize,
    body_limits: HttpReadLimits,
) -> Result<Vec<u8>> {
    if content_length > body_limits.max_body_bytes {
        bail!("HTTP body exceeds {} bytes", body_limits.max_body_bytes);
    }
    let body_end = header_end + content_length;
    let mut sent_continue = false;
    while raw.len() < body_end {
        if !sent_continue && parsed.expects_continue && content_length > 0 {
            stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").await?;
            sent_continue = true;
        }
        read_more(stream, raw).await?;
    }
    raw.truncate(body_end);
    Ok(raw[header_end..body_end].to_vec())
}

async fn rewrite_request_body_for_forwarding(
    path: &str,
    body: &[u8],
    plugin_manager: Option<&plugin::PluginManager>,
    requires_json_transform: bool,
) -> Result<RequestRewriteOutcome> {
    let mut outcome = RequestRewriteOutcome {
        body_json: None,
        request_object_request_ids: Vec::new(),
        request_path: path.to_string(),
        response_adapter: ResponseAdapter::None,
        rewritten_body: None,
    };
    if !requires_json_transform {
        return Ok(outcome);
    }

    outcome.body_json = serde_json::from_slice(body).ok();
    let Some(body_json) = outcome.body_json.as_mut() else {
        return Ok(outcome);
    };

    let normalization = normalize_openai_compat_request(path, body_json)?;
    let mut changed = normalization.changed;
    if let Some(rewritten_path) = normalization.rewritten_path {
        outcome.request_path = rewritten_path;
    }
    outcome.response_adapter = normalization.response_adapter;
    if let Some(plugin_manager) = plugin_manager {
        let resolved_request_ids =
            resolve_request_object_references(&outcome.request_path, body_json, plugin_manager)
                .await?;
        if !resolved_request_ids.is_empty() {
            outcome.request_object_request_ids = resolved_request_ids;
            changed = true;
        }
    }
    if changed {
        outcome.rewritten_body = Some(
            serde_json::to_vec(body_json)
                .context("serialize normalized OpenAI-compatible request body")?,
        );
    }
    Ok(outcome)
}

fn body_limits_for_path(path: &str, default: HttpReadLimits) -> HttpReadLimits {
    let path_only = path.split('?').next().unwrap_or(path);
    if path_only == "/api/objects" {
        HttpReadLimits {
            max_header_bytes: default.max_header_bytes,
            max_body_bytes: MAX_OBJECT_UPLOAD_BODY_BYTES,
            max_chunked_wire_bytes: MAX_OBJECT_UPLOAD_CHUNKED_WIRE_BYTES,
        }
    } else {
        default
    }
}

fn finalize_forwarded_request(
    mut raw: Vec<u8>,
    header_end: usize,
    strip_expect: bool,
    rewritten_path: Option<&str>,
    rewritten_body: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let original_body = raw.split_off(header_end);
    // Re-parse with httparse so we iterate over validated header structs.
    let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut req = httparse::Request::new(&mut headers_buf);
    let _ = req.parse(&raw).context("re-parse headers for forwarding")?;

    let method = req.method.unwrap_or("GET");
    let path = rewritten_path.unwrap_or_else(|| req.path.unwrap_or("/"));
    let version = req.version.unwrap_or(1);

    let mut rebuilt = format!("{method} {path} HTTP/1.{version}\r\n");

    for header in req.headers.iter() {
        let name = header.name;
        if name.eq_ignore_ascii_case("connection") {
            continue;
        }
        if strip_expect && name.eq_ignore_ascii_case("expect") {
            continue;
        }
        if rewritten_body.is_some()
            && (name.eq_ignore_ascii_case("content-length")
                || name.eq_ignore_ascii_case("transfer-encoding"))
        {
            continue;
        }
        let value = std::str::from_utf8(header.value).unwrap_or("");
        rebuilt.push_str(&format!("{name}: {value}\r\n"));
    }
    if let Some(body) = rewritten_body {
        rebuilt.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }

    // The proxy buffers exactly one request for routing, so force a single-request
    // connection contract upstream instead of reusing the client connection blindly.
    rebuilt.push_str("Connection: close\r\n\r\n");

    let mut forwarded = rebuilt.into_bytes();
    forwarded.extend_from_slice(rewritten_body.unwrap_or(&original_body));
    Ok(forwarded)
}

/// Read from the stream until httparse can fully parse the request headers.
/// Returns parsed metadata; `buf` contains all bytes read so far (headers +
/// any trailing body bytes that arrived in the same read).
async fn read_until_headers_parsed(
    stream: &mut TcpStream,
    buf: &mut Vec<u8>,
    max_header_bytes: usize,
) -> Result<ParsedHeaders> {
    loop {
        let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
        let mut req = httparse::Request::new(&mut headers_buf);
        match req.parse(buf) {
            Ok(httparse::Status::Complete(header_end)) => {
                let method = req.method.unwrap_or("GET").to_string();
                let path = req.path.unwrap_or("/").to_string();

                let mut content_length = None;
                let mut is_chunked = false;
                let mut expects_continue = false;

                for header in req.headers.iter() {
                    if header.name.eq_ignore_ascii_case("content-length") {
                        let val = std::str::from_utf8(header.value)
                            .context("invalid Content-Length encoding")?;
                        content_length = Some(
                            val.trim()
                                .parse::<usize>()
                                .with_context(|| format!("invalid Content-Length: {val}"))?,
                        );
                    } else if header.name.eq_ignore_ascii_case("transfer-encoding") {
                        let val = std::str::from_utf8(header.value).unwrap_or("");
                        is_chunked = val
                            .split(',')
                            .any(|part| part.trim().eq_ignore_ascii_case("chunked"));
                    } else if header.name.eq_ignore_ascii_case("expect") {
                        let val = std::str::from_utf8(header.value).unwrap_or("");
                        expects_continue = val
                            .split(',')
                            .any(|part| part.trim().eq_ignore_ascii_case("100-continue"));
                    }
                }

                // RFC 7230 §3.3.3: if both Transfer-Encoding and Content-Length
                // are present, Transfer-Encoding wins and Content-Length is ignored.
                if is_chunked {
                    content_length = None;
                }

                return Ok(ParsedHeaders {
                    header_end,
                    method,
                    path,
                    content_length,
                    is_chunked,
                    expects_continue,
                });
            }
            Ok(httparse::Status::Partial) => {
                if buf.len() >= max_header_bytes {
                    bail!("HTTP headers exceed {max_header_bytes} bytes");
                }
                read_more(stream, buf).await?;
            }
            Err(e) => bail!("HTTP parse error: {e}"),
        }
    }
}

async fn read_more(stream: &mut TcpStream, buf: &mut Vec<u8>) -> Result<()> {
    let mut chunk = [0u8; 8192];
    let n = stream.read(&mut chunk).await?;
    if n == 0 {
        bail!("unexpected EOF while reading HTTP request");
    }
    buf.extend_from_slice(&chunk[..n]);
    Ok(())
}

fn try_decode_chunked_body(buf: &[u8], max_body_bytes: usize) -> Result<Option<(usize, Vec<u8>)>> {
    let mut pos = 0usize;
    let mut decoded = Vec::new();

    loop {
        let Some(line_end_rel) = buf[pos..].windows(2).position(|window| window == b"\r\n") else {
            return Ok(None);
        };
        let line_end = pos + line_end_rel;
        let size_line = std::str::from_utf8(&buf[pos..line_end]).context("invalid chunk header")?;
        let size_text = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_text, 16)
            .with_context(|| format!("invalid chunk size: {size_text}"))?;
        pos = line_end + 2;

        if size == 0 {
            if buf.len() < pos + 2 {
                return Ok(None);
            }
            if &buf[pos..pos + 2] == b"\r\n" {
                return Ok(Some((pos + 2, decoded)));
            }
            let Some(trailer_end_rel) = buf[pos..]
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
            else {
                return Ok(None);
            };
            return Ok(Some((pos + trailer_end_rel + 4, decoded)));
        }

        if buf.len() < pos + size + 2 {
            return Ok(None);
        }
        decoded.extend_from_slice(&buf[pos..pos + size]);
        pos += size;

        if &buf[pos..pos + 2] != b"\r\n" {
            return Err(anyhow!("invalid chunk terminator"));
        }
        pos += 2;

        if decoded.len() > max_body_bytes {
            bail!("HTTP chunked body exceeds {max_body_bytes} bytes");
        }
    }
}

fn request_requires_json_transform(path: &str, body: &[u8], plugin_manager_present: bool) -> bool {
    let path_only = path.split('?').next().unwrap_or(path);
    if body.is_empty() {
        return false;
    }
    if path_only == "/v1/responses" {
        return true;
    }
    if path_only != "/v1/chat/completions" {
        return false;
    }

    let body_text = match std::str::from_utf8(body) {
        Ok(text) => text,
        Err(_) => return false,
    };

    body_text.contains("\"max_completion_tokens\"")
        || body_text.contains("\"max_output_tokens\"")
        || body_text_contains_chat_reasoning_template_options(body_text)
        || (plugin_manager_present
            && (body_text.contains("mesh://blob/")
                || body_text.contains("\"blob_token\"")
                || body_text.contains("\"mesh_token\"")
                || body_text.contains("\"input_audio\"")
                || body_text.contains("\"input_image\"")))
}

fn body_text_contains_chat_reasoning_template_options(body_text: &str) -> bool {
    body_text.contains("\"reasoning\"")
        || body_text.contains("\"reasoning_effort\"")
        || body_text.contains("\"thinking_budget\"")
        || body_text.contains("\"chat_template_kwargs\"")
        || openai_frontend::THINKING_BOOLEAN_ALIASES
            .iter()
            .any(|field| body_text.contains(&format!("\"{field}\"")))
}

pub(super) fn parse_json_body_from_http_request(raw: &[u8]) -> Option<serde_json::Value> {
    let header_end = raw.windows(4).position(|window| window == b"\r\n\r\n")? + 4;
    serde_json::from_slice(&raw[header_end..]).ok()
}

/// Inject `"mesh_hooks": true/false` into the JSON body of an HTTP request.
///
/// Inserts the field right after the opening `{` in the body, then rebuilds
/// the Content-Length header to match.
pub fn inject_mesh_hooks_flag(raw: &mut Vec<u8>, enabled: bool) {
    let Some(header_end) = raw.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4) else {
        return;
    };
    let body = &raw[header_end..];
    let Some(brace) = body.iter().position(|&b| b == b'{') else {
        return;
    };

    // Build new body with mesh_hooks injected after opening brace
    let fragment = if enabled {
        &b"\"mesh_hooks\":true,"[..]
    } else {
        &b"\"mesh_hooks\":false,"[..]
    };
    let mut new_body = Vec::with_capacity(body.len() + fragment.len());
    new_body.extend_from_slice(&body[..brace + 1]);
    new_body.extend_from_slice(fragment);
    new_body.extend_from_slice(&body[brace + 1..]);

    // Rebuild headers with correct Content-Length
    let headers = std::str::from_utf8(&raw[..header_end - 4]).unwrap_or("");
    let mut rebuilt = String::new();
    for line in headers.split("\r\n") {
        if line.to_ascii_lowercase().starts_with("content-length:") {
            rebuilt.push_str(&format!("Content-Length: {}", new_body.len()));
        } else {
            rebuilt.push_str(line);
        }
        rebuilt.push_str("\r\n");
    }
    rebuilt.push_str("\r\n");

    let mut result = rebuilt.into_bytes();
    result.extend_from_slice(&new_body);
    *raw = result;
}

/// Rewrite the JSON body `model` field and rebuild Content-Length.
pub fn rewrite_model_field(request: &mut BufferedHttpRequest, model: &str) {
    let Some(header_end) = request
        .raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4)
    else {
        return;
    };

    let Ok(mut body) = serde_json::from_slice::<serde_json::Value>(&request.raw[header_end..])
    else {
        return;
    };
    let Some(object) = body.as_object_mut() else {
        return;
    };

    object.insert(
        "model".to_string(),
        serde_json::Value::String(model.to_string()),
    );
    let Ok(new_body) = serde_json::to_vec(&body) else {
        return;
    };

    let headers = std::str::from_utf8(&request.raw[..header_end - 4]).unwrap_or("");
    let mut rebuilt = String::new();
    for line in headers.split("\r\n") {
        if line.to_ascii_lowercase().starts_with("content-length:") {
            rebuilt.push_str(&format!("Content-Length: {}", new_body.len()));
        } else {
            rebuilt.push_str(line);
        }
        rebuilt.push_str("\r\n");
    }
    rebuilt.push_str("\r\n");

    let mut raw = rebuilt.into_bytes();
    raw.extend_from_slice(&new_body);

    request.raw = raw;
    request.body_len_bytes = new_body.len();
    request.body_bytes = Some(new_body);
    request.body_json = Some(body);
    request.body_json_attempted = true;
    request.model_name = Some(model.to_string());
}

pub fn is_models_list_request(method: &str, path: &str) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    method == "GET" && (path == "/v1/models" || path == "/models")
}

pub fn is_drop_request(method: &str, path: &str) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    method == "POST" && path == "/mesh/drop"
}

pub fn pipeline_request_supported(path: &str, body: &serde_json::Value) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    path == "/v1/chat/completions"
        && body
            .get("messages")
            .map(|messages| messages.is_array())
            .unwrap_or(false)
}

pub fn rewrite_public_model_alias(
    request: &mut BufferedHttpRequest,
    models: &[String],
    descriptors: &[mesh::ServedModelDescriptor],
) {
    let Some(requested) = request.model_name.as_deref() else {
        return;
    };
    if requested == "auto" || models.iter().any(|model| model == requested) {
        return;
    }
    let Some(internal) = internal_model_for_public_id(requested, models, descriptors) else {
        return;
    };
    rewrite_model_field(request, &internal);
}

fn internal_model_for_public_id(
    requested: &str,
    models: &[String],
    descriptors: &[mesh::ServedModelDescriptor],
) -> Option<String> {
    let (requested_base, requested_profile) =
        crate::network::openai::ingress::parse_model_with_profile(requested);

    models.iter().find_map(|model| {
        let (model_base, model_profile) =
            crate::network::openai::ingress::parse_model_with_profile(model);
        let descriptor = descriptor_for_model(descriptors, model_base);
        let public_id = public_model_id(model_base, descriptor, model_profile);
        if public_id == requested {
            return Some(model.clone());
        }
        let (public_base, _public_profile) =
            crate::network::openai::ingress::parse_model_with_profile(&public_id);
        if public_base == requested_base && requested_profile.is_empty() {
            return Some(model.clone());
        }
        None
    })
}

pub(super) fn public_model_id(
    model_name: &str,
    descriptor: Option<&mesh::ServedModelDescriptor>,
    profile: &str,
) -> String {
    // A descriptor with an `artifact` field has enough information to
    // produce a public ID that round-trips to the same model. Without
    // it, the HuggingFace path collapses to just the repo name and
    // silently drops the quant-tag suffix the resolver needs (PR #566
    // review feedback — "some IDs in /v1/models dropped quant
    // suffixes"). Only use the descriptor-derived id when it can be
    // lossless; otherwise prefer the on-disk file (authoritative for
    // local models), and finally the internal model_name (which
    // always carries the quant suffix our resolver knows how to
    // route).
    let base_id = if let Some(descriptor) = descriptor
        && descriptor_can_produce_lossless_id(&descriptor.identity)
        && let Some(id) = public_model_id_from_identity(&descriptor.identity)
    {
        id
    } else if let Some(id) = public_model_id_from_local_path(model_name) {
        id
    } else {
        model_name.to_string()
    };

    // Append profile suffix for non-default profiles
    if profile.is_empty() {
        base_id
    } else {
        format!("{}#{}", base_id, profile)
    }
}

/// A descriptor identity carries enough information for
/// `public_model_id_from_identity` to produce an ID that round-trips
/// to the same model. For HuggingFace that means the `artifact` field
/// (the GGUF file name) is present so the quant selector can be
/// derived. Catalog identities always carry a `canonical_ref` with the
/// selector baked in.
fn descriptor_can_produce_lossless_id(identity: &mesh::ServedModelIdentity) -> bool {
    match identity.source_kind {
        mesh::ModelSourceKind::HuggingFace => identity.artifact.is_some(),
        mesh::ModelSourceKind::Catalog => identity.canonical_ref.is_some(),
        mesh::ModelSourceKind::LocalGguf
        | mesh::ModelSourceKind::DirectUrl
        | mesh::ModelSourceKind::Unknown => false,
    }
}

fn public_model_id_from_identity(identity: &mesh::ServedModelIdentity) -> Option<String> {
    match identity.source_kind {
        mesh::ModelSourceKind::HuggingFace => identity
            .repository
            .as_deref()
            .and_then(|repo| public_huggingface_model_ref(repo, identity.artifact.as_deref()))
            .or_else(|| {
                identity
                    .canonical_ref
                    .as_deref()
                    .and_then(|model_ref| model_ref::ModelRef::parse(model_ref).ok())
                    .map(|model_ref| model_ref.display_id())
            }),
        mesh::ModelSourceKind::Catalog => identity
            .canonical_ref
            .as_deref()
            .and_then(|model_ref| model_ref::ModelRef::parse(model_ref).ok())
            .map(|model_ref| model_ref.display_id()),
        mesh::ModelSourceKind::LocalGguf
        | mesh::ModelSourceKind::DirectUrl
        | mesh::ModelSourceKind::Unknown => None,
    }
}

fn public_model_id_from_local_path(model_name: &str) -> Option<String> {
    let path = crate::models::find_model_path(model_name);
    if !path.is_file() {
        return None;
    }
    if path.extension().and_then(|extension| extension.to_str()) != Some("gguf") {
        return None;
    }
    Some(crate::models::model_ref_for_path(&path))
}

fn public_huggingface_model_ref(repo: &str, artifact: Option<&str>) -> Option<String> {
    // `artifact` can be either a GGUF filename (e.g. `Falcon-Q4_K_M.gguf`)
    // or an already-extracted quant selector (e.g. `Q4_K_M` or
    // `qwen2.5-3b-instruct-q4_k_m`, when the descriptor was built from
    // a parsed `ModelRef::selector`). Handle both — if the artifact
    // looks like a quant selector use it directly; otherwise try to
    // pull a selector out of the filename.
    let selector = artifact.and_then(|a| {
        model_ref::quant_selector_from_gguf_file(a)
            .or_else(|| (!a.is_empty() && !a.ends_with(".gguf")).then(|| a.to_string()))
    });
    Some(model_ref::format_model_ref(repo, None, selector.as_deref()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    fn catalog_model_ref_descriptor(model_name: &str) -> mesh::ServedModelDescriptor {
        mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model_name.to_string(),
                source_kind: mesh::ModelSourceKind::Catalog,
                canonical_ref: Some("tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M".to_string()),
                ..Default::default()
            },
            ..Default::default()
        }
    }
    async fn read_request_from_parts_with_limits(
        parts: Vec<Vec<u8>>,
        limits: HttpReadLimits,
    ) -> BufferedHttpRequest {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_http_request_with_limits(&mut stream, limits, None)
                .await
                .unwrap()
        });

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.unwrap();
            for part in parts {
                stream.write_all(&part).await.unwrap();
            }
        });

        client.await.unwrap();
        server.await.unwrap()
    }

    async fn read_request_from_parts(parts: Vec<Vec<u8>>) -> BufferedHttpRequest {
        read_request_from_parts_with_limits(parts, HTTP_READ_LIMITS).await
    }
    fn build_chunked_request(body: &[u8], chunks: &[usize]) -> Vec<u8> {
        let mut out = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        let mut pos = 0usize;
        for &chunk_len in chunks {
            let end = pos + chunk_len;
            out.extend_from_slice(format!("{chunk_len:x}\r\n").as_bytes());
            out.extend_from_slice(&body[pos..end]);
            out.extend_from_slice(b"\r\n");
            pos = end;
        }
        out.extend_from_slice(b"0\r\n\r\n");
        out
    }

    fn build_chunked_request_one_byte_chunks(body: &[u8], extension_len: usize) -> Vec<u8> {
        let mut out = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        let extension = "x".repeat(extension_len);
        for byte in body {
            out.extend_from_slice(b"1");
            if !extension.is_empty() {
                out.extend_from_slice(b";");
                out.extend_from_slice(extension.as_bytes());
            }
            out.extend_from_slice(b"\r\n");
            out.push(*byte);
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(b"0\r\n\r\n");
        out
    }
    #[test]
    fn public_model_alias_rewrites_request_to_internal_model_name() {
        let models = vec!["Falcon-H1-1.5B-Instruct-Q4_K_M".to_string()];
        let descriptors = vec![catalog_model_ref_descriptor(&models[0])];
        let body = serde_json::json!({
            "model": "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let body_bytes = serde_json::to_vec(&body).unwrap();
        let mut raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            body_bytes.len()
        )
        .into_bytes();
        raw.extend_from_slice(&body_bytes);
        let mut request = BufferedHttpRequest {
            raw,
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            client_path: "/v1/chat/completions".to_string(),
            body_json: Some(body),
            body_json_attempted: true,
            body_bytes: Some(body_bytes),
            body_len_bytes: 0,
            completion_tokens: None,
            model_name: Some("tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M".to_string()),
            stream: None,
            request_object_request_ids: Vec::new(),
            response_adapter: ResponseAdapter::None,
        };

        rewrite_public_model_alias(&mut request, &models, &descriptors);

        assert_eq!(request.model_name.as_deref(), Some(models[0].as_str()));
        assert_eq!(request.body_json.as_ref().unwrap()["model"], models[0]);
    }
    #[test]
    fn test_pipeline_request_supported_chat_completions() {
        let body = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(pipeline_request_supported(
            "/v1/chat/completions?stream=1",
            &body
        ));
    }

    #[test]
    fn test_pipeline_request_supported_rejects_other_endpoint() {
        let body = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(!pipeline_request_supported("/v1/responses", &body));
    }
    #[test]
    fn test_pipeline_request_supported_rejects_missing_messages() {
        let body = serde_json::json!({"input":"hi"});
        assert!(!pipeline_request_supported("/v1/chat/completions", &body));
    }
    #[tokio::test]
    async fn test_read_http_request_fragmented_post_body() {
        let body =
            br#"{"model":"qwen","user":"alice","messages":[{"role":"user","content":"hi"}]}"#;
        let headers = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );

        let request = read_request_from_parts(vec![
            headers.as_bytes()[..40].to_vec(),
            headers.as_bytes()[40..].to_vec(),
            body[..12].to_vec(),
            body[12..].to_vec(),
        ])
        .await;

        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.model_name.as_deref(), Some("qwen"));
        assert_eq!(
            request.response_adapter,
            ResponseAdapter::OpenAiChatCompletionsJson
        );

        assert!(request.body_json.is_none());
    }

    #[tokio::test]
    async fn chat_reasoning_effort_none_is_canonicalized_before_forwarding() {
        let body = serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "none"
        })
        .to_string();
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

        let request = read_request_from_parts(vec![raw.into_bytes()]).await;
        let forwarded = parse_json_body_from_http_request(&request.raw).unwrap();

        assert_eq!(
            forwarded["chat_template_kwargs"]["enable_thinking"],
            serde_json::json!(false)
        );
        assert_eq!(request.body_json, Some(forwarded));
    }

    #[tokio::test]
    async fn chat_existing_template_kwargs_survive_forwarding_rewrite() {
        let body = serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 32,
            "reasoning_effort": "low",
            "chat_template_kwargs": {
                "enable_thinking": false,
                "custom": "kept"
            }
        })
        .to_string();
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

        let request = read_request_from_parts(vec![raw.into_bytes()]).await;
        let forwarded = parse_json_body_from_http_request(&request.raw).unwrap();

        assert_eq!(forwarded["max_tokens"], serde_json::json!(32));
        assert!(forwarded.get("max_completion_tokens").is_none());
        assert_eq!(
            forwarded["chat_template_kwargs"],
            serde_json::json!({"enable_thinking": false, "custom": "kept"})
        );
    }

    #[tokio::test]
    async fn chat_reasoning_enabled_false_wins_over_nested_effort_before_forwarding() {
        let body = serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"enabled": false, "effort": "low"}
        })
        .to_string();
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

        let request = read_request_from_parts(vec![raw.into_bytes()]).await;
        let forwarded = parse_json_body_from_http_request(&request.raw).unwrap();

        assert_eq!(
            forwarded["chat_template_kwargs"]["enable_thinking"],
            serde_json::json!(false)
        );
        assert_eq!(request.body_json, Some(forwarded));
    }

    #[tokio::test]
    async fn test_read_http_request_preserves_client_path_for_responses_capture() {
        let body = br#"{"model":"qwen","stream":true,"input":"hello"}"#;
        let request = format!(
            "POST /v1/responses?foo=1 HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            std::str::from_utf8(body).unwrap()
        );

        let request = read_request_from_parts(vec![request.into_bytes()]).await;

        assert_eq!(request.path, "/v1/chat/completions?foo=1");
        assert_eq!(request.client_path, "/v1/responses?foo=1");
    }
    #[tokio::test]
    async fn test_read_http_request_large_body_over_32k() {
        let large = "x".repeat(40_000);
        let body = serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": large}],
        })
        .to_string();
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

        let mut request = read_request_from_parts(vec![request.into_bytes()]).await;

        assert_eq!(request.model_name.as_deref(), Some("qwen"));
        request.ensure_body_json();
        let body_json = request.body_json.unwrap();
        let content = body_json["messages"][0]["content"].as_str().unwrap();
        assert_eq!(content.len(), 40_000);
    }

    #[tokio::test]
    async fn test_read_http_request_chunked_body() {
        let body = br#"{"model":"auto","session_id":"sess-42","messages":[{"role":"user","content":"hello"}]}"#;
        let request = build_chunked_request(body, &[18, 17, body.len() - 35]);

        let request = read_request_from_parts(vec![request]).await;

        assert_eq!(request.model_name.as_deref(), Some("auto"));

        assert!(request.body_json.is_none());
    }

    #[tokio::test]
    async fn test_read_http_request_chunked_body_allows_wire_overhead() {
        let limits = HttpReadLimits {
            max_header_bytes: MAX_HEADER_BYTES,
            max_body_bytes: 256,
            max_chunked_wire_bytes: 4 * 1024,
        };
        let large = "x".repeat(48);
        let body = serde_json::json!({
            "model": "qwen",
            "messages": [{"role": "user", "content": large}],
        })
        .to_string();
        let request = build_chunked_request_one_byte_chunks(body.as_bytes(), 16);

        let mut request = read_request_from_parts_with_limits(vec![request], limits).await;

        assert_eq!(request.model_name.as_deref(), Some("qwen"));
        assert!(request.raw.len() > limits.max_body_bytes);
        request.ensure_body_json();
        let body_json = request.body_json.unwrap();
        let content = body_json["messages"][0]["content"].as_str().unwrap();
        assert_eq!(content.len(), 48);
    }

    #[tokio::test]
    async fn test_read_http_request_allows_large_object_upload_body() {
        let body = vec![b'x'; MAX_BODY_BYTES + 1];
        let headers = format!(
            "POST /api/objects HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes();

        let request = read_request_from_parts(vec![headers, body.clone()]).await;

        assert_eq!(request.path, "/api/objects");
        assert!(request.raw.ends_with(&body));
        assert!(request.body_json.is_none());
        assert!(request.request_object_request_ids.is_empty());
    }

    #[tokio::test]
    async fn test_read_http_request_expect_100_continue() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let body = br#"{"model":"qwen","user":"bob","messages":[]}"#.to_vec();
        let headers = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nExpect: 100-continue\r\n\r\n",
            body.len()
        );

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_http_request(&mut stream).await.unwrap()
        });

        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.unwrap();
            stream.write_all(headers.as_bytes()).await.unwrap();

            let mut interim = [0u8; 64];
            let n = stream.read(&mut interim).await.unwrap();
            assert_eq!(
                std::str::from_utf8(&interim[..n]).unwrap(),
                "HTTP/1.1 100 Continue\r\n\r\n"
            );

            stream.write_all(&body).await.unwrap();
        });

        client.await.unwrap();
        let request = server.await.unwrap();
        assert_eq!(request.model_name.as_deref(), Some("qwen"));

        let raw = String::from_utf8(request.raw).unwrap();
        assert!(!raw.contains("Expect: 100-continue"));
        assert!(raw.contains("Connection: close"));
    }
    #[tokio::test]
    async fn test_read_http_request_truncates_pipelined_follow_up_bytes() {
        let request = read_request_from_parts(vec![
            b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\nGET /mesh/drop HTTP/1.1\r\nHost: localhost\r\n\r\n"
                .to_vec(),
        ])
        .await;

        let raw = String::from_utf8(request.raw).unwrap();
        assert!(raw.starts_with("GET /v1/models HTTP/1.1\r\n"));
        assert!(!raw.contains("/mesh/drop"));
        assert!(raw.contains("Connection: close\r\n\r\n"));
    }

    /// `probe_http_response_local` uses a much longer timeout (10 min)
    /// than `probe_http_response` (5 min), because local prefill can
    /// legitimately take minutes under load.
    ///
    /// This test sends a response after a 2s delay and verifies that
    /// `probe_http_response_local` waits for it (well within its 10-min
    #[test]
    fn test_inject_mesh_hooks_enabled() {
        let mut raw = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 25\r\n\r\n{\"model\":\"auto\",\"n\":1}".to_vec();
        inject_mesh_hooks_flag(&mut raw, true);
        let body_start = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        let body = std::str::from_utf8(&raw[body_start..]).unwrap();
        assert!(body.starts_with("{\"mesh_hooks\":true,"), "body: {body}");
        // Content-Length must match actual body length
        let cl_line = std::str::from_utf8(&raw[..body_start])
            .unwrap()
            .lines()
            .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
            .unwrap();
        let declared: usize = cl_line.split(':').nth(1).unwrap().trim().parse().unwrap();
        assert_eq!(declared, raw.len() - body_start);
    }

    #[test]
    fn test_inject_mesh_hooks_disabled() {
        let mut raw = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 25\r\n\r\n{\"model\":\"auto\",\"n\":1}".to_vec();
        inject_mesh_hooks_flag(&mut raw, false);
        let body_start = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        let body = std::str::from_utf8(&raw[body_start..]).unwrap();
        assert!(body.starts_with("{\"mesh_hooks\":false,"), "body: {body}");
    }

    #[test]
    fn test_inject_mesh_hooks_no_body() {
        let mut raw = b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec();
        let before = raw.clone();
        inject_mesh_hooks_flag(&mut raw, true);
        assert_eq!(raw, before, "GET with no body should be unchanged");
    }

    #[test]
    fn test_rewrite_model_field_updates_body_and_content_length() {
        let mut request = BufferedHttpRequest {
            raw: b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 45\r\n\r\n{\"model\":\"auto\",\"messages\":[],\"mesh_hooks\":true}".to_vec(),
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            client_path: "/v1/chat/completions".to_string(),
            body_json: None,
            body_json_attempted: false,
            body_bytes: None,
            body_len_bytes: 45,
            completion_tokens: None,
            model_name: Some("auto".to_string()),
            stream: None,
            request_object_request_ids: Vec::new(),
            response_adapter: ResponseAdapter::None,
        };

        rewrite_model_field(&mut request, "SmolLM2-135M-Instruct-Q8_0");

        let body_start = request
            .raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .unwrap()
            + 4;
        let body: serde_json::Value = serde_json::from_slice(&request.raw[body_start..]).unwrap();
        assert_eq!(body["model"], "SmolLM2-135M-Instruct-Q8_0");
        assert_eq!(body["mesh_hooks"], true);
        assert_eq!(
            request.model_name.as_deref(),
            Some("SmolLM2-135M-Instruct-Q8_0")
        );

        let cl_line = std::str::from_utf8(&request.raw[..body_start])
            .unwrap()
            .lines()
            .find(|line| line.to_ascii_lowercase().starts_with("content-length:"))
            .unwrap();
        let declared: usize = cl_line.split(':').nth(1).unwrap().trim().parse().unwrap();
        assert_eq!(declared, request.raw.len() - body_start);
        assert_eq!(declared, request.body_len_bytes);
    }

    #[test]
    fn public_model_id_with_named_profile() {
        let result = public_model_id("Qwen3-8B", None, "low-ctx");
        assert_eq!(result, "Qwen3-8B#low-ctx");
    }

    #[test]
    fn public_model_id_without_profile() {
        let result = public_model_id("Qwen3-8B", None, "");
        assert_eq!(result, "Qwen3-8B");
    }

    #[test]
    fn public_model_id_with_empty_profile() {
        let result = public_model_id("Qwen3-8B", None, "");
        assert_eq!(result, "Qwen3-8B");
    }

    #[test]
    fn public_model_id_with_huggingface_ref_and_profile() {
        let result = public_model_id("org/repo:Q4_K_M", None, "high-ctx");
        assert_eq!(result, "org/repo:Q4_K_M#high-ctx");
    }
}
