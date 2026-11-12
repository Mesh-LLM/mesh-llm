//! Inspect a complete, bounded response before any of its image bytes are released.
//!
//! Only the text/image representation supported by the first multimodal output
//! slice is admitted: literal inline Markdown with a PNG, JPEG, or WebP data URI.
//! Image decoding belongs on the caller's bounded blocking worker, not an async
//! executor thread. No request or response metadata is searched for arbitrary
//! strings that happen to resemble an image.

use anyhow::{Context, Result, bail, ensure};
use base64::Engine;
use image::{ImageFormat, ImageReader, Limits};
use pulldown_cmark::{Event, LinkType, Parser, Tag};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;
use std::ops::Range;

pub(crate) const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const MAX_IMAGE_URI_CHARS: usize = 4 * 1024 * 1024;
const MAX_IMAGE_REFERENCES: usize = 8;
// Responses streams may repeat each image in done/item/completed snapshots.
const MAX_IMAGE_INSPECTIONS: usize = 64;
const MAX_IMAGE_DIMENSION: u32 = 8192;
const MAX_IMAGE_PIXELS: u64 = 16 * 1024 * 1024;
const MAX_DECODE_BYTES: u64 = MAX_IMAGE_PIXELS * 4;

pub(crate) struct VerifiedImage {
    pub data_uri: String,
    pub mime_type: &'static str,
    pub bytes: Vec<u8>,
}

// Accidental diagnostics must not print pixels or their base64 representation.
impl std::fmt::Debug for VerifiedImage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VerifiedImage")
            .field("mime_type", &self.mime_type)
            .field("encoded_bytes", &self.bytes.len())
            .finish()
    }
}

/// Extract and fully decode every supported image from the buffered response.
///
/// Repeated final-text snapshots in SSE are inspected too, but byte-identical
/// images are returned once. Every returned byte buffer is still compressed;
/// decoded pixels are discarded immediately after validating the raster.
pub(crate) fn images_from_response(content_type: &str, body: &[u8]) -> Result<Vec<VerifiedImage>> {
    ensure!(
        body.len() <= MAX_RESPONSE_BYTES,
        "image response exceeds byte limit"
    );
    let media_type = content_type.split(';').next().unwrap_or_default().trim();
    let texts = if media_type.eq_ignore_ascii_case("text/event-stream") {
        texts_from_sse(body)?
    } else if media_type.eq_ignore_ascii_case("text/plain") {
        vec![
            std::str::from_utf8(body)
                .context("invalid response text")?
                .to_owned(),
        ]
    } else if media_type.is_empty() || media_type.eq_ignore_ascii_case("application/json") {
        texts_from_json(&super::json_keys::parse(body).context("invalid response JSON")?)?
    } else {
        bail!("unsupported image response content type");
    };
    let mut images = Vec::new();
    let mut seen = BTreeSet::new();
    let mut seen_uris = BTreeSet::new();
    let mut inspections = 0usize;
    for text in texts {
        for uri in image_uris(&text)? {
            inspections += 1;
            ensure!(
                inspections <= MAX_IMAGE_INSPECTIONS,
                "too many image snapshots"
            );
            if !seen_uris.insert(uri.clone()) {
                continue;
            }
            let image = decode_image(&uri)?;
            let digest: [u8; 32] = Sha256::digest(&image.bytes).into();
            if seen.insert(digest) {
                ensure!(
                    images.len() < MAX_IMAGE_REFERENCES,
                    "too many generated images"
                );
                images.push(image);
            }
        }
    }
    Ok(images)
}

fn texts_from_json(value: &Value) -> Result<Vec<String>> {
    ensure!(value.is_object(), "response must be a JSON object");
    if let Some(error) = value.get("error").filter(|error| !error.is_null()) {
        ensure!(
            value.get("choices").is_none()
                && value.get("output").is_none()
                && value.get("output_text").is_none(),
            "error mixed with model output"
        );
        let message = error
            .as_str()
            .or_else(|| error.get("message").and_then(Value::as_str))
            .context("invalid response error message")?;
        ensure!(
            image_uris(message)?.is_empty(),
            "image output in error response"
        );
        return Ok(Vec::new());
    }
    let mut texts = Vec::new();
    if let Some(choices) = value.get("choices") {
        ensure!(
            value.get("output").is_none()
                && value.get("output_text").is_none()
                && value.get("object").and_then(Value::as_str) != Some("response"),
            "conflicting response envelopes"
        );
        for choice in choices.as_array().context("invalid response choices")? {
            ensure!(
                choice.get("delta").is_none()
                    && (choice.get("message").is_some() || choice.get("text").is_some()),
                "unsupported nonstreaming response choice"
            );
            if let Some(message) = choice.get("message") {
                collect_message(message, &mut texts)?;
            }
            if let Some(text) = choice.get("text") {
                texts.push(text.as_str().context("invalid completion text")?.to_owned());
            }
        }
    } else if value.get("output").is_some()
        || value.get("object").and_then(Value::as_str) == Some("response")
    {
        collect_response(value, &mut texts)?;
    } else {
        bail!("unrecognized image response envelope");
    }
    Ok(texts)
}

fn collect_response(value: &Value, texts: &mut Vec<String>) -> Result<()> {
    if let Some(output) = value.get("output") {
        for item in output.as_array().context("invalid response output")? {
            collect_output_item(item, texts)?;
        }
    }
    if let Some(text) = value.get("output_text") {
        texts.push(
            text.as_str()
                .context("invalid response output text")?
                .to_owned(),
        );
    }
    Ok(())
}

fn collect_output_item(item: &Value, texts: &mut Vec<String>) -> Result<()> {
    match item.get("type").and_then(Value::as_str) {
        Some("message") => collect_message(item, texts),
        Some("reasoning") => {
            if let Some(summary) = item.get("summary") {
                collect_parts(summary, texts)?;
            }
            if let Some(content) = item.get("content") {
                collect_content(content, texts)?;
            }
            Ok(())
        }
        Some("image_generation_call" | "output_image" | "image") => {
            bail!("structured image output is not supported by image screening")
        }
        _ => Ok(()),
    }
}

fn collect_message(message: &Value, texts: &mut Vec<String>) -> Result<()> {
    ensure!(
        ["image", "images", "image_url"]
            .iter()
            .all(|key| message.get(*key).is_none()),
        "structured image output is not supported by image screening"
    );
    if let Some(content) = message.get("content") {
        collect_content(content, texts)?;
    }
    for field in ["reasoning_content", "reasoning", "refusal"] {
        if let Some(value) = message.get(field).filter(|v| !v.is_null()) {
            texts.push(
                value
                    .as_str()
                    .context("invalid message text field")?
                    .to_owned(),
            );
        }
    }
    Ok(())
}

fn collect_content(content: &Value, texts: &mut Vec<String>) -> Result<()> {
    match content {
        Value::String(text) => texts.push(text.clone()),
        Value::Array(_) => collect_parts(content, texts)?,
        Value::Null => {}
        _ => bail!("unsupported message content"),
    }
    Ok(())
}

fn collect_parts(content: &Value, texts: &mut Vec<String>) -> Result<()> {
    let mut text = String::new();
    for part in content.as_array().context("invalid text content parts")? {
        match part.get("type").and_then(Value::as_str) {
            Some("text" | "output_text" | "reasoning_text" | "summary_text") => {
                text.push_str(
                    part.get("text")
                        .and_then(Value::as_str)
                        .context("missing content text")?,
                );
            }
            Some("refusal") => {
                text.push_str(
                    part.get("refusal")
                        .and_then(Value::as_str)
                        .context("missing refusal text")?,
                );
            }
            Some("image" | "image_url" | "input_image" | "output_image") => {
                bail!("structured image output is not supported by image screening");
            }
            _ => {}
        }
    }
    if !text.is_empty() {
        texts.push(text);
    }
    Ok(())
}

#[derive(Default)]
struct StreamText {
    deltas: BTreeMap<String, String>,
    snapshots: Vec<String>,
    saw_data: bool,
    done: bool,
    response_completed: bool,
    rendered_content: String,
    rendered_reasoning: String,
}

fn texts_from_sse(body: &[u8]) -> Result<Vec<String>> {
    let text = std::str::from_utf8(body)
        .context("invalid response SSE encoding")?
        .replace("\r\n", "\n");
    ensure!(!text.contains('\r'), "unsupported SSE line ending");
    ensure!(
        text.ends_with("\n\n"),
        "incomplete image response SSE frame"
    );
    let mut stream = StreamText::default();
    for frame in text.split("\n\n") {
        process_sse_frame(frame, &mut stream)?;
    }
    ensure!(
        stream.saw_data && (stream.done || stream.response_completed),
        "incomplete image response SSE"
    );
    stream.snapshots.extend(stream.deltas.into_values());
    stream.snapshots.push(stream.rendered_content);
    stream.snapshots.push(stream.rendered_reasoning);
    Ok(stream.snapshots)
}

fn process_sse_frame(frame: &str, stream: &mut StreamText) -> Result<()> {
    let mut data = Vec::new();
    let mut event_name = None;
    for line in frame.lines() {
        if let Some(value) = line.strip_prefix("data:") {
            data.push(value.strip_prefix(' ').unwrap_or(value));
        } else if let Some(value) = line.strip_prefix("event:") {
            event_name = Some(value.trim());
        } else {
            ensure!(
                line.is_empty()
                    || line.starts_with(':')
                    || line.starts_with("id:")
                    || line.starts_with("retry:"),
                "unsupported SSE field"
            );
        }
    }
    if data.is_empty() {
        return Ok(());
    }
    ensure!(!stream.done, "SSE data after terminal marker");
    let data = data.join("\n");
    stream.saw_data = true;
    if data.trim() == "[DONE]" {
        stream.done = true;
        return Ok(());
    }
    ensure!(
        !stream.response_completed,
        "SSE data after completed response"
    );
    let value = super::json_keys::parse(data.as_bytes()).context("invalid SSE JSON")?;
    ensure!(
        value.get("error").is_none_or(Value::is_null),
        "image response did not complete successfully"
    );
    if let Some(name) = event_name.filter(|name| name.starts_with("response.")) {
        ensure!(
            value.get("type").and_then(Value::as_str) == Some(name),
            "conflicting SSE event type"
        );
    }
    if value.get("choices").is_some() {
        collect_chat_deltas(&value, stream)
    } else {
        collect_response_event(&value, stream)
    }
}

fn collect_chat_deltas(value: &Value, stream: &mut StreamText) -> Result<()> {
    for choice in value
        .get("choices")
        .and_then(Value::as_array)
        .context("invalid SSE choices")?
    {
        let index = choice.get("index").and_then(Value::as_u64).unwrap_or(0);
        if let Some(delta) = choice.get("delta") {
            ensure!(
                ["image", "images", "image_url"]
                    .iter()
                    .all(|key| delta.get(*key).is_none()),
                "structured image output is not supported by image screening"
            );
            for field in ["content", "reasoning_content", "reasoning", "refusal"] {
                if let Some(text) = delta.get(field).filter(|v| !v.is_null()) {
                    let key = format!("chat:{index}:{field}");
                    if text.is_array() {
                        let mut parts = Vec::new();
                        collect_parts(text, &mut parts)?;
                        for part in parts {
                            stream
                                .deltas
                                .entry(key.clone())
                                .or_default()
                                .push_str(&part);
                            append_rendered(stream, field == "content", &part);
                        }
                    } else {
                        let text = text.as_str().context("invalid SSE text delta")?;
                        stream.deltas.entry(key).or_default().push_str(text);
                        append_rendered(stream, field == "content", text);
                    }
                }
            }
        }
        if let Some(text) = choice.get("text") {
            let text = text.as_str().context("invalid completion delta")?;
            stream
                .deltas
                .entry(format!("completion:{index}"))
                .or_default()
                .push_str(text);
            append_rendered(stream, true, text);
        }
        if let Some(message) = choice.get("message") {
            collect_message(message, &mut stream.snapshots)?;
        }
    }
    Ok(())
}

fn collect_response_event(value: &Value, stream: &mut StreamText) -> Result<()> {
    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .context("unrecognized image SSE event")?;
    match kind {
        "response.output_text.delta"
        | "response.reasoning_summary_text.delta"
        | "response.reasoning_text.delta"
        | "response.refusal.delta" => {
            let output = value
                .get("output_index")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let content = value
                .get("content_index")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let item = value
                .get("item_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let key = format!("{kind}:{output}:{content}:{item}");
            let delta = value
                .get("delta")
                .and_then(Value::as_str)
                .context("missing response text delta")?;
            stream.deltas.entry(key).or_default().push_str(delta);
            append_rendered(
                stream,
                matches!(
                    kind,
                    "response.output_text.delta" | "response.refusal.delta"
                ),
                delta,
            );
        }
        "response.output_text.done" => {
            stream.snapshots.push(
                value
                    .get("text")
                    .and_then(Value::as_str)
                    .context("missing final output text")?
                    .to_owned(),
            );
        }
        "response.output_item.added" | "response.output_item.done" => {
            collect_output_item(
                value.get("item").context("missing response output item")?,
                &mut stream.snapshots,
            )?;
        }
        "response.content_part.added" | "response.content_part.done" => {
            collect_parts(
                &Value::Array(vec![
                    value
                        .get("part")
                        .context("missing response content part")?
                        .clone(),
                ]),
                &mut stream.snapshots,
            )?;
        }
        "response.completed" => {
            collect_response(
                value
                    .get("response")
                    .context("missing completed response")?,
                &mut stream.snapshots,
            )?;
            stream.response_completed = true;
        }
        "response.failed" | "response.incomplete" | "error" => {
            bail!("image response did not complete successfully")
        }
        kind if kind.starts_with("response.image_generation_call.") => {
            bail!("structured image output is not supported by image screening")
        }
        _ => {}
    }
    Ok(())
}

// The console appends deltas in arrival order, even if a backend changes the
// logical item indices. Validate that rendered view as well as each item.
fn append_rendered(stream: &mut StreamText, content: bool, text: &str) {
    if content {
        stream.rendered_content.push_str(text);
    } else {
        stream.rendered_reasoning.push_str(text);
    }
}

fn image_uris(text: &str) -> Result<Vec<String>> {
    let mut uris = Vec::new();
    let mut admitted_ranges = Vec::<Range<usize>>::new();
    for (event, range) in Parser::new(text).into_offset_iter() {
        match event {
            Event::Start(Tag::Image {
                link_type,
                dest_url,
                ..
            }) => {
                ensure!(
                    link_type == LinkType::Inline,
                    "only inline image output is supported"
                );
                ensure!(
                    uris.len() < MAX_IMAGE_REFERENCES,
                    "too many generated images"
                );
                ensure!(
                    dest_url
                        .get(..11)
                        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("data:image/")),
                    "external image output is not supported"
                );
                let offset = text[range.clone()]
                    .find(dest_url.as_ref())
                    .context("noncanonical image destination")?;
                let start = range.start + offset;
                admitted_ranges.push(start..start + dest_url.len());
                uris.push(dest_url.into_string());
            }
            Event::Html(html) | Event::InlineHtml(html) => {
                ensure!(
                    !contains_media_html(&html),
                    "HTML image output is not supported"
                );
            }
            _ => {}
        }
    }
    for (offset, _) in text.to_ascii_lowercase().match_indices("data:image/") {
        ensure!(
            admitted_ranges.iter().any(|range| range.contains(&offset)),
            "incomplete or unsupported inline image"
        );
    }
    Ok(uris)
}

fn contains_media_html(html: &str) -> bool {
    let lower = html.to_ascii_lowercase();
    [
        "<img", "<image", "<svg", "<picture", "<object", "<iframe", "<video",
    ]
    .iter()
    .any(|prefix| lower.contains(prefix))
}

fn decode_image(uri: &str) -> Result<VerifiedImage> {
    ensure!(
        uri.len() <= MAX_IMAGE_URI_CHARS,
        "generated image exceeds byte limit"
    );
    let (metadata, encoded) = uri.split_once(',').context("missing image data")?;
    let (mime_type, format) = match metadata.to_ascii_lowercase().as_str() {
        "data:image/png;base64" => ("image/png", ImageFormat::Png),
        "data:image/jpeg;base64" => ("image/jpeg", ImageFormat::Jpeg),
        "data:image/webp;base64" => ("image/webp", ImageFormat::WebP),
        _ => bail!("unsupported generated image format"),
    };
    ensure!(!encoded.is_empty(), "empty generated image");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .context("invalid image base64")?;
    ensure!(
        image::guess_format(&bytes).context("invalid image header")? == format,
        "image MIME type does not match bytes"
    );
    reject_animation_or_multiple_images(format, &bytes)?;
    let mut dimension_reader = ImageReader::with_format(Cursor::new(&bytes), format);
    dimension_reader.limits(image_limits());
    let dimensions = dimension_reader
        .into_dimensions()
        .context("invalid image dimensions")?;
    validate_dimensions(dimensions)?;
    let mut reader = ImageReader::with_format(Cursor::new(&bytes), format);
    reader.limits(image_limits());
    let decoded = reader.decode().context("invalid image pixels")?;
    ensure!(
        (decoded.width(), decoded.height()) == dimensions,
        "inconsistent image dimensions"
    );
    // Different alpha compositing backgrounds can expose different content to
    // a classifier and a browser. The first output-only slice admits opaque
    // pixels so both participants screen the same visible artifact.
    if decoded.color().has_alpha() {
        ensure!(
            decoded.to_rgba8().pixels().all(|pixel| pixel.0[3] == 255),
            "transparent images are unsupported"
        );
    }
    drop(decoded);
    Ok(VerifiedImage {
        data_uri: uri.to_owned(),
        mime_type,
        bytes,
    })
}

fn image_limits() -> Limits {
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIMENSION);
    limits.max_image_height = Some(MAX_IMAGE_DIMENSION);
    limits.max_alloc = Some(MAX_DECODE_BYTES);
    limits
}

fn validate_dimensions((width, height): (u32, u32)) -> Result<()> {
    ensure!(width > 0 && height > 0, "empty image dimensions");
    ensure!(
        width <= MAX_IMAGE_DIMENSION && height <= MAX_IMAGE_DIMENSION,
        "image dimensions exceed limit"
    );
    ensure!(
        u64::from(width) * u64::from(height) <= MAX_IMAGE_PIXELS,
        "image pixel count exceeds limit"
    );
    Ok(())
}

fn reject_animation_or_multiple_images(format: ImageFormat, bytes: &[u8]) -> Result<()> {
    match format {
        ImageFormat::Png => validate_png_chunks(bytes),
        ImageFormat::WebP => validate_webp_chunks(bytes),
        ImageFormat::Jpeg => validate_jpeg_segments(bytes),
        _ => bail!("unsupported image decoder"),
    }
}

fn read_u32(bytes: &[u8], offset: usize, little_endian: bool) -> Result<u32> {
    let value: [u8; 4] = bytes
        .get(offset..offset + 4)
        .context("truncated image chunk")?
        .try_into()?;
    Ok(if little_endian {
        u32::from_le_bytes(value)
    } else {
        u32::from_be_bytes(value)
    })
}

fn validate_png_chunks(bytes: &[u8]) -> Result<()> {
    let mut offset = 8usize;
    while offset < bytes.len() {
        let size = usize::try_from(read_u32(bytes, offset, false)?)?;
        let kind = bytes
            .get(offset + 4..offset + 8)
            .context("truncated PNG chunk")?;
        let end = offset
            .checked_add(size)
            .and_then(|n| n.checked_add(12))
            .context("PNG chunk overflow")?;
        ensure!(end <= bytes.len(), "truncated PNG chunk");
        ensure!(
            !matches!(kind, b"acTL" | b"fcTL" | b"fdAT"),
            "animated PNG is not supported"
        );
        let crc = read_u32(bytes, end - 4, false)?;
        ensure!(
            crc32fast::hash(&bytes[offset + 4..end - 4]) == crc,
            "invalid PNG checksum"
        );
        if kind == b"IEND" {
            ensure!(size == 0 && end == bytes.len(), "invalid PNG termination");
            return Ok(());
        }
        offset = end;
    }
    bail!("missing PNG termination")
}

fn validate_webp_chunks(bytes: &[u8]) -> Result<()> {
    let declared = usize::try_from(read_u32(bytes, 4, true)?)?;
    ensure!(
        declared.checked_add(8) == Some(bytes.len()),
        "invalid WebP length"
    );
    let mut offset = 12usize;
    while offset < bytes.len() {
        let kind = bytes
            .get(offset..offset + 4)
            .context("truncated WebP chunk")?;
        let size = usize::try_from(read_u32(bytes, offset + 4, true)?)?;
        let end = offset
            .checked_add(8)
            .and_then(|n| n.checked_add(size))
            .and_then(|n| n.checked_add(size & 1))
            .context("WebP chunk overflow")?;
        ensure!(end <= bytes.len(), "truncated WebP chunk");
        ensure!(
            !matches!(kind, b"ANIM" | b"ANMF"),
            "animated WebP is not supported"
        );
        if kind == b"VP8X" {
            ensure!(
                size > 0 && bytes[offset + 8] & 0x02 == 0,
                "animated WebP is not supported"
            );
        }
        offset = end;
    }
    ensure!(offset == bytes.len(), "invalid WebP termination");
    Ok(())
}

fn validate_jpeg_segments(bytes: &[u8]) -> Result<()> {
    let mut offset = 2usize;
    while offset < bytes.len() {
        ensure!(bytes[offset] == 0xff, "invalid JPEG marker");
        while bytes.get(offset) == Some(&0xff) {
            offset += 1;
        }
        let marker = *bytes.get(offset).context("truncated JPEG marker")?;
        offset += 1;
        if marker == 0xd9 {
            ensure!(
                offset == bytes.len(),
                "multiple or trailing JPEG images are not supported"
            );
            return Ok(());
        }
        ensure!(
            marker != 0xd8 && marker != 0 && !(0xd0..=0xd7).contains(&marker),
            "invalid JPEG segment"
        );
        if marker == 0x01 {
            continue;
        }
        let length_bytes: [u8; 2] = bytes
            .get(offset..offset + 2)
            .context("truncated JPEG length")?
            .try_into()?;
        let length = usize::from(u16::from_be_bytes(length_bytes));
        ensure!(length >= 2, "invalid JPEG segment length");
        let end = offset
            .checked_add(length)
            .context("JPEG segment overflow")?;
        let segment = bytes
            .get(offset + 2..end)
            .context("truncated JPEG segment")?;
        ensure!(
            marker != 0xe2 || !segment.starts_with(b"MPF\0"),
            "multipicture JPEG is not supported"
        );
        offset = end;
        if marker == 0xda {
            offset = jpeg_entropy_end(bytes, offset)?;
        }
    }
    bail!("missing JPEG termination")
}

fn jpeg_entropy_end(bytes: &[u8], mut offset: usize) -> Result<usize> {
    while offset < bytes.len() {
        if bytes[offset] != 0xff {
            offset += 1;
            continue;
        }
        let marker_start = offset;
        while bytes.get(offset) == Some(&0xff) {
            offset += 1;
        }
        let marker = *bytes.get(offset).context("truncated JPEG scan")?;
        if marker == 0 || (0xd0..=0xd7).contains(&marker) {
            offset += 1;
        } else {
            return Ok(marker_start);
        }
    }
    bail!("incomplete JPEG scan")
}

#[cfg(test)]
mod tests;
