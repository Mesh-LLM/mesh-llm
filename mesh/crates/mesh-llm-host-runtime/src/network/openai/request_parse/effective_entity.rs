//! Extract the entity actually forwarded, independently of HTTP transfer framing.

use std::borrow::Cow;

use anyhow::{Context, Result, bail};

use super::{BufferedHttpRequest, HTTP_READ_LIMITS, body_limits_for_path, http_header_terminator};

impl BufferedHttpRequest {
    /// Read the current forwarding bytes, including any model/body transformations.
    /// Unchunked entities are borrowed; chunked entities use the parser's bounded decoder.
    pub(crate) fn effective_http_entity(&self) -> Result<Cow<'_, [u8]>> {
        let (header_end, _) = http_header_terminator(&self.raw)
            .context("effective request headers are incomplete")?;
        // Ingress already bounded client headers. The host adds correlation
        // headers while forwarding, so parsing that trusted prefix must not
        // impose the original client header-count ceiling a second time.
        let capacity = self.raw[..header_end].split(|byte| *byte == b'\n').count();
        let mut headers = vec![httparse::EMPTY_HEADER; capacity];
        let mut parsed = httparse::Request::new(&mut headers);
        let httparse::Status::Complete(end) = parsed
            .parse(&self.raw[..header_end])
            .context("parse effective request headers")?
        else {
            bail!("effective request headers are incomplete");
        };
        let chunked = parsed.headers.iter().any(|header| {
            header.name.eq_ignore_ascii_case("transfer-encoding")
                && header.value.split(|byte| *byte == b',').any(|coding| {
                    std::str::from_utf8(coding)
                        .is_ok_and(|coding| coding.trim().eq_ignore_ascii_case("chunked"))
                })
        });
        let wire = &self.raw[end..];
        if !chunked {
            // Normalization or object expansion may exceed the original ingress
            // limit. Observation borrows these already-prepared bytes unchanged.
            return Ok(Cow::Borrowed(wire));
        }
        let limit = body_limits_for_path(&self.client_path, HTTP_READ_LIMITS).max_body_bytes;
        let (consumed, body) = super::try_decode_chunked_body(wire, limit)?
            .context("effective chunked request entity is incomplete")?;
        if consumed != wire.len() {
            bail!("effective chunked request contains bytes after its entity");
        }
        Ok(Cow::Owned(body))
    }
}
