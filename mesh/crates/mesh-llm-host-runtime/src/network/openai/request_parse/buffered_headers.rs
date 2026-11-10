//! Reading headers back off a request this node already buffered.
//!
//! Ingress bounds what a client sends, then adds headers of its own to the
//! request it keeps: the canonical request id, the capsule nonce,
//! `Connection`, the lifecycle marker. Routing reads that kept request again
//! for the headers that steer it (`x-mesh-target`, `x-mesh-exclude`, the
//! trust policy), so a head ingress accepted must always read back, and a
//! head that does not is an error, never "no such header".

/// The head of an already-buffered request: its request line and headers,
/// through the empty line that ends them.
fn buffered_head(raw: &[u8]) -> &[u8] {
    let header_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .unwrap_or(raw.len());
    &raw[..header_end.saturating_add(4).min(raw.len())]
}

/// Room for every header of an already-buffered head.
///
/// A request that arrives with all the headers a client may send
/// ([`super::MAX_HEADERS`]) has more once ingress has added its own. Reading
/// it back with room for the client's count only would fail, so that
/// ceiling is not applied a second time: a head has no more headers than it
/// has lines.
pub(super) fn buffered_head_header_capacity(head: &[u8]) -> usize {
    head.split(|byte| *byte == b'\n').count()
}

/// Why a header could not be read back off an already-buffered request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeaderReadError {
    /// The head does not parse, so whether the header is there is unknown.
    /// Never the same as "the header is absent".
    UnreadableHead,
    /// An occurrence of the header has non-UTF-8 bytes.
    InvalidUtf8,
}

impl HeaderReadError {
    /// What a client is told when header `name` could not be read.
    pub(crate) fn describe(self, name: &str) -> String {
        match self {
            Self::UnreadableHead => {
                format!("the request head could not be read back to check {name}")
            }
            Self::InvalidUtf8 => format!("{name} header contains invalid UTF-8"),
        }
    }
}

/// Every value of a given header name, read back off an already-rebuilt raw
/// HTTP request. Only the request-header block is scanned. Order matches the
/// wire order; duplicates are returned as separate entries. An error means
/// the header may be there and could not be read: the head does not parse,
/// or an occurrence of `name` has non-UTF-8 bytes. The caller must reject
/// the request rather than carry on as if the header were absent.
pub(crate) fn header_values_from_raw(
    raw: &[u8],
    name: &str,
) -> Result<Vec<String>, HeaderReadError> {
    untrimmed_header_values_from_raw(raw, name).map(|values| {
        values
            .iter()
            .map(|value| value.trim().to_string())
            .collect()
    })
}

/// Every value of header `name`, as UTF-8, exactly as httparse returned it.
pub(super) fn untrimmed_header_values_from_raw(
    raw: &[u8],
    name: &str,
) -> Result<Vec<String>, HeaderReadError> {
    let head = buffered_head(raw);
    let mut headers_buf = vec![httparse::EMPTY_HEADER; buffered_head_header_capacity(head)];
    let mut req = httparse::Request::new(&mut headers_buf);
    if !matches!(req.parse(head), Ok(httparse::Status::Complete(_))) {
        return Err(HeaderReadError::UnreadableHead);
    }
    req.headers
        .iter()
        .filter(|header| header.name.eq_ignore_ascii_case(name))
        .map(|header| {
            std::str::from_utf8(header.value)
                .map(str::to_string)
                .map_err(|_| HeaderReadError::InvalidUtf8)
        })
        .collect()
}

/// Read the stabilized `x-capsule-client-nonce` / `x-capsule-nonce-origin`
/// headers back out of an already-rebuilt raw HTTP request.
///
/// Only the request-header block is scanned. Used by request-rebuilding proxy
/// paths (pipeline / MoA) that would otherwise emit an outbound request with no
/// nonce because they construct it from a parsed JSON body rather than
/// forwarding `raw`.
pub(super) fn capsule_nonce_headers_from_raw(raw: &[u8]) -> (Option<String>, Option<String>) {
    let head = buffered_head(raw);
    let mut headers_buf = vec![httparse::EMPTY_HEADER; buffered_head_header_capacity(head)];
    let mut req = httparse::Request::new(&mut headers_buf);
    if req.parse(head).is_err() {
        return (None, None);
    }
    let nonce_header = skippy_inference_api::lifecycle::CLIENT_NONCE_HEADER.as_str();
    let origin_header = skippy_inference_api::lifecycle::CLIENT_NONCE_ORIGIN_HEADER.as_str();
    let find = |name: &str| {
        req.headers
            .iter()
            .find(|header| header.name.eq_ignore_ascii_case(name))
            .and_then(|header| std::str::from_utf8(header.value).ok())
            .map(str::to_string)
    };
    (find(nonce_header), find(origin_header))
}

#[cfg(test)]
mod tests {
    use super::super::{BufferedHttpRequest, MAX_HEADERS, read_http_request, rewrite_model_field};
    use super::*;
    use tokio::io::AsyncWriteExt;

    /// A chat request with as many headers as a client may send, `steering`
    /// among them.
    fn full_request(steering: &[(&str, &str)]) -> Vec<u8> {
        let body = r#"{"model":"m","messages":[]}"#;
        let length = body.len().to_string();
        let mut headers = vec![
            ("Host", "t"),
            ("Content-Type", "application/json"),
            ("Content-Length", length.as_str()),
        ];
        headers.extend_from_slice(steering);
        let mut head = String::from("POST /v1/chat/completions HTTP/1.1\r\n");
        for (name, value) in &headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        for filler in headers.len()..MAX_HEADERS {
            head.push_str(&format!("x-filler-{filler}: v\r\n"));
        }
        format!("{head}\r\n{body}").into_bytes()
    }

    /// `raw` as ingress keeps it: read off the wire and rebuilt.
    async fn ingested(raw: Vec<u8>) -> BufferedHttpRequest {
        let (mut client, mut server) = tokio::io::duplex(128 * 1024);
        client.write_all(&raw).await.expect("write request");
        read_http_request(&mut server)
            .await
            .expect("a request with 64 headers is accepted")
    }

    fn header_count(raw: &[u8]) -> usize {
        let head = buffered_head(raw);
        // the request line and the empty line are not headers
        head.split(|byte| *byte == b'\n').count() - 3
    }

    #[tokio::test]
    async fn a_request_with_every_header_a_client_may_send_keeps_the_ones_that_steer_it() {
        let request = ingested(full_request(&[
            ("x-mesh-target", "aabb"),
            ("x-mesh-exclude", "ccdd"),
            ("x-mesh-trust", "receipts"),
            ("x-mesh-twin-bracket", "b1"),
        ]))
        .await;
        assert!(
            header_count(&request.raw) > MAX_HEADERS,
            "ingress adds headers of its own to the request it keeps"
        );

        let (target, exclude) = request.mesh_routing_header_values().unwrap();
        assert_eq!(target, ["aabb"]);
        assert_eq!(exclude, ["ccdd"]);
        assert_eq!(request.twin_bracket_header_values().unwrap(), ["b1"]);
        assert_eq!(
            header_values_from_raw(&request.raw, "x-mesh-trust"),
            Ok(vec!["receipts".to_owned()])
        );
        let (nonce, _origin) = request.capsule_nonce_headers();
        assert!(nonce.is_some(), "the stabilized nonce reads back too");
    }

    #[tokio::test]
    async fn the_model_of_a_request_with_every_header_is_still_rewritten() {
        let mut request = ingested(full_request(&[])).await;
        rewrite_model_field(&mut request, "other");
        let body_start = buffered_head(&request.raw).len();
        let body: serde_json::Value =
            serde_json::from_slice(&request.raw[body_start..]).expect("a JSON body");
        assert_eq!(body["model"], "other");
    }

    #[test]
    fn a_head_that_does_not_read_back_is_an_error_not_an_absent_header() {
        for unreadable in [
            &b"not a request\r\n\r\n"[..],
            // cut off before the empty line that ends the head
            &b"POST / HTTP/1.1\r\nx-mesh-trust: receipts\r\n"[..],
        ] {
            assert_eq!(
                header_values_from_raw(unreadable, "x-mesh-trust"),
                Err(HeaderReadError::UnreadableHead)
            );
        }
        let readable = b"POST / HTTP/1.1\r\nHost: t\r\n\r\n";
        assert_eq!(header_values_from_raw(readable, "x-mesh-trust"), Ok(vec![]));
        let mut invalid = b"POST / HTTP/1.1\r\nx-mesh-trust: ".to_vec();
        invalid.extend_from_slice(&[0xff, 0xfe]);
        invalid.extend_from_slice(b"\r\n\r\n");
        assert_eq!(
            header_values_from_raw(&invalid, "x-mesh-trust"),
            Err(HeaderReadError::InvalidUtf8)
        );
    }
}
