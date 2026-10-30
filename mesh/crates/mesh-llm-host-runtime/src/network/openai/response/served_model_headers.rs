//! The served-model headers a serving node returns to the mesh peer that
//! asked it to serve: which model it says it ran. Gossip never carries the
//! weights digest (see `protocol/convert.rs`), so these per-response headers
//! are how the routing node learns it. The routing node records them as the
//! peer's own, unverified statement (see `PeerCapsuleIdSink`) and never
//! relays them on to its own client.

#[cfg(feature = "payments")]
use super::probe::try_parse_response_headers;
use super::probe::{
    ResponseProbe, insert_header_before_body, peer_response_header_value, remove_existing_header,
};
use crate::mesh;
use crate::mesh::model_identity::routable_descriptor_for_name;
#[cfg(feature = "payments")]
use crate::network::openai::request_parse::MAX_HEADER_BYTES;

/// The SHA-256 of the GGUF the serving node loaded for the model
/// (`ServedModelIdentity::weights_digest`).
pub(in crate::network::openai) const SERVED_WEIGHTS_DIGEST_HEADER: &str = "x-mesh-weights-digest";

/// The served model's canonical ref (`ServedModelIdentity::canonical_ref`),
/// present only when the model was launched from a Hugging Face or catalog
/// ref; a model launched from a local file has none.
pub(in crate::network::openai) const SERVED_MODEL_CANONICAL_REF_HEADER: &str =
    "x-mesh-model-canonical-ref";

/// The served-model facts one response carries. Each field is a real value
/// or `None`; nothing is inferred.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ServedModelHeaders {
    pub(crate) weights_digest: Option<String>,
    pub(crate) canonical_ref: Option<String>,
}

impl ServedModelHeaders {
    /// This node's facts for `model`, matched under every name routing
    /// accepts for it: the served descriptor's `model_name`, then its public
    /// model ID. `None` when this node has neither fact.
    pub(crate) async fn for_served_model(node: &mesh::Node, model: &str) -> Option<Self> {
        let descriptors = node.served_model_descriptors().await;
        let identity = routable_descriptor_for_name(&descriptors, model)?
            .identity
            .clone();
        let headers = Self {
            weights_digest: identity.weights_digest.filter(|v| is_header_value(v)),
            canonical_ref: identity.canonical_ref.filter(|v| is_header_value(v)),
        };
        (!headers.is_empty()).then_some(headers)
    }

    /// The values a peer sent, read from its untouched response head.
    pub(super) fn read(buf: &[u8], header_end: usize) -> Self {
        let header = |name| peer_response_header_value(buf, header_end, name);
        Self {
            weights_digest: header(SERVED_WEIGHTS_DIGEST_HEADER),
            canonical_ref: header(SERVED_MODEL_CANONICAL_REF_HEADER),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.weights_digest.is_none() && self.canonical_ref.is_none()
    }

    fn entries(&self) -> impl Iterator<Item = (&'static str, &str)> {
        [
            (SERVED_WEIGHTS_DIGEST_HEADER, self.weights_digest.as_deref()),
            (
                SERVED_MODEL_CANONICAL_REF_HEADER,
                self.canonical_ref.as_deref(),
            ),
        ]
        .into_iter()
        .filter_map(|(name, value)| value.map(|value| (name, value)))
    }

    /// Append the present headers to a hand-built response head, for the
    /// response adapters that rebuild it instead of relaying it raw.
    pub(super) fn append_to(&self, header: &mut String) {
        for (name, value) in self.entries() {
            header.push_str(&format!("{name}: {value}\r\n"));
        }
    }
}

/// Visible ASCII and spaces only, so a value never breaks the response head.
fn is_header_value(value: &str) -> bool {
    !value.trim().is_empty() && value.bytes().all(|byte| (0x20..=0x7e).contains(&byte))
}

/// Set the served-model headers on a buffered raw response head: drop any
/// the upstream sent, then splice in `served` on a 2xx response, which is the
/// only kind that served anything. Returns the new `header_end`.
fn set_on_raw_head(
    buf: &mut Vec<u8>,
    mut header_end: usize,
    status_code: u16,
    served: Option<&ServedModelHeaders>,
) -> usize {
    for name in [
        SERVED_WEIGHTS_DIGEST_HEADER,
        SERVED_MODEL_CANONICAL_REF_HEADER,
    ] {
        header_end -= remove_existing_header(buf, header_end, name);
    }
    let Some(served) = served.filter(|_| (200..300).contains(&status_code)) else {
        return header_end;
    };
    for (name, value) in served.entries() {
        let delta = insert_header_before_body(buf, header_end, name, value);
        header_end = header_end.saturating_add_signed(delta);
    }
    header_end
}

/// Set the served-model headers on a probed response before any relay path
/// reads the probe. Raw relays forward the head as is; response adapters
/// that rebuild it echo the headers from `ParsedResponseHeaders::served_model`.
/// With `None` the headers are only removed, so a peer's values are never
/// relayed on to this node's own client.
pub(super) fn set_served_model_headers(
    probe: &mut ResponseProbe,
    served: Option<&ServedModelHeaders>,
) {
    probe.header_end = set_on_raw_head(
        &mut probe.buffered,
        probe.header_end,
        probe.status_code,
        served,
    );
}

/// Sets the served-model headers on a raw HTTP response that is forwarded
/// chunk by chunk without a probe: the paid seller's backend stream. It holds
/// bytes back only until the response head is complete. A head that never
/// parses, or outgrows `MAX_HEADER_BYTES`, is forwarded unchanged.
#[cfg(feature = "payments")]
pub(crate) struct ServedModelHeadRewrite {
    served: Option<ServedModelHeaders>,
    /// The head bytes held back so far; `None` once the head has been sent.
    pending: Option<Vec<u8>>,
}

#[cfg(feature = "payments")]
impl ServedModelHeadRewrite {
    pub(crate) fn new(served: Option<ServedModelHeaders>) -> Self {
        Self {
            served,
            pending: Some(Vec::new()),
        }
    }

    /// Take the next chunk read from the backend; returns the bytes to
    /// forward now, which may be empty while the head is incomplete.
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Vec<u8> {
        let Some(pending) = self.pending.as_mut() else {
            return chunk.to_vec();
        };
        pending.extend_from_slice(chunk);
        match try_parse_response_headers(pending) {
            Ok(Some(parsed)) => {
                let mut head = self.pending.take().unwrap_or_default();
                set_on_raw_head(
                    &mut head,
                    parsed.header_end,
                    parsed.status_code,
                    self.served.as_ref(),
                );
                head
            }
            Ok(None) if pending.len() <= MAX_HEADER_BYTES => Vec::new(),
            _ => self.pending.take().unwrap_or_default(),
        }
    }

    /// The bytes still held back when the backend ended mid-head.
    pub(crate) fn finish(&mut self) -> Vec<u8> {
        self.pending.take().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn served() -> ServedModelHeaders {
        ServedModelHeaders {
            weights_digest: Some("74a4da8c".into()),
            canonical_ref: Some("org/repo@main:Q4_K_M".into()),
        }
    }

    const OK: &[u8] =
        b"HTTP/1.1 200 OK\r\nx-mesh-weights-digest: upstream\r\nContent-Length: 2\r\n\r\n{}";

    #[cfg(feature = "payments")]
    #[test]
    fn a_rewrite_replaces_upstream_values_with_this_nodes_on_a_2xx_head() {
        let mut rewrite = ServedModelHeadRewrite::new(Some(served()));
        // The head arrives split across two chunks.
        assert!(rewrite.push(&OK[..20]).is_empty());
        let out = rewrite.push(&OK[20..]);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("x-mesh-weights-digest: 74a4da8c\r\n"));
        assert!(text.contains("x-mesh-model-canonical-ref: org/repo@main:Q4_K_M\r\n"));
        assert!(!text.contains("upstream"));
        assert!(text.ends_with("\r\n\r\n{}"));
        // The body after the head passes through untouched.
        assert_eq!(rewrite.push(b"tail"), b"tail");
        assert!(rewrite.finish().is_empty());
    }

    #[cfg(feature = "payments")]
    #[test]
    fn a_rewrite_only_removes_the_headers_without_values_or_on_an_error() {
        let mut none = ServedModelHeadRewrite::new(None);
        let text = String::from_utf8(none.push(OK)).unwrap();
        assert!(!text.contains("x-mesh-weights-digest"));

        let error = b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n";
        let mut failed = ServedModelHeadRewrite::new(Some(served()));
        let text = String::from_utf8(failed.push(error)).unwrap();
        assert!(!text.contains("x-mesh-"));
    }

    #[cfg(feature = "payments")]
    #[test]
    fn a_rewrite_forwards_an_unparseable_or_unfinished_head_unchanged() {
        let mut garbage = ServedModelHeadRewrite::new(Some(served()));
        assert_eq!(garbage.push(b"not http\r\n\r\n"), b"not http\r\n\r\n");

        let mut cut = ServedModelHeadRewrite::new(Some(served()));
        assert!(cut.push(b"HTTP/1.1 200 OK\r\n").is_empty());
        assert_eq!(cut.finish(), b"HTTP/1.1 200 OK\r\n");
    }

    #[test]
    fn values_that_would_break_the_head_are_not_sent() {
        assert!(is_header_value("org/repo@main:Q4_K_M"));
        assert!(!is_header_value(""));
        assert!(!is_header_value("a\r\nx-evil: 1"));
        assert!(!is_header_value("caf\u{e9}"));
    }

    #[test]
    fn adapters_append_only_the_present_headers() {
        let mut header = String::new();
        ServedModelHeaders {
            weights_digest: None,
            canonical_ref: Some("org/repo".into()),
        }
        .append_to(&mut header);
        assert_eq!(header, "x-mesh-model-canonical-ref: org/repo\r\n");
    }
}
