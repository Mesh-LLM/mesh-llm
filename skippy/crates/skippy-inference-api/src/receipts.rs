//! What a request asks an inference receipt to bind: the request headers
//! that name the client's key and the parent turn, and how they are read.
//!
//! The frontend only carries the binding to the backend
//! ([`crate::InferenceLifecycleContext::receipt`],
//! [`crate::InferenceRequestContext::receipt`]); a serving node that issues
//! receipts commits the turn and puts the receipt in the response's
//! `receipt` field.

use axum::http::{HeaderMap, header::HeaderName};

/// The client's VRF key for an inference receipt, 64 hex digits. A serving
/// node that issues receipts binds it into the turn, so the client is bound
/// to its key before it sees what it will challenge.
pub static RECEIPT_CLIENT_KEY_HEADER: HeaderName = HeaderName::from_static("x-receipt-client-key");
/// The turn root of the receipt this request continues, 64 hex digits.
pub static RECEIPT_PARENT_HEADER: HeaderName = HeaderName::from_static("x-receipt-parent");
/// The seed the client asks a sampled turn to draw with, a decimal `u32` above
/// 0. A serving node that issues receipts uses it when the request body sets
/// no seed of its own, and states the seed it used, so the client can replay
/// the draw with a seed the node did not choose. Unlike a seed in the body,
/// it does not turn sampling on for a request that would decode greedily.
pub static RECEIPT_SEED_HEADER: HeaderName = HeaderName::from_static("x-receipt-seed");

/// What a request asks a receipt to bind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReceiptRequest {
    pub client_key: [u8; 32],
    pub parent: Option<[u8; 32]>,
    /// The seed the client asks a sampled turn to draw with.
    pub seed: Option<u32>,
}

impl ReceiptRequest {
    /// The receipt binding a request asks for: a well-formed client key, and
    /// the parent turn root when it names a well-formed one.
    pub fn from_headers(headers: &HeaderMap) -> Option<Self> {
        let key = |name: &HeaderName| {
            let text = headers.get(name)?.to_str().ok()?;
            let bytes = hex::decode(text.trim()).ok()?;
            <[u8; 32]>::try_from(bytes.as_slice()).ok()
        };
        Some(Self {
            client_key: key(&RECEIPT_CLIENT_KEY_HEADER)?,
            parent: key(&RECEIPT_PARENT_HEADER),
            seed: headers
                .get(&RECEIPT_SEED_HEADER)
                .and_then(|value| value.to_str().ok())
                .and_then(|text| text.trim().parse::<u32>().ok())
                .filter(|seed| *seed != 0),
        })
    }
}

/// The responses and stream events that carry a turn's receipt in their
/// `receipt` field.
macro_rules! carries_receipt {
    ($($response:ty),+ $(,)?) => {$(
        impl $response {
            /// Carries the turn's inference receipt, when there is one.
            pub fn with_receipt(mut self, receipt: Option<serde_json::Value>) -> Self {
                self.receipt = receipt;
                self
            }
        }
    )+};
}

carries_receipt!(
    crate::chat::ChatCompletionResponse,
    crate::chat::ChatCompletionChunk,
    crate::completions::CompletionResponse,
    crate::completions::CompletionChunk,
    crate::generation::GenerationResponse,
    crate::generation::GenerationEvent,
);

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::chat::{ChatCompletionChunk, ChatCompletionResponse};
    use crate::common::Usage;

    #[test]
    fn the_receipt_request_reads_its_headers() {
        use axum::http::{HeaderMap, HeaderValue};

        let key = "ab".repeat(32);
        let mut headers = HeaderMap::new();
        assert_eq!(super::ReceiptRequest::from_headers(&headers), None);
        headers.insert(
            &super::RECEIPT_CLIENT_KEY_HEADER,
            HeaderValue::from_str(&key).unwrap(),
        );
        let plain = super::ReceiptRequest::from_headers(&headers).expect("a request");
        assert_eq!(plain.client_key, [0xab; 32]);
        assert_eq!((plain.parent, plain.seed), (None, None));
        // a seed is a decimal u32 above 0; anything else names none
        for (text, seed) in [
            ("77", Some(77)),
            (" 4294967295 ", Some(u32::MAX)),
            ("0", None),
            ("4294967296", None),
            ("-1", None),
            ("0x10", None),
        ] {
            headers.insert(
                &super::RECEIPT_SEED_HEADER,
                HeaderValue::from_str(text).unwrap(),
            );
            let request = super::ReceiptRequest::from_headers(&headers).expect("a request");
            assert_eq!(request.seed, seed, "{text:?}");
        }
        // a seed without a key asks for no receipt
        headers.remove(&super::RECEIPT_CLIENT_KEY_HEADER);
        assert_eq!(super::ReceiptRequest::from_headers(&headers), None);
    }

    #[test]
    fn the_receipt_rides_only_when_there_is_one() {
        let done = ChatCompletionChunk::done("model");
        assert!(
            serde_json::to_value(&done)
                .unwrap()
                .get("receipt")
                .is_none()
        );
        let done = done.with_receipt(Some(json!({"receipt_id": "r"})));
        assert_eq!(
            serde_json::to_value(done).unwrap()["receipt"]["receipt_id"],
            "r"
        );
        let response = ChatCompletionResponse::new("model", "text", Usage::new(1, 1))
            .with_receipt(Some(json!({"receipt_id": "r"})));
        assert_eq!(
            serde_json::to_value(response).unwrap()["receipt"]["receipt_id"],
            "r"
        );
    }
}
