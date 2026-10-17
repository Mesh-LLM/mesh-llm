//! One twin's half of a comparison: its signed record and the request and
//! answer it covers, and the checks a half must pass before its answer is
//! read at all.

use serde_json::{json, Value};

/// `serving_provenance.role` of a provider's own record: a provider keeps no
/// answer body, so its half alone has no answer to compare.
pub const ROLE_PROVIDER: &str = "provider";

/// A half that cannot be read: refused, never ruled on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfError {
    /// The record fails its own checks (changed after sealing).
    Forged,
    /// The answer is not the body the record's `response_digest` covers, or a
    /// separately supplied text is not that body's text.
    AnswerNotTheSignedBody,
}

/// One twin's half: its signed record, the request it answered and the
/// answer it served, and who served it on which weights.
#[derive(Debug, Clone)]
pub struct Half {
    pub capsule: Value,
    /// The request it answered; `{}` when none was disclosed.
    pub request_body: Value,
    /// The answer it served; `None` when this node holds none.
    pub response_body: Option<Value>,
    /// A text supplied beside the answer body; read only when the body has
    /// no text of its own, and refused when it differs from it.
    pub response_text: Option<String>,
    /// The node that served it.
    pub node_id: Option<String>,
    /// The weights it was served on, as its record says; `None` when unknown.
    pub weights_digest: Option<String>,
}

fn poc(record: &Value) -> Option<&Value> {
    record.pointer("/model_attestation/compute_attestation/x-mesh-poc-v1")
}

/// A non-empty string member at `pointer`.
fn text_at<'a>(record: &'a Value, pointer: &str) -> Option<&'a str> {
    record
        .pointer(pointer)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// The node a record names as its server (`serving_provenance.served_by_node_id`).
pub fn served_by(record: &Value) -> Option<&str> {
    poc(record).and_then(|poc| text_at(poc, "/serving_provenance/served_by_node_id"))
}

/// The weights a record says it was served on (`serving_provenance.model.weights_digest`).
pub fn weights_digest(record: &Value) -> Option<&str> {
    poc(record).and_then(|poc| text_at(poc, "/serving_provenance/model/weights_digest"))
}

/// The model a record says it served (`serving_provenance.model.identity_hash`).
pub fn model_hash(record: &Value) -> Option<&str> {
    poc(record).and_then(|poc| text_at(poc, "/serving_provenance/model/identity_hash"))
}

/// The record passes its own checks (its structure, and its `capsule_id`
/// recomputes).
pub fn record_checks(record: &Value) -> bool {
    record.get("capsule_id").and_then(Value::as_str).is_some()
        && capsule_emit_lib::structure::check_structure(record).ok()
}

/// The key that signed `record`'s producer envelope, when it verifies.
pub fn signer(record: &Value) -> Option<String> {
    capsule_emit_lib::cose::verify_producer_envelope(record).ok()
}

/// A temperature that says "sampled": a number, or a string that reads as
/// one, above 0. A boolean, or anything unreadable, says nothing.
fn above_zero(value: Option<&Value>) -> bool {
    match value {
        Some(Value::Number(n)) => n.as_f64().is_some_and(|t| t > 0.0),
        Some(Value::String(s)) => s.trim().parse::<f64>().is_ok_and(|t| t > 0.0),
        _ => false,
    }
}

impl Half {
    pub fn capsule_id(&self) -> &str {
        self.capsule
            .get("capsule_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
    }

    /// The text the signed body carries (`choices[0].message.content`).
    fn body_text(&self) -> Option<&str> {
        self.response_body
            .as_ref()?
            .pointer("/choices/0/message/content")?
            .as_str()
    }

    /// The text this half is compared on: the signed body's own text, never
    /// a separately supplied one; that is read only when the body has none.
    pub fn text(&self) -> &str {
        self.body_text()
            .or(self.response_text.as_deref())
            .unwrap_or_default()
    }

    /// The record passes its own checks, else [`HalfError::Forged`].
    pub fn check_record(&self) -> Result<(), HalfError> {
        if record_checks(&self.capsule) {
            Ok(())
        } else {
            Err(HalfError::Forged)
        }
    }

    /// A provider's own record with no answer held: there is nothing of this
    /// half to compare.
    pub fn no_answer_held(&self) -> bool {
        let role = poc(&self.capsule).and_then(|poc| text_at(poc, "/serving_provenance/role"));
        role == Some(ROLE_PROVIDER)
            && !self
                .response_body
                .as_ref()
                .is_some_and(|body| body.as_object().is_some_and(|o| !o.is_empty()))
    }

    /// The answer is the body the record's `response_digest` covers, and any
    /// text supplied beside it is that body's text.
    pub fn check_answer(&self) -> Result<(), HalfError> {
        let declared = text_at(&self.capsule, "/effect/response_digest");
        let body = self.response_body.clone().unwrap_or_else(|| json!({}));
        let recomputed = crate::capsule_emit::digest_json(&body).ok();
        if declared.is_none() || recomputed.as_deref() != declared {
            return Err(HalfError::AnswerNotTheSignedBody);
        }
        if let (Some(body_text), Some(supplied)) = (self.body_text(), &self.response_text) {
            if body_text != supplied {
                return Err(HalfError::AnswerNotTheSignedBody);
            }
        }
        Ok(())
    }

    /// Sampled (temperature above 0) as its record's `decoding` or the host's
    /// `generation_parameters` say, or as the request asked. Any one is
    /// enough; an absent temperature says nothing.
    pub fn sampled(&self) -> bool {
        let attestation = self
            .capsule
            .pointer("/model_attestation/compute_attestation");
        let at = |pointer: &str| attestation.and_then(|a| a.pointer(pointer));
        above_zero(at("/decoding/temperature"))
            || above_zero(at("/x-mesh-poc-v1/generation_parameters/temperature"))
            || above_zero(self.request_body.get("temperature"))
    }

    /// Whether the request carried chat messages (a re-answer of it can
    /// repeat the agreed words).
    pub fn request_has_messages(&self) -> bool {
        match self.request_body.get("messages") {
            Some(Value::Array(items)) => !items.is_empty(),
            Some(Value::Object(map)) => !map.is_empty(),
            Some(Value::String(s)) => !s.is_empty(),
            Some(Value::Bool(b)) => *b,
            Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
            _ => false,
        }
    }
}
