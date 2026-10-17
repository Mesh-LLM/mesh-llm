//! Twin texts a client supplies for a pair it marked: the request and both
//! answers, kept only when each is the body this node already sealed.
//!
//! A host sends this plugin digests, not text (`exchange_text`), so the
//! referee has nothing to re-ask with. The client that marked the pair holds
//! the request and both answers; it hands them to its own node through
//! [`SUPPLY_TWIN_TEXTS_OPERATION`]. Nothing is taken on trust. The texts are
//! kept only when:
//!
//! 1. the referee is on (`adjudicate_differing_twins`): off, nothing is kept;
//! 2. this node holds exactly two of its own exchanges with that bracket id;
//! 3. the request's digest is both records' `request_digest`;
//! 4. each answer's digest is one record's `response_digest`, a different
//!    record each;
//! 5. the answers differ: equal digests need no referee, so no text is kept.
//!
//! Then they are kept as `exchange_text` keeps a host's bodies (the same
//! file per exchange, the same modes and retention), and the pair is
//! considered as it would be had the host sent them (`live::consider`).
//! Supplying is the client's choice, per pair: a referee that is asked
//! receives the request and both answers.

use std::path::Path;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::capsule_emit::{digest_json, CapsuleState};
use crate::lifecycle_channel::ExchangeBodies;

/// The operation a client calls on its own node with a pair's texts.
pub const SUPPLY_TWIN_TEXTS_OPERATION: &str = "referee_supply_twin_texts";

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SupplyTwinTextsArgs {
    /// The pair's twin bracket id, as the client marked it.
    pub twin_bracket_id: String,
    /// The request the client sent as both twins (its JSON body).
    pub request_body: Value,
    /// The two answers, in either order (their JSON bodies).
    pub response_bodies: Vec<Value>,
}

/// Why supplied texts were not kept. Nothing is kept in any of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// The referee is off on this node.
    Off,
    /// This node does not hold exactly two of its own exchanges in the bracket.
    NotTwoExchanges,
    /// The request is not the one both records sealed.
    RequestNotSealed,
    /// The answers are not the two the records sealed, one each.
    AnswersNotSealed,
    /// The two answers are identical: no referee is needed.
    AnswersAgree,
}

impl Refused {
    pub fn reason(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::NotTwoExchanges => "not_two_exchanges_in_bracket",
            Self::RequestNotSealed => "request_not_the_sealed_request",
            Self::AnswersNotSealed => "answers_not_the_sealed_answers",
            Self::AnswersAgree => "twins_agree",
        }
    }
}

fn poc_text<'a>(record: &'a Value, pointer: &str) -> Option<&'a str> {
    record
        .pointer("/model_attestation/compute_attestation/x-mesh-poc-v1")?
        .pointer(pointer)?
        .as_str()
        .filter(|s| !s.is_empty())
}

/// One of this node's own exchanges in a bracket: its id and sealed digests.
struct Sealed {
    exchange_id: String,
    request_digest: String,
    response_digest: String,
}

/// This node's own exchanges in `bracket`, the latest record of each, in
/// exchange id order (the order `live::assemble` reads a pair in).
fn own_in_bracket(own: &[Value], bracket: &str) -> Vec<Sealed> {
    let mut found: Vec<Sealed> = Vec::new();
    for record in own.iter().rev() {
        if poc_text(record, "/serving_provenance/twin_bracket_id") != Some(bracket)
            || poc_text(record, "/serving_provenance/role") == Some(super::half::ROLE_PROVIDER)
        {
            continue;
        }
        let (Some(exchange_id), Some(request_digest), Some(response_digest)) = (
            poc_text(record, "/serving_provenance/exchange_id"),
            record
                .pointer("/effect/request_digest")
                .and_then(Value::as_str),
            record
                .pointer("/effect/response_digest")
                .and_then(Value::as_str),
        ) else {
            continue;
        };
        if found.iter().any(|s| s.exchange_id == exchange_id) {
            continue;
        }
        found.push(Sealed {
            exchange_id: exchange_id.to_string(),
            request_digest: request_digest.to_string(),
            response_digest: response_digest.to_string(),
        });
    }
    found.sort_by(|a, b| a.exchange_id.cmp(&b.exchange_id));
    found
}

/// Match supplied texts to this node's two sealed exchanges in `bracket`
/// (`own`: its own records). Returns each exchange id with its answer.
pub fn check(
    own: &[Value],
    bracket: &str,
    request_body: &Value,
    response_bodies: &[Value],
) -> Result<[(String, Value); 2], Refused> {
    let sealed = own_in_bracket(own, bracket);
    let [a, b] = sealed.as_slice() else {
        return Err(Refused::NotTwoExchanges);
    };
    let request = digest_json(request_body).ok();
    if request.as_deref() != Some(a.request_digest.as_str())
        || request.as_deref() != Some(b.request_digest.as_str())
    {
        return Err(Refused::RequestNotSealed);
    }
    if a.response_digest == b.response_digest {
        return Err(Refused::AnswersAgree);
    }
    let [first, second] = response_bodies else {
        return Err(Refused::AnswersNotSealed);
    };
    let (first_digest, second_digest) = (digest_json(first).ok(), digest_json(second).ok());
    let is =
        |body: &Option<String>, s: &Sealed| body.as_deref() == Some(s.response_digest.as_str());
    let pairs = if is(&first_digest, a) && is(&second_digest, b) {
        [(a, first), (b, second)]
    } else if is(&first_digest, b) && is(&second_digest, a) {
        [(a, second), (b, first)]
    } else {
        return Err(Refused::AnswersNotSealed);
    };
    Ok(pairs.map(|(s, body)| (s.exchange_id.clone(), body.clone())))
}

/// Keep matched texts beside the ledger, as `exchange_text` keeps a host's.
fn keep(
    ledger_dir: &Path,
    bracket: &str,
    request_body: &Value,
    matched: &[(String, Value); 2],
) -> std::io::Result<()> {
    for (exchange_id, response_body) in matched {
        let bodies = ExchangeBodies {
            request: Some(request_body.clone()),
            response: Some(response_body.clone()),
        };
        crate::exchange_text::keep(ledger_dir, exchange_id, Some(bracket), &bodies)?;
    }
    Ok(())
}

/// The operation: check, keep, then consider the pair.
pub async fn supply(
    args: SupplyTwinTextsArgs,
    context: &mut mesh_llm_plugin::PluginContext<'_>,
    capsules: Arc<CapsuleState>,
    self_id: Option<String>,
) -> mesh_llm_plugin::PluginResult<Value> {
    let refused = |reason: Refused| json!({ "kept": false, "reason": reason.reason() });
    if !super::request::adjudicate_differing_twins() {
        return Ok(refused(Refused::Off));
    }
    let ledger_dir = capsules.ledger_dir().to_path_buf();
    let own = crate::evidence_panes::read_capsule_records(&ledger_dir);
    let matched = match check(
        &own,
        &args.twin_bracket_id,
        &args.request_body,
        &args.response_bodies,
    ) {
        Ok(matched) => matched,
        Err(reason) => return Ok(refused(reason)),
    };
    keep(
        &ledger_dir,
        &args.twin_bracket_id,
        &args.request_body,
        &matched,
    )
    .map_err(|error| {
        mesh_llm_plugin::PluginError::internal(format!("the texts could not be kept: {error}"))
    })?;
    let Some(self_id) = self_id else {
        return Ok(
            json!({ "kept": true, "considered": "this node does not know its own peer id yet" }),
        );
    };
    let considered =
        match super::live::consider(context, &capsules, &args.twin_bracket_id, &self_id, false)
            .await
        {
            super::live::Considered::Row(row) => json!({ "row": row }),
            super::live::Considered::Deciding => json!("being decided"),
            super::live::Considered::NotRecorded => {
                json!("the call could not be recorded, so no referee was asked")
            }
            super::live::Considered::NotComplete => {
                json!("waiting for both providers' signed halves")
            }
        };
    Ok(json!({ "kept": true, "considered": considered }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(text: &str) -> Value {
        json!({"choices": [{"message": {"role": "assistant", "content": text}}]})
    }

    fn request() -> Value {
        json!({"model": "m", "temperature": 0, "messages": [{"role": "user", "content": "2+2?"}]})
    }

    fn record(exchange: &str, bracket: &str, response: &Value) -> Value {
        json!({
            "effect": {
                "request_digest": digest_json(&request()).unwrap(),
                "response_digest": digest_json(response).unwrap(),
            },
            "model_attestation": {"compute_attestation": {"x-mesh-poc-v1": {"serving_provenance": {
                "exchange_id": exchange, "twin_bracket_id": bracket, "role": "requester",
            }}}}
        })
    }

    fn own() -> Vec<Value> {
        vec![
            record("ex-1", "pair-7", &body("4")),
            record("ex-2", "pair-7", &body("5")),
            record("ex-3", "pair-8", &body("4")),
        ]
    }

    #[test]
    fn texts_that_are_the_sealed_bodies_are_matched_either_way_round() {
        let matched = check(&own(), "pair-7", &request(), &[body("5"), body("4")]).unwrap();
        assert_eq!(matched[0], ("ex-1".to_string(), body("4")));
        assert_eq!(matched[1], ("ex-2".to_string(), body("5")));
    }

    #[test]
    fn a_text_that_is_not_the_sealed_body_is_refused() {
        assert_eq!(
            check(&own(), "pair-7", &request(), &[body("4"), body("five")]).unwrap_err(),
            Refused::AnswersNotSealed
        );
        assert_eq!(
            check(&own(), "pair-7", &request(), &[body("4"), body("4")]).unwrap_err(),
            Refused::AnswersNotSealed,
            "one sealed answer twice"
        );
        assert_eq!(
            check(&own(), "pair-7", &request(), &[body("4")]).unwrap_err(),
            Refused::AnswersNotSealed,
            "one answer"
        );
        let mut other = request();
        other["messages"][0]["content"] = json!("2+3?");
        assert_eq!(
            check(&own(), "pair-7", &other, &[body("4"), body("5")]).unwrap_err(),
            Refused::RequestNotSealed
        );
    }

    #[test]
    fn only_a_bracket_of_exactly_two_own_exchanges_whose_answers_differ() {
        assert_eq!(
            check(&own(), "pair-8", &request(), &[body("4"), body("4")]).unwrap_err(),
            Refused::NotTwoExchanges,
            "one exchange"
        );
        assert_eq!(
            check(&own(), "unknown", &request(), &[body("4"), body("5")]).unwrap_err(),
            Refused::NotTwoExchanges
        );
        let agree = vec![
            record("ex-1", "p", &body("4")),
            record("ex-2", "p", &body("4")),
        ];
        assert_eq!(
            check(&agree, "p", &request(), &[body("4"), body("4")]).unwrap_err(),
            Refused::AnswersAgree
        );
        let mut provider = record("ex-9", "pair-7", &body("6"));
        provider["model_attestation"]["compute_attestation"]["x-mesh-poc-v1"]
            ["serving_provenance"]["role"] = json!(super::super::half::ROLE_PROVIDER);
        let mut with_provider = own();
        with_provider.push(provider);
        assert!(
            check(
                &with_provider,
                "pair-7",
                &request(),
                &[body("4"), body("5")]
            )
            .is_ok(),
            "a provider-side record is not one of this node's twins"
        );
    }

    #[test]
    fn a_re_sealed_exchange_counts_once() {
        let mut own = own();
        own.push(record("ex-1", "pair-7", &body("4")));
        assert!(check(&own, "pair-7", &request(), &[body("4"), body("5")]).is_ok());
    }

    #[test]
    fn matched_texts_are_kept_where_the_live_call_reads_them() {
        let dir = tempfile::tempdir().unwrap();
        let matched = check(&own(), "pair-7", &request(), &[body("4"), body("5")]).unwrap();
        keep(dir.path(), "pair-7", &request(), &matched).unwrap();
        for (exchange, answer) in &matched {
            let kept: Value = serde_json::from_slice(
                &std::fs::read(
                    dir.path()
                        .join(format!("disclosures/by-exchange/{exchange}.json")),
                )
                .unwrap(),
            )
            .unwrap();
            assert_eq!(kept["twin_bracket_id"], "pair-7");
            assert_eq!(kept["request_body"], request());
            assert_eq!(&kept["response_body"], answer);
        }
    }
}
