use crate::mesh::{self, ServedModelDescriptor};
use crate::network::openai::ingress::parse_model_with_profile;
use crate::network::openai::model_names::public_model_id;
use crate::network::openai::routing_rank::descriptor_for_model;
use mesh_llm_payments_types::pricing::Pricing;
use std::collections::BTreeMap;

/// Require payment when an offer matches the served name or its public identity.
pub(super) async fn requires_payment(
    node: &mesh::Node,
    model: Option<&str>,
    prices: &BTreeMap<String, Pricing>,
) -> bool {
    if prices.is_empty() {
        return false;
    }
    let Some(model) = model else {
        return false;
    };
    let descriptors = node.all_served_model_descriptors().await;
    let aliases = public_identities_for_model(model, &descriptors);
    priced_model_requires_payment(Some(model), prices, &aliases)
}

/// Resolve the public spellings ingress can rewrite to the admitted served model.
fn public_identities_for_model(model: &str, descriptors: &[ServedModelDescriptor]) -> Vec<String> {
    let (base, profile) = parse_model_with_profile(model);
    let descriptor = descriptor_for_model(descriptors, base);
    let mut identities = vec![public_model_id(base, descriptor, profile)];
    if !profile.is_empty() {
        identities.push(public_model_id(base, descriptor, ""));
    }
    identities.retain(|identity| identity != model);
    identities.dedup();
    identities
}

/// Match either the request's current model key or an alias resolved from it.
fn priced_model_requires_payment(
    model: Option<&str>,
    prices: &BTreeMap<String, Pricing>,
    aliases: &[String],
) -> bool {
    let Some(model) = model else {
        return false;
    };
    prices.contains_key(model) || aliases.iter().any(|alias| prices.contains_key(alias))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::openai::request_normalize::ResponseAdapter;
    use crate::network::openai::request_parse::{
        BufferedHttpRequest, parse_json_body_from_http_request, rewrite_public_model_alias,
    };
    use mesh_llm_events::logging::identifiers::RequestId;

    /// Build a served descriptor with a distinct public Hugging Face ID.
    fn public_alias_descriptor() -> mesh::ServedModelDescriptor {
        mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: "Falcon-H1-1.5B-Instruct-Q4_K_M".to_string(),
                source_kind: mesh::ModelSourceKind::HuggingFace,
                repository: Some("tiiuae/Falcon-H1-1.5B-Instruct-GGUF".to_string()),
                revision: Some("0d3a6cfe25fb4eeab0153fb8623aac5b69d6bd0a".to_string()),
                artifact: Some("Falcon-H1-1.5B-Instruct-Q4_K_M.gguf".to_string()),
                canonical_ref: Some(
                    "tiiuae/Falcon-H1-1.5B-Instruct-GGUF@0d3a6cfe25fb4eeab0153fb8623aac5b69d6bd0a/Falcon-H1-1.5B-Instruct-Q4_K_M.gguf"
                        .to_string(),
                ),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create a minimal buffered completion request for alias rewriting.
    fn buffered_request(model: &str) -> BufferedHttpRequest {
        let body = serde_json::json!({"model": model, "prompt": "Hi"});
        let bytes = serde_json::to_vec(&body).unwrap();
        let mut raw = format!(
            "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            bytes.len()
        )
        .into_bytes();
        raw.extend_from_slice(&bytes);
        BufferedHttpRequest {
            raw,
            method: "POST".into(),
            path: "/v1/completions".into(),
            client_path: "/v1/completions".into(),
            request_id: RequestId::default(),
            body_json: Some(body),
            body_json_attempted: true,
            body_len_bytes: bytes.len(),
            body_bytes: Some(bytes),
            completion_tokens: None,
            model_name: Some(model.to_string()),
            stream: None,
            request_object_request_ids: Vec::new(),
            response_adapter: ResponseAdapter::None,
            correlation_id: None,
            exchange_observation_id: None,
        }
    }

    /// Price a model at a minimal nonzero rate for admission tests.
    fn priced_offer(model: &str) -> BTreeMap<String, Pricing> {
        BTreeMap::from([(
            model.to_string(),
            Pricing {
                input_msat_per_million: 1,
                output_msat_per_million: 1,
                cache_discount_percent: 0,
            },
        )])
    }

    /// The exact pricing key still requires payment when raw chunked JSON cannot be reparsed.
    #[test]
    fn decoded_chunked_model_cannot_bypass_seller_payment_admission() {
        const MODEL: &str = "paid-model";
        const BODY: &str = r#"{"model":"paid-model","prompt":"Hello"}"#;
        let raw = format!(
            "POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{BODY}\r\n0\r\n\r\n",
            BODY.len(),
        );
        assert!(
            parse_json_body_from_http_request(raw.as_bytes()).is_none(),
            "raw JSON reparsing must demonstrate the chunk-framing bypass"
        );
        let prices = priced_offer(MODEL);
        assert!(priced_model_requires_payment(Some(MODEL), &prices, &[]));
    }

    /// #2246: ingress rewrites the public ID before seller payment admission.
    #[test]
    fn priced_public_alias_still_requires_payment_after_ingress_alias_rewrite() {
        let descriptor = public_alias_descriptor();
        let internal = descriptor.identity.model_name.clone();
        let public = mesh::public_model_id_from_identity(&descriptor.identity).unwrap();
        assert_ne!(
            public, internal,
            "the fixture must model a public alias distinct from the internal served name"
        );
        let descriptors = vec![descriptor];
        let served = vec![internal.clone()];
        let mut request = buffered_request(&public);

        rewrite_public_model_alias(&mut request, &served, &descriptors);
        assert_eq!(
            request.model_name.as_deref(),
            Some(internal.as_str()),
            "ingress must rewrite the advertised public alias to the internal served name"
        );

        let prices = priced_offer(&public);
        let aliases =
            public_identities_for_model(request.model_name.as_deref().unwrap(), &descriptors);
        assert!(
            priced_model_requires_payment(request.model_name.as_deref(), &prices, &aliases),
            "a model priced under its advertised public ID {public} must require payment even \
             though ingress rewrote the request to the internal served name {internal}"
        );
    }

    /// #2246: an unprofiled public offer covers ingress fallback to a profiled served model.
    #[test]
    fn unprofiled_public_alias_requires_payment_after_profile_fallback() {
        let descriptor = public_alias_descriptor();
        let public = mesh::public_model_id_from_identity(&descriptor.identity).unwrap();
        let internal = format!("{}#low-ctx", descriptor.identity.model_name);
        let descriptors = vec![descriptor];
        let served = vec![internal.clone()];
        let mut request = buffered_request(&public);

        rewrite_public_model_alias(&mut request, &served, &descriptors);
        assert_eq!(request.model_name.as_deref(), Some(internal.as_str()));

        let prices = priced_offer(&public);
        let aliases =
            public_identities_for_model(request.model_name.as_deref().unwrap(), &descriptors);
        assert!(aliases.contains(&public));
        assert!(
            priced_model_requires_payment(request.model_name.as_deref(), &prices, &aliases),
            "an offer for the unprofiled public ID {public} must cover internal model {internal}"
        );
    }
}
